"""Durable local loop. Every GPU operation goes through an admitted finite segment."""
import time
import os
from pathlib import Path
from filelock import FileLock
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.contracts import CheckpointRef, EpochEvent, TaskSpec
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_trials
from mas_demo.control.verification import verify_round
from mas_demo.data import prepare_data
from mas_demo.execution.local import LocalBackend
from mas_demo.io import append_event, read_json, write_json
from mas_demo.memory.store import MemoryStore, settings_context
from mas_demo.observation.collector import build_evidence
from mas_demo.training.worker import resolve_device


def run(spec, directory, resume=False, backend=None, gateway=None):
    if gateway is None and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("OPENROUTER_API_KEY is required; all agents use the LLM API")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory/"run.lock"), timeout=0):
        return _run(spec, directory, resume, backend or LocalBackend(), gateway)


def _run(spec, directory, resume, backend, gateway=None):
    state_path = directory/"state.json"
    log = directory/"events.jsonl"
    memory = MemoryStore(directory/"memory.sqlite")
    resource = resolve_device(spec.device)
    if state_path.exists():
        if not resume:
            raise ValueError("Run exists; use resume or choose a new output directory")
        saved = read_json(directory/"config.json")
        if TaskSpec.model_validate(saved).model_dump(mode="json") != spec.model_dump(mode="json"):
            raise ValueError("Resume config differs from saved config")
        state = read_json(state_path)
        if state["status"] == "COMPLETED":
            return state
        if state.get("phase") != "idle":
            # A crash may have killed a worker after GPU execution but before cost commit.
            # Fail closed; no automatic retry or fabricated exact cost.
            state.update(status="INTERRUPTED", reason="Uncommitted work: inspect worker results/costs before creating a continuation run")
            write_json(state_path, state)
            return state
        if state.get("schema_version", 1) < 2:
            migrated = memory.migrate_legacy_contexts(log, spec.dataset)
            append_event(log, "memory_migration", copied_outcomes=migrated)
            state["schema_version"] = 2
    else:
        manifest = prepare_data(spec)
        write_json(directory/"config.json", spec)
        write_json(directory/"dataset_manifest.json", manifest)
        state = {"schema_version": 2, "status": "RUNNING", "phase": "idle", "epoch": 0, "round": 0,
                 "checkpoint": None, "best_checkpoint": None, "best_accuracy": -1,
                 "history": [], "budget": BudgetLedger(spec.budget_seconds).snapshot(),
                 "resource_type": resource, "gpu_hours": None, "origin": manifest.origin,
                 "analysis_mode": spec.analysis_mode, "reason": "",
                 "cost_measure": "worker wall-time upper bound for allocated resource time; includes startup/teardown"}
    ledger = BudgetLedger(spec.budget_seconds, state["budget"]["consumed"])
    history = [EpochEvent.model_validate(e) for e in state["history"]]
    checkpoint = CheckpointRef.model_validate(state["checkpoint"]) if state["checkpoint"] else None
    estimate = spec.epoch_seconds_estimate

    def save(phase="idle", **updates):
        state.update(updates)
        state["phase"] = phase
        state["budget"] = ledger.snapshot()
        state["gpu_hours"] = ledger.consumed/3600 if resource == "cuda" else None
        state["history"] = [e.model_dump() for e in history]
        state["checkpoint"] = checkpoint.model_dump() if checkpoint else None
        write_json(state_path, state)

    def commit_segment(segment):
        nonlocal checkpoint, estimate
        if segment.checkpoint and segment.events:
            checkpoint = segment.checkpoint
            valid_events = [e for e in segment.events if e.val_accuracy is not None]
            history.extend(e.model_copy(update={"branch_id": "main"}) for e in valid_events)
            state["epoch"] = checkpoint.epoch
            if segment.best_checkpoint and valid_events:
                quality = max(e.val_accuracy for e in valid_events)
                if quality > state["best_accuracy"]:
                    state["best_accuracy"] = quality
                    state["best_checkpoint"] = segment.best_checkpoint.model_dump()
            if valid_events:
                # Include child startup and teardown, not just hot epoch timing.
                estimate = max(estimate, segment.seconds / len(valid_events))

    save(status="RUNNING")
    try:
        while state["epoch"] < spec.max_epochs:
            if (directory/"stop.flag").exists():
                save(status="PAUSED", reason="Stop requested at safe segment boundary")
                return state
            epochs = min(spec.observation_interval, spec.max_epochs - state["epoch"])
            requested = (estimate * epochs + spec.worker_overhead_seconds) * (1 + spec.safety_margin)
            if requested > ledger.available:
                epochs = 1
                requested = (estimate + spec.worker_overhead_seconds) * (1 + spec.safety_margin)
            if not ledger.reserve("main", requested):
                save(status="BUDGET_EXHAUSTED", reason="Insufficient budget for another safe training segment")
                return state
            save(phase="training")
            segment_dir = directory/"segments"/f"main-after-{state['epoch']:04}"
            segment = backend.run(spec, checkpoint, epochs, "main", segment_dir, requested)
            ledger.settle("main", segment.seconds)
            append_event(log, "segment", branch="main", result=segment, output=str(segment_dir))
            commit_segment(segment)
            save()
            if segment.status != "COMPLETE":
                save(status=segment.status, reason=segment.error or "Training segment did not finish")
                return state
            if state["epoch"] >= spec.max_epochs:
                break
            evidence = build_evidence(history, "main", spec.window)
            append_event(log, "evidence", evidence=evidence)
            if evidence.status != "READY":
                append_event(log, "decision", intent="NOOP", reason="Observation window is incomplete")
                continue
            state["round"] += 1
            number = state["round"]
            current_settings = {name: getattr(history[-1], name) for name in
                                ("learning_rate", "momentum", "weight_decay", "grad_clip_norm")}
            context = settings_context(spec.dataset, evidence.signals, current_settings)
            previous = memory.retrieve(context, number)
            diagnosis, metadata = diagnose(spec, evidence, previous, gateway=gateway)
            append_event(log, "diagnosis", diagnosis=diagnosis, metadata=metadata, round=number)
            if diagnosis.status == "ABSTAIN":
                append_event(log, "decision", intent="NOOP", reason=diagnosis.reason, round=number)
                memory.record(context, 0, "ABSTAIN", 0, 0, number)
                save()
                continue
            proposed = {a.key: a for h in diagnosis.hypotheses for a in h.interventions}
            remaining_epochs = spec.max_epochs - state["epoch"]
            admission = None
            trials = []
            max_horizon = min(spec.trial_horizon, remaining_epochs)
            if max_horizon >= spec.min_trial_horizon and spec.max_trials:
                planned, plan_metadata = plan_with_agent(spec, diagnosis, previous, number, max_horizon,
                                                        gateway=gateway, current_settings=current_settings)
                append_event(log, "planning", round=number, trials=planned, metadata=plan_metadata)
            else:
                planned = []
            for horizon in range(max_horizon, spec.min_trial_horizon - 1, -1):
                if not planned:
                    break
                trials = [t.model_copy(update={"horizon": horizon}) for t in planned]
                continuation = (estimate + spec.worker_overhead_seconds) * (1 + spec.safety_margin) if horizon < remaining_epochs else 0
                admission = admit_trials(trials, ledger, estimate, spec.worker_overhead_seconds,
                                          spec.safety_margin, continuation, number)
                if admission.approved:
                    break
            append_event(log, "trial_plan", round=number, trials=trials, admission=admission)
            if not admission or not admission.approved:
                for action in proposed.values():
                    memory.record(context, action, "NOT_EXECUTED", 0, 0, number)
                append_event(log, "decision", intent="NOOP", reason="No feasible trial within budget/cooldown", round=number)
                save()
                continue
            trial_map = {t.trial_id: t for t in trials}
            for skipped in admission.skipped:
                memory.record(context, trial_map[skipped].intervention, "NOT_EXECUTED", 0, 0, number)
            round_dir = directory/"rounds"/f"round-{number:03}"
            write_json(round_dir/"admission.json", admission)
            save(phase="verification")
            outcome = verify_round(spec, checkpoint, admission, ledger, backend, round_dir)
            valid_trials = {name: r.events for name, r in outcome.trials.items() if r.status == "COMPLETE"}
            verdict, evaluation_metadata = evaluate_with_agent(
                spec, outcome.control.events if outcome.control.status == "COMPLETE" else [],
                valid_trials, outcome.comparison_epoch, gateway=gateway)
            selected = outcome.trials[verdict.selected_branch] if verdict.decision == "PROMOTE" else outcome.control
            append_event(log, "evaluation", round=number, result=outcome, verdict=verdict, metadata=evaluation_metadata)
            if selected.checkpoint and selected.status == "COMPLETE":
                commit_segment(selected)
            for trial in admission.approved:
                result = outcome.trials[trial.trial_id]
                base = outcome.control.events[-1] if outcome.control.events else None
                end = result.events[-1] if result.events else None
                gain = end.val_accuracy - base.val_accuracy if end and base and end.val_accuracy is not None and base.val_accuracy is not None else 0
                status = "ACCEPT" if verdict.selected_branch == trial.trial_id and verdict.decision == "PROMOTE" else "REJECT"
                if result.status != "COMPLETE":
                    status = "NOT_EXECUTED" if not result.events and result.seconds == 0 else "FAILED"
                memory.record(context, trial.intervention, status, gain, result.seconds, number)
            save()
            if outcome.control.status != "COMPLETE":
                save(status="INCONCLUSIVE", reason="Control incomplete; checkpoint retained, no verified improvement")
                return state
        save(status="COMPLETED", reason="Configured epoch limit reached")
        return state
    except KeyboardInterrupt:
        save(phase=state["phase"], status="INTERRUPTED", reason="Interrupted; inspect uncommitted work before continuation")
        raise
    except Exception as error:
        save(phase=state["phase"], status="FAILED", reason=f"{type(error).__name__}: {error}")
        raise
