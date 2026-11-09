"""Durable protocol-v3 loop: model-generated config trials and shared horizon."""

from pathlib import Path
import time
from filelock import FileLock
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.configuration import (
    config_hash,
    compile_plan,
    training_capabilities,
    ProposalError,
)
from mas_demo.contracts import TaskSpec, EpochEvent, TrainingConfig, CheckpointRef
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_plan
from mas_demo.control.verification import verify_round
from mas_demo.data import prepare_data
from mas_demo.execution.local import LocalBackend
from mas_demo.io import append_event, read_json, write_json
from mas_demo.memory.store import MemoryStore
from mas_demo.observation.collector import build_evidence
from mas_demo.training.checkpoint import validate_checkpoint
from mas_demo.training.worker import resolve_device
from mas_demo.training.model import model_signature, task_description
from mas_demo.llm.settings import create_gateway


def run(spec, directory, resume=False, backend=None, gateway=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "run.lock"), timeout=0):
        return _run(
            spec,
            directory,
            resume,
            backend or LocalBackend(),
            gateway or create_gateway(spec),
        )


def _run(spec, directory, resume, backend, gateway):
    state_path, log = directory / "state.json", directory / "events.jsonl"
    memory = MemoryStore(directory / "memory.sqlite")
    resource = resolve_device(spec.device)
    context = f"{spec.dataset}:{model_signature(spec)}:{spec.seed}:{spec.train_samples}:{spec.val_samples}"
    policy = spec.policy
    if state_path.exists():
        if not resume:
            raise ValueError("Run exists; choose a new output directory or resume")
        state = read_json(state_path)
        if state.get("schema_version") != 3:
            raise ValueError(
                "Legacy run is view-only. Migrate its config and start a new v3 run."
            )
        if TaskSpec.model_validate(read_json(directory / "config.json")) != spec:
            raise ValueError("Resume config differs from saved configuration")
        if state["status"] in ("COMPLETED", "EARLY_STOPPED"):
            return state
        if state["phase"] != "idle":
            state.update(
                status="INTERRUPTED",
                reason="Uncommitted work; reconcile costs before continuation",
            )
            write_json(state_path, state)
            return state
        active = TrainingConfig.model_validate(state["active_config"])
        checkpoint = (
            CheckpointRef.model_validate(state["checkpoint"])
            if state["checkpoint"]
            else None
        )
        if state["active_config_hash"] != config_hash(active) or (
            checkpoint and checkpoint.config_hash != config_hash(active)
        ):
            raise ValueError("Active checkpoint/config hash mismatch")
        if checkpoint:
            validate_checkpoint(checkpoint)
    else:
        manifest = prepare_data(spec)
        write_json(directory / "config.json", spec)
        write_json(directory / "dataset_manifest.json", manifest)
        active, checkpoint = spec.training.model_copy(deep=True), None
        state = dict(
            schema_version=3,
            task=task_description(spec),
            status="RUNNING",
            phase="idle",
            epoch=0,
            round=0,
            checkpoint=None,
            best_checkpoint=None,
            best_accuracy=-1,
            history=[],
            budget=BudgetLedger(spec.budget_seconds).snapshot(),
            resource_type=resource,
            gpu_hours=None,
            origin=manifest.origin,
            analysis_mode=spec.analysis_mode,
            reason="",
            active_config=active.model_dump(),
            active_config_hash=config_hash(active),
            config_revision=0,
            llm_seconds=0.0,
            wall_seconds=0.0,
            cost_measure="Training worker wall time includes startup and teardown; LLM reported separately.",
        )
    history = [EpochEvent.model_validate(value) for value in state["history"]]
    ledger = BudgetLedger(spec.budget_seconds, state["budget"]["consumed"])
    estimate = max(policy.epoch_seconds_estimate, state.get("epoch_estimate", 0))
    started, previous_wall = time.monotonic(), state.get("wall_seconds", 0)

    def save(phase="idle", **updates):
        state.update(updates)
        state.update(
            phase=phase,
            checkpoint=checkpoint.model_dump() if checkpoint else None,
            history=[event.model_dump() for event in history],
            budget=ledger.snapshot(),
            active_config=active.model_dump(),
            active_config_hash=config_hash(active),
            epoch_estimate=estimate,
            gpu_hours=ledger.consumed / 3600 if resource == "cuda" else None,
            wall_seconds=previous_wall + time.monotonic() - started,
        )
        # Config file is immutable by hash, written before the atomic active manifest.
        write_json(
            directory / "config-revisions" / f"{config_hash(active)}.json", active
        )
        write_json(state_path, state)

    def commit_segment(segment):
        nonlocal checkpoint, active, estimate
        if not segment.checkpoint or not segment.training_config or not segment.events:
            return
        validate_checkpoint(segment.checkpoint)
        if segment.checkpoint.config_hash != config_hash(segment.training_config):
            raise ValueError("Worker checkpoint does not match effective configuration")
        if segment.checkpoint.epoch != segment.events[-1].epoch:
            raise ValueError("Worker checkpoint epoch mismatch")
        if config_hash(active) != config_hash(segment.training_config):
            state["config_revision"] += 1
        checkpoint, active = segment.checkpoint, segment.training_config
        history.extend(
            event.model_copy(update={"branch_id": "main"}) for event in segment.events
        )
        state["epoch"] = checkpoint.epoch
        best = max(
            event.val_accuracy
            for event in segment.events
            if event.val_accuracy is not None
        )
        if best > state["best_accuracy"]:
            state.update(
                best_accuracy=best,
                best_checkpoint=(segment.best_checkpoint or checkpoint).model_dump(),
            )
        estimate = max(estimate, segment.seconds / len(segment.events))

    def agent_cost(metadata):
        state["llm_seconds"] += metadata.get("latency_seconds", 0)

    save(status="RUNNING")
    try:
        while state["epoch"] < spec.max_epochs:
            if (directory / "stop.flag").exists():
                save(status="PAUSED", reason="Stop requested at a safe boundary")
                return state
            requested = (estimate + policy.worker_overhead_seconds) * (
                1 + policy.safety_margin
            )
            if not ledger.reserve("main", requested):
                save(
                    status="BUDGET_EXHAUSTED",
                    reason="Insufficient budget for another complete epoch",
                )
                return state
            save(phase="training")
            output = directory / "segments" / f"main-after-{state['epoch']:04}"
            segment = backend.run(
                spec, checkpoint, 1, "main", output, requested, training_config=active
            )
            ledger.settle("main", segment.seconds)
            append_event(log, "segment", result=segment, output=str(output))
            commit_segment(segment)
            save()
            if segment.status not in ("COMPLETE", "EARLY_STOPPED"):
                save(
                    status=segment.status,
                    reason=segment.error or "Main segment incomplete or early-stopped",
                )
                return state
            evidence = build_evidence(history, "main", spec.window)
            state["round"] += 1
            number = state["round"]
            append_event(log, "evidence", evidence=evidence)
            # Compact outcomes contain provenance IDs; full records remain in SQLite.
            previous = memory.retrieve(context, number, limit=4)
            diagnosis, metadata = diagnose(
                spec, evidence, previous, gateway=gateway, current_config=active
            )
            agent_cost(metadata)
            append_event(
                log, "diagnosis", diagnosis=diagnosis, metadata=metadata, round=number
            )
            if segment.status == "EARLY_STOPPED":
                save(
                    status="EARLY_STOPPED",
                    reason="Training early-stopping criterion reached",
                )
                return state
            if state["epoch"] >= spec.max_epochs:
                break
            if diagnosis.status == "ABSTAIN" or not spec.max_trials:
                append_event(
                    log,
                    "decision",
                    intent="NOOP",
                    reason=diagnosis.reason,
                    round=number,
                )
                save()
                continue
            remaining = spec.max_epochs - state["epoch"]
            admission = compiled = None
            previous_response, errors = None, None
            for attempt in range(2):
                response, metadata = plan_with_agent(
                    spec,
                    diagnosis,
                    evidence,
                    active,
                    training_capabilities(),
                    previous,
                    number,
                    remaining,
                    ledger.available,
                    {
                        "epoch_seconds": estimate,
                        "worker_overhead_seconds": policy.worker_overhead_seconds,
                        "safety_margin": policy.safety_margin,
                        "control_required": True,
                        "continuation_required_if_not_final": True,
                    },
                    gateway=gateway,
                    previous_response=previous_response,
                    validation_errors=errors,
                )
                agent_cost(metadata)
                append_event(
                    log,
                    "planning",
                    response=(
                        response.model_dump(exclude_unset=True) if response else None
                    ),
                    metadata=metadata,
                    round=number,
                    repair_attempt=attempt,
                )
                previous_response = (
                    response.model_dump(exclude_unset=True)
                    if response
                    else metadata.get("raw_output")
                )
                try:
                    if response is None:
                        raise ProposalError(
                            "INVALID_MODEL_RESPONSE",
                            metadata.get("error", "Planner response invalid"),
                        )
                    compiled = compile_plan(
                        active,
                        response,
                        policy,
                        evidence.epochs,
                        remaining,
                        {h.hypothesis_id for h in diagnosis.hypotheses},
                    )
                    if not compiled.candidates:
                        break
                    memory.check_cooldown(
                        context, compiled, number, policy.cooldown_rounds
                    )
                    continuation = (
                        (estimate + policy.worker_overhead_seconds)
                        * (1 + policy.safety_margin)
                        if compiled.verification_epochs < remaining
                        else 0
                    )
                    admission = admit_plan(
                        compiled,
                        ledger,
                        estimate,
                        policy.worker_overhead_seconds,
                        policy.safety_margin,
                        continuation,
                        number,
                        current_config=active,
                        train_samples=spec.train_samples,
                    )
                    break
                except ProposalError as error:
                    errors = [error.as_dict()]
                    append_event(
                        log,
                        "plan_rejected",
                        round=number,
                        repair_attempt=attempt,
                        errors=errors,
                    )
            if admission is None:
                append_event(
                    log,
                    "decision",
                    intent="NOOP",
                    reason="Planner abstained or plan remained infeasible",
                    round=number,
                )
                save()
                continue
            round_dir = directory / "rounds" / f"round-{number:03}"
            write_json(round_dir / "admission.json", admission)
            append_event(
                log, "trial_plan", round=number, plan=compiled, admission=admission
            )
            save(phase="verification")
            outcome = verify_round(
                spec,
                checkpoint,
                admission,
                ledger,
                backend,
                round_dir,
                current_config=active,
            )
            valid_trials = {
                key: part.events
                for key, part in outcome.trials.items()
                if part.status == "COMPLETE"
            }
            verdict, metadata = evaluate_with_agent(
                spec,
                outcome.control.events if outcome.control.status == "COMPLETE" else [],
                valid_trials,
                outcome.comparison_epoch,
                gateway=gateway,
                candidates={
                    candidate.trial_id: candidate.model_dump()
                    for candidate in admission.approved
                },
                verification_epochs=admission.horizon,
            )
            agent_cost(metadata)
            append_event(
                log,
                "evaluation",
                round=number,
                result=outcome,
                verdict=verdict,
                metadata=metadata,
            )
            selected = (
                outcome.trials[verdict.selected_branch]
                if verdict.decision == "PROMOTE"
                else outcome.control
            )
            base_hash = config_hash(active)
            for candidate in admission.approved:
                part = outcome.trials[candidate.trial_id]
                status = (
                    "ACCEPT"
                    if verdict.decision == "PROMOTE"
                    and verdict.selected_branch == candidate.trial_id
                    else "REJECT"
                )
                if part.status != "COMPLETE":
                    status = "FAILED" if part.seconds else "NOT_EXECUTED"
                base_event = (
                    outcome.control.events[-1]
                    if outcome.control.status == "COMPLETE" and outcome.control.events
                    else None
                )
                trial_event = (
                    part.events[-1]
                    if part.status == "COMPLETE" and part.events
                    else None
                )
                gain = (
                    trial_event.val_accuracy - base_event.val_accuracy
                    if trial_event and base_event
                    else None
                )
                memory.record(
                    context,
                    base_hash,
                    candidate.config_hash,
                    number,
                    status=status,
                    diff=candidate.diff,
                    verification_epochs=admission.horizon,
                    comparison_epoch=outcome.comparison_epoch,
                    gain=gain,
                    seconds=part.seconds,
                    proposal_id=candidate.proposal_id,
                )
            if selected.status == "COMPLETE":
                commit_segment(selected)
            save()
            if outcome.control.status != "COMPLETE":
                save(
                    status="INCONCLUSIVE",
                    reason="Control did not reach shared comparison epoch",
                )
                return state
        save(status="COMPLETED", reason="Configured epoch limit reached")
        return state
    except KeyboardInterrupt:
        save(
            phase=state["phase"],
            status="INTERRUPTED",
            reason="Interrupted; reconcile uncommitted work",
        )
        raise
    except Exception as error:
        save(
            phase=state["phase"],
            status="FAILED",
            reason=f"{type(error).__name__}: {error}",
        )
        raise
