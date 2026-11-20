"""Distinct authored v3 scenarios; validate raw output and executable plan semantics."""

import json
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
import numpy as np
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.configuration import (
    training_capabilities,
    compile_plan,
    compile_proposal,
    config_hash,
    ProposalError,
)
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_plan
from mas_demo.contracts import (
    DiagnosisReport,
    EvidenceWindow,
    EpochEvent,
    Hypothesis,
    ConfigPlanResponse,
    EvaluationResponse,
)
from mas_demo.llm.gateway import ResponsePayload, response_schema


def load_cases(path=None):
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parent
        / "scenarios/config_agent_cases.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def metric(epoch=4, accuracy=0.5, loss=1.0, branch="main", **settings):
    return EpochEvent(
        epoch=epoch,
        branch_id=branch,
        train_loss=loss,
        val_loss=loss,
        val_accuracy=accuracy,
        learning_rate=settings.pop("learning_rate", 0.01),
        grad_norm=settings.pop("grad_norm", 1.0),
        momentum=settings.pop("momentum", 0.9),
        weight_decay=settings.pop("weight_decay", 0.0),
        **settings
    )


def scenario_inputs(scenario, spec):
    updates, losses, accuracies, gradients = {}, [2.32, 2.34], [0.1, 0.09], [1.0, 1.1]
    if scenario in ("healthy", "noop"):
        losses, accuracies = [1.0, 0.7], [0.5, 0.7]
    elif scenario == "plateau":
        losses, accuracies = [1.2, 1.199], [0.4, 0.4]
    elif scenario == "unstable":
        updates = {"learning_rate": 0.1}
        losses, accuracies, gradients = [1.0, 4.0], [0.5, 0.1], [2.0, 10.0]
    elif scenario == "rising_loss":
        updates = {"learning_rate": 0.03}
        losses, accuracies = [1.2, 2.5], [0.4, 0.2]
    elif scenario in ("high_momentum", "momentum"):
        updates = {"momentum": 0.99}
        losses, accuracies, gradients = [1.5, 2.0], [0.4, 0.35], [4.0, 7.0]
    elif scenario in ("high_decay", "decay"):
        updates = {"weight_decay": 0.5}
        losses, accuracies = [2.30, 2.31], [0.1, 0.1]
    elif scenario in ("over_clipping", "clipping"):
        updates = {"grad_clip_norm": 0.001}
        losses, accuracies, gradients = [1.8, 1.79], [0.2, 0.2], [3.0, 4.0]
    elif scenario == "mixed":
        updates = {"learning_rate": 0.1, "momentum": 0.99, "weight_decay": 0.5}
        losses, accuracies, gradients = [1.3, 3.0], [0.3, 0.1], [5.0, 9.0]
    elif scenario == "history":
        updates = {"learning_rate": 0.027}
        losses, accuracies = [1.4, 1.35], [0.4, 0.42]
    spec.training = type(spec.training).model_validate(
        {**spec.training.model_dump(), **updates}
    )
    events = [
        metric(
            epoch=3 + i,
            loss=losses[i],
            accuracy=accuracies[i],
            grad_norm=gradients[i],
            learning_rate=spec.training.learning_rate,
            momentum=spec.training.momentum,
            weight_decay=spec.training.weight_decay,
            grad_clip_norm=spec.training.grad_clip_norm,
        )
        for i in range(2)
    ]
    if scenario == "insufficient":
        events = []
    elif scenario == "single_epoch":
        events = events[:1]
    return EvidenceWindow(
        epochs=[e.epoch for e in events],
        branch_id="main",
        status="READY" if len(events) >= 2 else "INSUFFICIENT",
        recent_events=events,
    )


def analysis_stage_scores(metadata):
    def valid(call):
        if call.get("finish_reason") not in (None, "stop") or not call.get("raw_output") or not call.get("response_schema"):
            return False
        try:
            Draft202012Validator(call["response_schema"]).validate(json.loads(call["raw_output"]))
            return True
        except (ValueError, ValidationError):
            return False
    scores = {}
    required = ["detection"]
    if (metadata.get("detection") or {}).get("status") == "DIAGNOSED":
        required.append("hypotheses")
    for stage in required:
        calls = [c for c in metadata.get("calls", []) if c.get("stage") == stage]
        first, last = (calls[0], calls[-1]) if calls else ({}, {})
        scores[stage] = dict(schema_valid=valid(last),
            validator_valid=valid(last) and not last.get("error") and not last.get("validation_error"),
            initial_schema_valid=valid(first),
            initial_validator_valid=valid(first) and not first.get("error") and not first.get("validation_error"))
    return scores


def run_case(case, spec, gateway):
    role, scenario = case["role"], case["scenario"]
    spec = spec.model_copy(deep=True)
    evidence = scenario_inputs(scenario, spec)
    compiled = None
    if role == "analysis":
        result, meta = diagnose(spec, evidence, [], gateway=gateway)
        actual = (meta.get("detection") or {}).get("status") or (result.status if result else "ERROR")
        schema = ResponsePayload
    elif role == "planner":
        remaining, budget = 8, 800
        history = []
        if scenario in ("max_one", "one"):
            spec.policy.max_trials = 1
        if scenario == "max_one":
            remaining, budget = 4, 200
        if scenario == "empty":
            spec.policy.max_trials = 0
        if scenario in ("cooldown", "history"):
            base = spec.training
            previous_lr = 0.027 if scenario == "cooldown" else 0.043
            candidate = base.model_copy(update={"learning_rate": previous_lr})
            history = [
                dict(
                    round_id=1,
                    base_config_hash=config_hash(base),
                    config_hash=config_hash(candidate),
                    status="REJECT" if scenario == "cooldown" else "ACCEPT",
                    gain=-0.01 if scenario == "cooldown" else 0.1,
                    diff={
                        "learning_rate": {
                            "before": base.learning_rate,
                            "after": previous_lr,
                        }
                    },
                    verification_epochs=2,
                    seconds=50,
                )
            ]
        healthy = scenario == "noop"
        report = DiagnosisReport(
            status="ABSTAIN" if healthy else "DIAGNOSED",
            mode=spec.analysis_mode,
            hypotheses=(
                []
                if healthy
                else [
                    Hypothesis(
                        hypothesis_id="observed" if rank == 1 else f"observed-{rank}",
                        cause=cause,
                        evidence_epochs=evidence.epochs,
                        rank=rank, confidence=0.6,
                    )
                    for rank, cause in enumerate([
                        "Learning rate may hinder progress", "Optimizer dynamics may be unstable",
                        "Gradient magnitudes may hinder updates"], 1)
                ]
            ),
            reason="Observed progress" if healthy else "Test alternatives",
        )
        diagnosis_source = "authored_default"
        if "diagnosis" in case:
            report = DiagnosisReport.model_validate(case["diagnosis"])
            diagnosis_source = "fixture"
        remaining = case.get("remaining_epochs", remaining)
        budget = case.get("remaining_budget_seconds", budget)
        result, meta = plan_with_agent(
            spec,
            report,
            evidence,
            spec.training,
            training_capabilities(),
            history,
            2,
            remaining,
            gateway=gateway,
        )
        meta["diagnosis_source"] = diagnosis_source
        meta["controller_inputs"] = {"max_trials": spec.max_trials, "available_seconds": budget}
        actual, schema = "INVALID", ConfigPlanResponse
        if result is not None:
            try:
                compiled = compile_plan(
                    spec.training,
                    result,
                    spec.policy,
                    evidence.epochs,
                    remaining,
                    {h.hypothesis_id: h.rank for h in report.hypotheses},
                )
                actual = "NONEMPTY" if compiled.candidates else "EMPTY"
                meta.update(proposed_count=len(compiled.candidates), approved_count=0, skipped_count=0,
                            admission_outcome="NONE")
                if compiled.candidates:
                    cooldown = {}
                    for candidate in compiled.candidates:
                        if any(
                            row["status"] in ("REJECT", "FAILED")
                            and row["base_config_hash"] == candidate.base_config_hash
                            and row["config_hash"] == candidate.config_hash
                            and 2 - row["round_id"] <= spec.policy.cooldown_rounds
                            for row in history
                        ):
                            cooldown[candidate.trial_id] = "Candidate repeats a recently rejected config"
                    admission = admit_plan(
                        compiled,
                        BudgetLedger(max(1.0, budget), consumed=max(1.0, budget) - budget),
                        20,
                        10,
                        0.3,
                        remaining,
                        2,
                        current_config=spec.training,
                        train_samples=spec.train_samples,
                        max_trials=spec.max_trials,
                        cooldown_reasons=cooldown,
                    )
                    meta.update(admission=admission.model_dump(), approved_count=len(admission.approved),
                                skipped_count=len(admission.skipped),
                                admission_outcome=("FULL" if len(admission.approved) == len(compiled.candidates)
                                                   else "PARTIAL" if admission.approved else "NONE"))
                meta["compiled"] = compiled.model_dump()
                meta["budget_feasible"] = bool(meta["approved_count"])
            except ProposalError as error:
                meta.update(error=str(error), validation_error=error.as_dict())
    else:
        control = [metric()]
        trial = metric(accuracy=0.7, loss=0.8, branch="candidate-A")
        if scenario == "no_gain":
            trial.val_accuracy, trial.val_loss = 0.49, 1.0
        elif scenario == "equal":
            trial.val_accuracy, trial.val_loss = 0.5, 1.0
        elif scenario == "worse_loss":
            trial.val_loss = 1.1
        elif scenario == "below_gain":
            trial.val_accuracy = 0.501
        elif scenario == "wrong_epoch":
            trial.epoch = 3
        if scenario == "missing_control":
            control = []
        if scenario == "later_control":
            control.append(metric(5, accuracy=0.99, loss=0.1))
        trials = {"candidate-A": [trial]} if scenario != "missing_trial" else {}
        if scenario == "two":
            trials["candidate-B"] = [
                metric(accuracy=0.8, loss=0.7, branch="candidate-B")
            ]
        result, meta = evaluate_with_agent(spec, control, trials, 4, gateway=gateway)
        actual, schema = result.decision, EvaluationResponse
    raw, schema_valid = meta.get("raw_output"), False
    if raw is not None and meta.get("finish_reason") in (None, "stop"):
        try:
            Draft202012Validator(
                meta.get("response_schema") or response_schema(
                    role, schema, meta.get("mode", spec.analysis_mode), spec.max_trials
                )
            ).validate(json.loads(raw))
            schema_valid = True
        except (ValueError, ValidationError):
            pass
    semantic_valid = schema_valid and "error" not in meta
    initial = (meta.get("calls") or [meta])[0]
    initial_schema_valid = False
    if isinstance(initial.get("raw_output"), str) and initial.get("finish_reason") in (None, "stop"):
        try:
            Draft202012Validator(initial.get("response_schema") or response_schema(role, schema, meta.get("mode", spec.analysis_mode), spec.max_trials)).validate(json.loads(initial["raw_output"]))
            initial_schema_valid = True
        except (ValueError, ValidationError):
            pass
    initial_semantic_valid = initial_schema_valid and not initial.get("error") and not initial.get("validation_error")
    stage_scores = None
    if role == "analysis" and meta.get("pipeline_version") == 2:
        stage_scores = analysis_stage_scores(meta)
        schema_valid = all(s["schema_valid"] for s in stage_scores.values())
        semantic_valid = all(s["validator_valid"] for s in stage_scores.values()) and not meta.get("error")
        initial_schema_valid = all(s["initial_schema_valid"] for s in stage_scores.values())
        initial_semantic_valid = all(s["initial_validator_valid"] for s in stage_scores.values())
    correct = semantic_valid and (
        case["expected"] == "VALID" or actual == case["expected"]
    )
    initial_numbers = {
        value
        for value in spec.training.model_dump().values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }
    novel = (
        sum(
            any(
                isinstance(change["after"], (float, int))
                and not isinstance(change["after"], bool)
                and change["after"] not in initial_numbers
                for change in candidate.diff.values()
            )
            for candidate in compiled.candidates
        )
        if compiled
        else 0
    )
    return dict(
        case_id=case["id"],
        role=role,
        scenario=scenario,
        expected=case["expected"],
        actual=actual,
        schema_valid=schema_valid,
        initial_schema_valid=initial_schema_valid,
        initial_semantic_valid=initial_semantic_valid,
        semantic_valid=semantic_valid,
        transport_ok=bool(meta.get("request_id") or meta.get("upstream_errors")),
        api_success=bool(meta.get("request_id")),
        model_abstained=(actual in ("ABSTAIN", "EMPTY") and semantic_valid),
        error_fallback=("error" in meta),
        selected_horizon=compiled.verification_epochs if compiled else None,
        multi_field_proposals=(
            sum(len(c.diff) > 1 for c in compiled.candidates) if compiled else 0
        ),
        novel_numeric_proposals=novel,
        correct=correct,
        **(dict(stage_scores=stage_scores,
                detection_correct=actual == case["expected"],
                diagnosis_ready=result is not None and result.status == "DIAGNOSED") if role == "analysis" else {}),
        **meta
    )


def summarize(records):
    def rates(rows):
        return {
            "count": len(rows),
            **{
                key
                + "_rate": (
                    sum(bool(r.get(key)) for r in rows) / len(rows) if rows else None
                )
                for key in ("schema_valid", "semantic_valid", "initial_schema_valid", "initial_semantic_valid", "correct", "transport_ok", "error_fallback", "repair_attempted")
            },
        }

    result = rates(records)
    result["roles"] = {
        role: rates([r for r in records if r["role"] == role])
        for role in ("analysis", "planner", "evaluation")
    }
    planners = [row for row in records if row["role"] == "planner"]
    result["planner_outcomes"] = {
        "count": len(planners),
        "valid_nonempty_count": sum(bool(row.get("semantic_valid")) and row.get("actual") == "NONEMPTY" for row in planners),
        "valid_empty_count": sum(bool(row.get("semantic_valid")) and row.get("actual") == "EMPTY" for row in planners),
        "noop_count": sum((row.get("validation_error") or {}).get("code") == "NO_OP" for row in planners),
        "budget_rejected_count": sum((row.get("validation_error") or {}).get("code") == "INSUFFICIENT_BUDGET" for row in planners),
        "truncated_count": sum(row.get("finish_reason") == "length" for row in planners),
    }
    result["admission_outcomes"] = {
        outcome.lower(): sum(row.get("admission_outcome") == outcome for row in planners)
        for outcome in ("FULL", "PARTIAL", "NONE")
    }
    values = [r["latency_seconds"] for r in records]
    result["latency_seconds"] = {
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "total": sum(values),
    }
    for role, row in result["roles"].items():
        timings = [r["latency_seconds"] for r in records if r["role"] == role]
        row["latency_seconds"] = {"p50": float(np.percentile(timings, 50)) if timings else None,
                                  "p95": float(np.percentile(timings, 95)) if timings else None}
    result["token_usage"] = {}
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, dict):
            for name, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result["token_usage"][name] = result["token_usage"].get(name, 0) + value
    result["repair_count"] = sum(bool(r.get("repair_attempted")) for r in records)
    result["error_count"] = sum(bool(r.get("error_fallback")) for r in records)
    measurements = [r.get("server", {}) for r in records]
    result["resources"] = {
        "peak_vram_mb": max(
            (
                m["peak_vram_mb"]
                for m in measurements
                if m.get("peak_vram_mb") is not None
            ),
            default=None,
        ),
        "peak_rss_mb": max(
            (m["rss_mb"] for m in measurements if m.get("rss_mb") is not None),
            default=None,
        ),
    }
    result["server_revisions"] = sorted(
        {m["revision"] for m in measurements if m.get("revision")}
    )
    result["server_dtypes"] = sorted(
        {m["dtype"] for m in measurements if m.get("dtype")}
    )
    result["demo_thresholds_met"] = (
        bool(records)
        and result["schema_valid_rate"] >= 0.95
        and all(
            row["count"] >= 30
            and row["correct_rate"] >= 0.8
            and len({r.get("case_id") for r in records if r["role"] == role}) >= 10
            for role, row in result["roles"].items()
        )
    )
    valid_horizons = {
        r.get("selected_horizon")
        for r in records
        if r.get("role") == "planner"
        and r.get("semantic_valid")
        and r.get("actual") == "NONEMPTY"
    }
    valid_horizons.discard(None)
    result["autonomy"] = {
        "distinct_valid_horizons": sorted(valid_horizons),
        "novel_numeric_proposals": sum(
            r.get("novel_numeric_proposals", 0)
            for r in records
            if r.get("semantic_valid")
        ),
        "multi_field_proposals": sum(
            r.get("multi_field_proposals", 0)
            for r in records
            if r.get("semantic_valid")
        ),
    }
    result["demo_thresholds_met"] = (
        result["demo_thresholds_met"]
        and len(valid_horizons) >= 2
        and result["autonomy"]["novel_numeric_proposals"] > 0
    )
    result["limitations"] = (
        "Authored demo expectations; Planner VALID accepts valid abstention, while NONEMPTY requires compiler-valid proposals. Controller FULL/PARTIAL/NONE is scored separately. Fixture/authored diagnoses test Planner independently, not Analysis quality or causal truth. Does not measure CIFAR generalization. Guard rejection tests are separate from model quality."
    )
    result["quality_thresholds_met"] = (result["demo_thresholds_met"]
        and all(row["semantic_valid_rate"] is not None and row["semantic_valid_rate"] >= 0.95 for row in result["roles"].values())
        and result["latency_seconds"]["p95"] is not None and result["latency_seconds"]["p95"] <= 60)
    return result
