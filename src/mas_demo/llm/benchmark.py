"""Authored agent scenarios; scores the model output before fallback decisions.

Analysis labels express a demo preference, not ground-truth causal diagnoses.
Planner nonempty labels express a preference; the actual agent may abstain.
"""
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.contracts import DiagnosisReport, EvidenceWindow, EpochEvent, Hypothesis, Intervention, PlanResponse, EvaluationResponse
from mas_demo.llm.gateway import ResponsePayload, response_schema


def load_cases(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[3] / "tests/fixtures/agent_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metric(epoch=4, accuracy=0.5, loss=1.0, branch="main", **settings):
    return EpochEvent(epoch=epoch, branch_id=branch, train_loss=loss, val_loss=loss,
        val_accuracy=accuracy, learning_rate=settings.pop("learning_rate", 1.0),
        grad_norm=settings.pop("grad_norm", 10), momentum=settings.pop("momentum", 0.9),
        weight_decay=settings.pop("weight_decay", 0), **settings)


def run_case(case, spec, gateway):
    role, scenario = case["role"], case["scenario"]
    spec = spec.model_copy(deep=True)
    actions = spec.intervention_allowlist
    if role == "analysis":
        changes = {"high_momentum": {"momentum": 0.99}, "high_decay": {"weight_decay": 0.5},
                   "over_clipping": {"grad_clip_norm": 0.001}, "mixed": {"momentum": 0.99, "weight_decay": 0.5}}.get(scenario, {})
        if scenario in ("high_momentum", "high_decay", "over_clipping", "mixed"):
            extra = {"high_momentum": ("momentum", 0.5), "high_decay": ("weight_decay", 0.001),
                     "over_clipping": ("grad_clip_norm", 1.0), "mixed": ("weight_decay", 0.001)}[scenario]
            spec.allowed_interventions = [Intervention(parameter=extra[0], operation="set", value=extra[1])]
        events = [metric(3, loss=1, **changes), metric(4, loss=4 if scenario in ("unstable", "rising_loss", "mixed") else 1, **changes)]
        signals = [scenario]
        if scenario == "healthy":
            events = [metric(3, accuracy=0.5, loss=1, learning_rate=0.01), metric(4, accuracy=0.7, loss=0.7, learning_rate=0.01)]
            signals = []
        if scenario == "insufficient":
            events, signals = [], []
        elif scenario == "single_epoch":
            events, signals = events[:1], []
        evidence = EvidenceWindow(epochs=[e.epoch for e in events], branch_id="main",
            status="READY" if len(events) >= 2 else "INSUFFICIENT", signals=signals, recent_events=events)
        result, meta = diagnose(spec, evidence, [], gateway=gateway)
        actual = result.status
        schema = ResponsePayload
    elif role == "planner":
        history, current = [], None
        if scenario in ("momentum", "decay", "clipping", "noop"):
            parameter, value = {"momentum": ("momentum", 0.5), "decay": ("weight_decay", 0.001),
                                "clipping": ("grad_clip_norm", 1.0), "noop": ("momentum", 0.5)}[scenario]
            actions = [Intervention(parameter=parameter, operation="set", value=value)]
            spec.allowed_interventions = actions
            if scenario == "noop":
                current = {parameter: value}
        elif scenario == "one":
            actions = actions[:1]
        elif scenario == "empty":
            actions = []
        elif scenario == "cooldown":
            history = [{"intervention": a.model_dump(), "status": "REJECT", "round_id": 1, "gain": 0, "seconds": 1} for a in actions]
        elif scenario == "history":
            history = [{"intervention": actions[0].model_dump(), "status": "ACCEPT", "round_id": 1, "gain": 0.1, "seconds": 1}]
        if scenario == "max_one":
            spec.max_trials = 1
        report = DiagnosisReport(status="DIAGNOSED" if actions else "ABSTAIN", mode=spec.analysis_mode,
            hypotheses=[Hypothesis(hypothesis_id="observed", cause="Loss plateau", evidence_epochs=[3, 4], interventions=actions, confidence=0.6)] if actions else [], reason="Observed loss plateau")
        result, meta = plan_with_agent(spec, report, history, 2, 1, gateway=gateway, current_settings=current)
        actual = "NONEMPTY" if result else "EMPTY"
        schema = PlanResponse
    else:
        control = [metric()]
        trial = metric(accuracy=0.7, loss=0.8, branch="candidate-A")
        if scenario in ("no_gain", "equal"):
            trial = metric(accuracy=0.5, loss=1, branch="candidate-A")
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
            trials["candidate-B"] = [metric(accuracy=0.8, loss=0.7, branch="candidate-B")]
        result, meta = evaluate_with_agent(spec, control, trials, 4, gateway=gateway)
        actual = result.decision
        schema = EvaluationResponse
    raw = meta.get("raw_output")
    schema_valid = False
    if raw is not None and meta.get("finish_reason") in (None, "stop"):
        try:
            Draft202012Validator(response_schema(role, schema, meta.get("mode", spec.analysis_mode))).validate(json.loads(raw))
            schema_valid = True
        except (ValueError, ValidationError):
            pass
    semantic_valid = schema_valid and "error" not in meta
    return {"case_id": case["id"], "role": role, "scenario": scenario, "expected": case["expected"],
            "actual": actual, "schema_valid": schema_valid, "semantic_valid": semantic_valid,
            "transport_ok": bool(meta.get("request_id") or meta.get("upstream_errors")),
            "api_success": bool(meta.get("request_id")),
            "correct": semantic_valid and actual == case["expected"], **meta}


def summarize(records):
    def rates(rows):
        return {"count": len(rows), **{key + "_rate": sum(bool(r.get(key)) for r in rows) / len(rows) if rows else None
            for key in ("schema_valid", "semantic_valid", "correct", "transport_ok")}}
    result = rates(records)
    result["roles"] = {role: rates([r for r in records if r["role"] == role]) for role in ("analysis", "planner", "evaluation")}
    values = [r["latency_seconds"] for r in records]
    result["latency_seconds"] = {"p50": float(np.percentile(values, 50)) if values else None,
                                 "p95": float(np.percentile(values, 95)) if values else None,
                                 "total": sum(values)}
    measurements = [r.get("server", {}) for r in records]
    result["resources"] = {"peak_vram_mb": max((m["peak_vram_mb"] for m in measurements if m.get("peak_vram_mb") is not None), default=None),
                           "peak_rss_mb": max((m["rss_mb"] for m in measurements if m.get("rss_mb") is not None), default=None)}
    result["server_revisions"] = sorted({m["revision"] for m in measurements if m.get("revision")})
    result["server_dtypes"] = sorted({m["dtype"] for m in measurements if m.get("dtype")})
    result["demo_thresholds_met"] = bool(records) and result["schema_valid_rate"] >= 0.95 and all(
        row["count"] >= 30 and row["correct_rate"] >= 0.8 and
        len({r.get("case_id") for r in records if r["role"] == role}) >= 10
        for role, row in result["roles"].items())
    result["limitations"] = "Authored demo expectations; no calibrated causal ground truth. Does not measure CIFAR generalization. Guard rejection tests are separate from model quality."
    return result
