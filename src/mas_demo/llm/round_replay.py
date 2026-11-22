"""Replay one saved Analysis/Planner round without training or artifact writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from mas_demo.agents.analysis import diagnose
from mas_demo.agents.analysis_validation import validate_hypotheses, select_distinct_hypotheses
from mas_demo.agents.planner import plan_with_agent
from mas_demo.configuration import ProposalError, compile_plan, training_capabilities
from mas_demo.contracts import (
    ConfigPlanResponse, DiagnosisReport, EpochLogHypothesesResponse, EvidenceWindow, TaskSpec, TrainingConfig,
)
from mas_demo.observation.facts import build_config_facts


def load_round(directory, round_id):
    directory = Path(directory).resolve()
    data = (directory / "events.jsonl").read_bytes()
    rows = []
    for line in data.splitlines(keepends=True):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if line.endswith(b"\n"):
                raise
            # A currently running process may still be appending the last event.
    diagnosis = next((r for r in rows if r.get("round") == round_id and r.get("kind") == "diagnosis"), None)
    planning = next((r for r in rows if r.get("round") == round_id and r.get("kind") == "planning"), None)
    if diagnosis is None or planning is None:
        raise ValueError(f"Saved round {round_id} needs diagnosis and planning input records")
    payload = planning["metadata"]["input_payload"]
    calls = diagnosis.get("metadata", {}).get("calls", [])
    hypothesis_call = next((c for c in reversed(calls) if c.get("stage") == "hypotheses"), {})
    generated = hypothesis_call.get("output")
    selected = (payload.get("diagnosis") or diagnosis.get("diagnosis")
                or hypothesis_call.get("selected_output") or generated or {})
    return dict(
        source=dict(directory=str(directory), round=round_id,
                    events_sha256=hashlib.sha256(data).hexdigest()),
        spec=json.loads((directory / "config.json").read_text(encoding="utf-8")),
        evidence=payload["evidence"], current_config=payload["current_config"],
        history=payload.get("history", []), remaining_epochs=payload["remaining_epochs"] if "remaining_epochs" in payload else None,
        hypotheses_response=generated, plan_hypotheses=selected.get("hypotheses", []),
        plan_response=planning.get("response"),
    )


def run_round_replay(case, gateway, emit=None):
    case = deepcopy(case)
    emit = emit or (lambda event: None)
    spec = TaskSpec.model_validate(case["spec"])
    evidence = EvidenceWindow.model_validate(case["evidence"])
    current = TrainingConfig.model_validate(case["current_config"])
    history = case.get("history", [])
    remaining = case.get("remaining_epochs")
    if remaining is None:
        remaining = spec.max_epochs - max(evidence.epochs)
    original = dict(hypothesis_errors=[], plan_errors=[])
    try:
        old = EpochLogHypothesesResponse.model_validate(case.get("hypotheses_response"))
        original["hypothesis_errors"] = validate_hypotheses(old, evidence, [], build_config_facts(current))
    except ValidationError as error:
        original["hypothesis_errors"] = [dict(code="invalid_saved_hypotheses", reason=str(error))]
    try:
        old_plan = ConfigPlanResponse.model_validate(case.get("plan_response"))
        plan_hypotheses = case.get("plan_hypotheses")
        if plan_hypotheses is None:
            plan_hypotheses = (case.get("hypotheses_response") or {}).get("hypotheses", [])
        ranks = {h["hypothesis_id"]: h["rank"] for h in plan_hypotheses}
        compile_plan(current, old_plan, spec.policy, evidence.epochs, remaining, ranks)
    except (ProposalError, ValidationError) as error:
        original["plan_errors"] = [error.as_dict() if isinstance(error, ProposalError)
                                   else dict(code="invalid_saved_plan", reason=str(error))]

    planner_only = "provided_hypotheses" in case
    report = dict(source=case.get("source"), original=original, outcome="analysis_failed",
                  mode="planner_only" if planner_only else "pipeline",
                  grounding_preserved=False, diagnosis=None, planning=[], compiled_plan=None,
                  manual_review_pending=True)
    if planner_only:
        diagnosis = None
        metadata = dict(outcome="invalid_provided_hypotheses", calls=[])
        try:
            provided = EpochLogHypothesesResponse.model_validate(case["provided_hypotheses"])
            metadata["output"] = provided.model_dump()
            provided, metadata["selection"] = select_distinct_hypotheses(provided)
            errors = validate_hypotheses(provided, evidence, [], build_config_facts(current))
            metadata["validation_errors"] = errors
            if not errors:
                diagnosis = DiagnosisReport(status="DIAGNOSED", mode=spec.analysis_mode,
                    hypotheses=provided.hypotheses, reason=provided.reason)
                metadata.update(outcome="provided_hypotheses", generated_hypotheses=provided.model_dump()["hypotheses"])
        except ValidationError as error:
            metadata["error"] = str(error)
    else:
        diagnosis, metadata = diagnose(spec, evidence, history, gateway=gateway, current_config=current)
    report["diagnosis"] = dict(output=diagnosis.model_dump() if diagnosis else None, metadata=metadata)
    emit(dict(kind="diagnosis", **report["diagnosis"]))
    if diagnosis is None:
        return report
    if diagnosis.status == "ABSTAIN":
        report["outcome"] = "analysis_abstained"
        return report
    report["grounding_preserved"] = [h.model_dump() for h in diagnosis.hypotheses] == metadata.get("generated_hypotheses")
    previous, errors = None, None
    report["outcome"] = "planning_failed"
    for attempt in range(2):
        response, meta = plan_with_agent(spec, diagnosis, evidence, current, training_capabilities(),
            history, (case.get("source") or {}).get("round", 4), remaining, gateway=gateway,
            previous_response=previous, validation_errors=errors)
        output = response.model_dump(exclude_unset=True) if response else None
        previous = output if response else meta.get("raw_output")
        compiled = None
        try:
            if response is None:
                raise ProposalError("INVALID_MODEL_RESPONSE", meta.get("error", "Planner response invalid"))
            compiled = compile_plan(current, response, spec.policy, evidence.epochs, remaining,
                                    {h.hypothesis_id: h.rank for h in diagnosis.hypotheses})
            errors = []
        except ProposalError as error:
            errors = [error.as_dict()]
        record = dict(attempt=attempt, response=output, metadata=meta, validation_errors=errors)
        report["planning"].append(record)
        emit(dict(kind="planning", **record))
        if compiled is not None:
            report["compiled_plan"] = compiled.model_dump()
            report["outcome"] = "ready" if len(compiled.candidates) == len(diagnosis.hypotheses) and report["grounding_preserved"] else "invalid_plan"
            break
    return report
