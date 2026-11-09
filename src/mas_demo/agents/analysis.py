"""Diagnose observed training, independently of candidate generation."""

from mas_demo.contracts import DiagnosisReport
from mas_demo.configuration import config_hash
from mas_demo.llm.gateway import GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway
from mas_demo.training.model import task_description


def _validation_error(report, evidence):
    if report.status == "DIAGNOSED" and not report.hypotheses:
        return {
            "code": "missing_hypothesis",
            "reason": "Diagnosis requires an evidence-backed hypothesis",
        }
    if report.status == "ABSTAIN" and report.hypotheses:
        return {
            "code": "contradictory_status",
            "reason": "ABSTAIN requires an empty hypotheses list",
        }
    ids = [h.hypothesis_id for h in report.hypotheses]
    if len(ids) != len(set(ids)):
        return {
            "code": "duplicate_hypothesis",
            "reason": "Hypothesis IDs must be unique",
        }
    for hypothesis in report.hypotheses:
        if not hypothesis.evidence_epochs or not set(hypothesis.evidence_epochs) <= set(
            evidence.epochs
        ):
            return {
                "code": "unavailable_epochs",
                "reason": "Hypothesis cites unavailable evidence epochs",
            }
    return None


def _audit(calls, outcome):
    # Top-level costs cover both calls; each raw response/error remains available.
    metadata = dict(calls[-1])
    usage = {}
    for call in calls:
        call_usage = call.get("usage")
        if not isinstance(call_usage, dict):
            continue
        for name, value in call_usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[name] = usage.get(name, 0) + value
    metadata.update(
        calls=calls,
        outcome=outcome,
        repair_attempted=len(calls) == 2,
        repair_succeeded=outcome == "repaired",
        attempts=sum(call.get("attempts", 0) for call in calls),
        latency_seconds=sum(call.get("latency_seconds", 0) for call in calls),
        usage=usage,
    )
    return metadata


def diagnose(spec, evidence, history, gateway=None, current_config=None):
    gateway = gateway or create_gateway(spec)
    current_config = current_config or spec.training
    payload = {
        "schema_version": 3,
        "evidence": evidence.model_dump(exclude={"signals"}),
        "history": sorted(history, key=lambda row: row.get("round_id", 0))[-4:],
        "current_config": current_config.model_dump(),
        "config_hash": config_hash(current_config),
        "task": task_description(spec),
    }
    calls = []
    for attempt in range(2):
        metadata = {}
        try:
            report, metadata = gateway.complete(payload)
            metadata = dict(metadata)
            metadata["output"] = report.model_dump(exclude={"mode"})
            error = _validation_error(report, evidence)
            if error:
                metadata["validation_error"] = error
                raise GatewayError(error["reason"])
        except GatewayError as error:
            metadata = failure_metadata(error, "analysis", spec, metadata)
            # Repair model output only. Transport/auth/context failures are not
            # another chance to diagnose and keep their existing gateway policy.
            rejected_output = any(
                item.get("code") == "invalid_model_output"
                for item in metadata.get("upstream_errors", [])
            )
            if metadata.get("raw_output") and (
                str(error) == "Invalid upstream response schema" or rejected_output
            ):
                metadata["validation_error"] = {
                    "code": "invalid_schema",
                    "reason": str(error),
                }
            calls.append(metadata)
            invalid = bool(metadata.get("validation_error"))
            if invalid and attempt == 0:
                payload = {
                    **payload,
                    "previous_response": metadata.get(
                        "output", metadata.get("raw_output")
                    ),
                    "validation_errors": [metadata["validation_error"]],
                    "repair_attempt": 1,
                }
                continue
            reason = (
                "Analysis output invalid after one repair: "
                if invalid
                else "Analysis call failed: "
            ) + str(error)
            return DiagnosisReport(
                status="ABSTAIN", mode=spec.analysis_mode, reason=reason
            ), _audit(calls, "invalid_output" if invalid else "gateway_error")
        calls.append(metadata)
        return report, _audit(calls, "repaired" if attempt else "valid")
