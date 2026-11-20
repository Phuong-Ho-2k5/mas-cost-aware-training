"""Detect observed abnormalities, then generate grounded causal hypotheses."""
from mas_demo.contracts import DiagnosisReport, AnalysisHypothesis, EpochLogDetectionResponse, EpochLogHypothesesResponse
from mas_demo.llm.gateway import GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway
from mas_demo.observation.facts import build_config_facts, build_epoch_log_payload
from mas_demo.prompts import load_prompt
from mas_demo.training.model import task_description
from mas_demo.agents.analysis_validation import validate_epoch_detection, validate_hypotheses


def _stage(spec, gateway, stage, payload, response_type, validator, calls):
    role = f"analysis_{stage}"
    for attempt in range(2):
        metadata = {}
        try:
            response, metadata = gateway.structured(role, payload, response_type, load_prompt(role))
            metadata = dict(metadata)
            metadata["output"] = response.model_dump()
            errors = validator(response)
            if errors:
                metadata.update(validation_errors=errors, validation_error=errors[0])
                raise GatewayError(errors[0]["reason"])
        except GatewayError as error:
            metadata = failure_metadata(error, "analysis", spec, metadata)
            rejected = any(e.get("code") == "invalid_model_output" for e in metadata.get("upstream_errors", []))
            if metadata.get("finish_reason") == "length":
                errors = [dict(code="output_truncated", reason="Phản hồi trước vượt giới hạn token cố định. Viết cause, tên mechanism và reason ngắn hơn; chỉ viện dẫn epoch thiết yếu. Giữ đủ các hypothesis bắt buộc và detection đã khóa.")]
                metadata.update(validation_errors=errors, validation_error=errors[0])
            elif metadata.get("raw_output") and (str(error) == "Invalid upstream response schema" or rejected):
                errors = [dict(code="invalid_schema", reason="Phản hồi không đúng schema. Trả JSON đầy đủ theo response_format và các ràng buộc của bước hiện tại.")]
                metadata.update(validation_errors=errors, validation_error=errors[0])
            metadata.update(stage=stage, attempt_index=attempt)
            calls.append(metadata)
            invalid = bool(metadata.get("validation_errors"))
            if invalid and attempt == 0:
                payload = {**payload, "previous_response": metadata.get("output", metadata.get("raw_output")),
                           "validation_errors": metadata["validation_errors"], "repair_attempt": 1}
                continue
            return None, dict(outcome="invalid_output" if invalid else "gateway_error", error=str(error),
                              repair_attempted=bool(attempt), repair_succeeded=False)
        metadata.update(stage=stage, attempt_index=attempt)
        calls.append(metadata)
        return response, dict(outcome="repaired" if attempt else "valid", repair_attempted=bool(attempt), repair_succeeded=bool(attempt))


def _audit(spec, calls, stages, outcome, detection, facts, config_facts, report=None, hypotheses=None):
    metadata = dict(calls[-1]) if calls else {}
    # A previous rejected attempt may contain an error; only a failed pipeline
    # has a top-level error. Per-call errors remain untouched in the audit.
    metadata.pop("error", None)
    usage = {}
    for call in calls:
        values = call.get("usage")
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[name] = usage.get(name, 0) + value
    repaired = any(c["attempt_index"] == 1 for c in calls)
    metadata.update(agent="analysis", mode=spec.analysis_mode, model=spec.model_id,
        pipeline_version=2, input_format="epoch_logs", outcome=outcome, detection=detection.model_dump() if detection else None,
        facts=[f.model_dump(exclude_none=True) for f in facts],
        config_facts=[f.model_dump() for f in config_facts], stages=stages, calls=calls,
        output=report.model_dump(exclude={"mode"}) if report else None,
        generated_hypotheses=hypotheses.model_dump()["hypotheses"] if hypotheses else None,
        repair_attempted=repaired, repair_succeeded=repaired and report is not None,
        attempts=sum(c.get("attempts", 0) for c in calls),
        latency_seconds=sum(c.get("latency_seconds", 0) for c in calls), usage=usage)
    if report is None:
        failed = stages.get("hypotheses", stages.get("detection", {}))
        metadata["error"] = failed.get("error", outcome)
    return metadata


def diagnose(spec, evidence, history, gateway=None, current_config=None):
    gateway = gateway or create_gateway(spec)
    current_config = current_config or spec.training
    calls, stages = [], {}
    try:
        payload = build_epoch_log_payload(evidence)
        facts = []
        config_facts = build_config_facts(current_config)
    except ValueError as error:
        stages["detection"] = dict(outcome="invalid_input", error=str(error))
        return None, _audit(spec, calls, stages, "detection_failed", None, [], [])
    detection, stages["detection"] = _stage(spec, gateway, "detection",
        payload, EpochLogDetectionResponse,
        lambda response: validate_epoch_detection(response, evidence), calls)
    if detection is None:
        return None, _audit(spec, calls, stages, "detection_failed", None, facts, config_facts)
    if detection.status == "ABSTAIN":
        report = DiagnosisReport(status="ABSTAIN", mode=spec.analysis_mode, reason=detection.reason)
        return report, _audit(spec, calls, stages, "abstained", detection, facts, config_facts, report)
    payload = dict(**payload,
        history=sorted(history, key=lambda row: row.get("round_id", 0))[-4:],
        current_config=current_config.model_dump(),
        task=task_description(spec), detection=detection.model_dump())
    hypotheses, stages["hypotheses"] = _stage(spec, gateway, "hypotheses", payload,
        EpochLogHypothesesResponse, lambda response: validate_hypotheses(response, evidence, facts, config_facts), calls)
    if hypotheses is None:
        return None, _audit(spec, calls, stages, "hypothesis_generation_failed", detection, facts, config_facts)
    report = DiagnosisReport(status="DIAGNOSED", mode=spec.analysis_mode,
        hypotheses=[AnalysisHypothesis.model_validate(h.model_dump(include=set(AnalysisHypothesis.model_fields))) for h in hypotheses.hypotheses],
        reason=f"{detection.reason} {hypotheses.reason}")
    return report, _audit(spec, calls, stages, "diagnosed", detection, facts, config_facts, report, hypotheses)
