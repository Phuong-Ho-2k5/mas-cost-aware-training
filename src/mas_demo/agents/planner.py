"""The Planner generates configuration patches and their shared horizon."""

from mas_demo.contracts import ConfigPlanResponse
from mas_demo.configuration import config_hash
from mas_demo.llm.gateway import GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway
from mas_demo.observation.facts import clean_evidence_for_llm
from mas_demo.prompts import load_prompt
from mas_demo.training.model import model_summary, task_description


def plan_with_agent(
    spec,
    diagnosis,
    evidence,
    current_config,
    capabilities,
    history,
    round_id,
    remaining_epochs,
    gateway=None,
    previous_response=None,
    validation_errors=None,
):
    gateway = gateway or create_gateway(spec)
    history_fields = {"round_id", "base_config_hash", "config_hash", "proposal_id", "hypothesis_id",
                      "hypothesis_rank", "status", "diff", "gain", "verification_epochs",
                      "comparison_epoch"}
    experiment_history = [{key: value for key, value in row.items() if key in history_fields}
                          for row in history if row.get("status") in ("ACCEPT", "REJECT", "FAILED", "NOT_EXECUTED")]
    payload = {
        "schema_version": 3,
        "task": task_description(spec),
        "model_summary": model_summary(spec, current_config),
        # The hypotheses retain causes, confidence and evidence references;
        # Analysis's long prose remains in its own log instead of being repeated.
        "diagnosis": diagnosis.model_dump(exclude={"mode", "reason"}) if diagnosis else None,
        "evidence": clean_evidence_for_llm(evidence),
        "current_config": current_config.model_dump(),
        "base_config_hash": config_hash(current_config),
        "capabilities": capabilities,
        "history": sorted(experiment_history, key=lambda row: row.get("round_id", 0))[-4:],
        "round_id": round_id,
        "remaining_epochs": remaining_epochs,
        "max_verification_epochs": min(remaining_epochs, spec.policy.max_verification_epochs),
        "required_trial_count": len(diagnosis.hypotheses) if diagnosis and diagnosis.status == "DIAGNOSED" else 0,
        "comparison_protocol": "Same complete epochs from same fork for control and all trials.",
    }
    if validation_errors is not None:
        payload.update(
            previous_response=previous_response,
            validation_errors=validation_errors,
            repair_attempt=1,
        )
    try:
        response, metadata = gateway.structured(
            "planner", payload, ConfigPlanResponse, load_prompt("planner")
        )
        metadata["output"] = response.model_dump(exclude_unset=True)
        return response, metadata
    except GatewayError as error:
        return None, failure_metadata(error, "planner", spec)


def repair_plan_with_agent(*args, **kwargs):
    return plan_with_agent(*args, **kwargs)
