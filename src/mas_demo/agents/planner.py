"""The Planner generates configuration patches and their shared horizon."""

from mas_demo.contracts import ConfigPlanResponse
from mas_demo.configuration import config_hash
from mas_demo.llm.gateway import GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway
from mas_demo.prompts import load_prompt
from mas_demo.training.model import task_description


def plan_with_agent(
    spec,
    diagnosis,
    evidence,
    current_config,
    capabilities,
    history,
    round_id,
    remaining_epochs,
    remaining_budget_seconds,
    cost_estimate,
    gateway=None,
    previous_response=None,
    validation_errors=None,
):
    gateway = gateway or create_gateway(spec)
    payload = {
        "schema_version": 3,
        "task": task_description(spec),
        "diagnosis": diagnosis.model_dump() if diagnosis else None,
        "evidence": evidence.model_dump(exclude={"signals"}),
        "current_config": current_config.model_dump(),
        "base_config_hash": config_hash(current_config),
        "capabilities": capabilities,
        "history": sorted(history, key=lambda row: row.get("round_id", 0))[-4:],
        "round_id": round_id,
        "remaining_epochs": remaining_epochs,
        "remaining_budget_seconds": remaining_budget_seconds,
        "cost_estimate": cost_estimate,
        "max_trials": spec.max_trials,
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
