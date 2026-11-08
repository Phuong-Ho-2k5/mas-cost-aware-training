import os
from mas_demo.contracts import TrialSpec, PlanResponse
from mas_demo.llm.gateway import OpenRouterGateway, GatewayError, failure_metadata
from mas_demo.llm.settings import create_gateway
from mas_demo.memory.store import normalize_action


def eligible_trials(factors, history, round_id, horizon, cooldown=2, current_settings=None):
    """Construct valid choices only; the LLM decides which trials to run and their order."""
    trials = []
    actions = {a.key: a for value in factors if (a := normalize_action(value)) is not None}
    for index, action in enumerate(actions.values()):
        if current_settings and action.operation == "set" and current_settings.get(action.parameter) == action.value:
            continue
        previous = [r for r in history if normalize_action(r.get("intervention", r.get("factor"))) == action
                    and r["status"] in ("ACCEPT", "REJECT", "FAILED")]
        if any(r["status"] in ("REJECT", "FAILED") and round_id - r["round_id"] <= cooldown for r in previous):
            continue
        trials.append(TrialSpec(trial_id=f"trial-{index + 1}", intervention=action,
                                horizon=horizon))
    return trials


def plan_with_agent(spec, diagnosis, history, round_id, horizon, gateway=None, current_settings=None):
    actions = [a for h in diagnosis.hypotheses for a in h.interventions]
    metadata = {}
    try:
        if any(a not in spec.intervention_allowlist for a in actions):
            raise GatewayError("Diagnosis contains a forbidden intervention")
        candidates = eligible_trials(actions, history, round_id, horizon, current_settings=current_settings)
        gateway = gateway or create_gateway(spec)
        response, metadata = gateway.structured("planner", {
            "schema_version": 2, "diagnosis": diagnosis.model_dump(), "history": history, "round_id": round_id,
            "current_settings": current_settings,
            "eligible_trials": [t.model_dump() for t in candidates], "max_trials": spec.max_trials,
            "horizon": horizon}, PlanResponse,
            "Choose and order eligible trial_ids using diagnosis and observed gain/cost history. "
            "Only use IDs from eligible_trials; do not repeat IDs or invent parameter values. "
            "The list may be empty to abstain. Return trial_ids and reason. Code enforces max_trials.")
        eligible = {t.trial_id: t for t in candidates}
        if len(response.trial_ids) != len(set(response.trial_ids)) or any(t not in eligible for t in response.trial_ids):
            raise GatewayError("Planner returned duplicate or ineligible trial IDs")
        metadata.update(reason=response.reason, output=response.model_dump())
        return [eligible[t] for t in response.trial_ids][:spec.max_trials], metadata
    except GatewayError as error:
        return [], failure_metadata(error, "planner", spec, metadata)
