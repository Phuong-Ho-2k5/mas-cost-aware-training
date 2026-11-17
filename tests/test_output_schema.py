from copy import deepcopy
import json

from jsonschema import Draft202012Validator, ValidationError
import pytest

from mas_demo.contracts import ConfigPlanResponse, EvaluationResponse, TaskSpec
from mas_demo.configuration import config_hash, training_capabilities
from mas_demo.llm.gateway import ResponsePayload, response_schema


def planner_payload():
    spec = TaskSpec()
    return {"current_config": spec.training.model_dump(), "base_config_hash": config_hash(spec.training),
            "diagnosis": {"hypotheses": [{"hypothesis_id": "observed"}]},
            "evidence": {"epochs": [3, 4]}, "remaining_epochs": 4, "max_trials": 3,
            "capabilities": training_capabilities()}


def plan(payload, changes=None):
    return {"proposals": [{"proposal_id": "p1", "hypothesis_id": "observed",
              "patch": {"base_config_hash": payload["base_config_hash"],
                        "changes": changes or {"learning_rate": 0.027},
                        "reason": "Test loss", "evidence_epochs": [3, 4]},
              "expected_effect": "Improve loss"}], "verification_epochs": 2, "reason": "Compare"}


def schema(payload):
    return response_schema("planner", ConfigPlanResponse, "vllm", payload["max_trials"], payload=payload)


@pytest.mark.parametrize("changes", [
    {"learning_rate": 0.027, "beta1": 0.9},
    {"learning_rate": 0.027, "scheduler_gamma": 0.1},
    {"optimizer": "adam", "momentum": 0.8},
    {"scheduler": "cosine", "scheduler_step_size": 100},
])
def test_schema_rejects_inactive_fields(changes):
    payload = planner_payload()
    with pytest.raises(ValidationError):
        Draft202012Validator(schema(payload)).validate(plan(payload, changes))


@pytest.mark.parametrize("changes", [
    {"learning_rate": 0.027}, {"optimizer": "adam", "beta1": 0.8},
    {"optimizer": "adamw", "beta2": 0.95, "scheduler": "step", "scheduler_gamma": 0.8},
    {"scheduler": "cosine", "scheduler_t_max": 7},
])
def test_schema_retains_free_values_and_component_switches(changes):
    payload = planner_payload()
    Draft202012Validator(schema(payload)).validate(plan(payload, changes))


@pytest.mark.parametrize("current,changes", [
    ({"optimizer": "adam"}, {"beta1": 0.8}),
    ({"scheduler": "step"}, {"scheduler_gamma": 0.7}),
    ({"optimizer": "adamw", "scheduler": "cosine"}, {"beta2": 0.95, "scheduler_t_max": 7}),
])
def test_active_options_can_be_updated_without_repeating_components(current, changes):
    payload = planner_payload();payload["current_config"].update(current)
    Draft202012Validator(schema(payload)).validate(plan(payload, changes))


@pytest.mark.parametrize("fault", ["hash", "hypothesis", "epoch", "horizon", "empty_horizon", "empty_changes"])
def test_schema_rejects_invalid_references_and_couplings(fault):
    payload = planner_payload();value = plan(payload)
    if fault == "hash":value["proposals"][0]["patch"]["base_config_hash"] = "wrong"
    if fault == "hypothesis":value["proposals"][0]["hypothesis_id"] = "h1"
    if fault == "epoch":value["proposals"][0]["patch"]["evidence_epochs"] = [99]
    if fault == "horizon":value["verification_epochs"] = 5
    if fault == "empty_horizon":value["proposals"] = []
    if fault == "empty_changes":value["proposals"][0]["patch"]["changes"] = {}
    with pytest.raises(ValidationError):Draft202012Validator(schema(payload)).validate(value)


def test_schema_accepts_abstention_and_disallows_trials_when_none_available():
    payload = planner_payload();empty = {"proposals": [], "verification_epochs": None, "reason": "Observe"}
    Draft202012Validator(schema(payload)).validate(empty)
    payload["max_trials"] = 0
    with pytest.raises(ValidationError):Draft202012Validator(schema(payload)).validate(plan(payload))


@pytest.mark.parametrize("field", ["reason", "expected_effect"])
def test_planner_nested_text_has_a_generation_bound(field):
    payload = planner_payload()
    value = plan(payload)
    target = value["proposals"][0]["patch"] if field == "reason" else value["proposals"][0]
    target[field] = "x" * 120
    Draft202012Validator(schema(payload)).validate(value)
    target[field] = "x" * 121
    with pytest.raises(ValidationError):
        Draft202012Validator(schema(payload)).validate(value)


def test_reference_schema_without_payload_remains_available_for_scoring():
    definitions = response_schema("planner", ConfigPlanResponse)["$defs"]
    assert "maxLength" not in definitions["ConfigPatch"]["properties"]["reason"]
    assert "maxLength" not in definitions["ConfigProposal"]["properties"]["expected_effect"]


def test_analysis_status_and_evidence_are_grounded():
    payload = {"evidence": {"epochs": [3, 4]}}
    s = response_schema("analysis", ResponsePayload, "vllm", payload=payload)
    good = {"status": "ABSTAIN", "hypotheses": [], "reason": "Observe"}
    Draft202012Validator(s).validate(good)
    bad = {**good, "status": "DIAGNOSED"}
    with pytest.raises(ValidationError):Draft202012Validator(s).validate(bad)
    hypothesis = {"hypothesis_id": "h1", "cause": "Loss", "confidence": 0.7, "evidence_epochs": [99]}
    with pytest.raises(ValidationError):Draft202012Validator(s).validate({**bad, "hypotheses": [hypothesis]})
    with pytest.raises(ValidationError):Draft202012Validator(s).validate({**good, "hypotheses": [hypothesis]})


def test_evaluation_branches_follow_comparable_snapshots():
    payload = {"control": {"epoch": 4}, "trials": {"A": {"epoch": 4}}, "eligible_branches": ["A"]}
    s = response_schema("evaluation", EvaluationResponse, "vllm", payload=payload)
    Draft202012Validator(s).validate({"decision": "PROMOTE", "selected_branch": "A", "reason": "Gain"})
    with pytest.raises(ValidationError):Draft202012Validator(s).validate({"decision": "KEEP_MAIN", "selected_branch": "A", "reason": "Keep"})
    payload["control"] = None
    s = response_schema("evaluation", EvaluationResponse, "vllm", payload=payload)
    with pytest.raises(ValidationError):Draft202012Validator(s).validate({"decision": "KEEP_MAIN", "selected_branch": "main", "reason": "Keep"})
    Draft202012Validator(s).validate({"decision": "INCONCLUSIVE", "selected_branch": "main", "reason": "No control"})
