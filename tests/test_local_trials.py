import pytest
from mas_demo.contracts import (
    TaskSpec,
    ConfigPlanResponse,
    EvidenceWindow,
    DiagnosisReport,
    Hypothesis,
)
from mas_demo.configuration import training_capabilities, config_hash, compile_plan
from mas_demo.agents.planner import plan_with_agent
from mas_demo.llm.gateway import OpenAICompatibleGateway


@pytest.mark.parametrize("count", [1, 2, 3])
def test_local_schema_accepts_multiple_model_generated_trials(provider, count):
    upstream, state = provider
    gateway = OpenAICompatibleGateway(
        "", base_url=upstream.base_url, mode="local", attempts=1
    )
    spec = TaskSpec()
    proposals = [
        {
            "proposal_id": f"p{i}",
            "hypothesis_id": "h1",
            "patch": {
                "base_config_hash": config_hash(spec.training),
                "changes": {"learning_rate": lr},
                "reason": "Verify",
                "evidence_epochs": [1, 2],
            },
            "expected_effect": "Improve",
        }
        for i, lr in enumerate([0.027, 0.043, 0.067][:count])
    ]
    state["responses"]["training_plan"] = {
        "proposals": proposals,
        "verification_epochs": 3,
        "reason": "Compare",
    }
    evidence = EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY")
    diagnosis = DiagnosisReport(
        status="DIAGNOSED",
        mode="local",
        hypotheses=[
            Hypothesis(
                hypothesis_id="h1",
                cause="Stalled",
                evidence_epochs=[1, 2],
                confidence=0.6,
            )
        ],
    )
    response, metadata = plan_with_agent(
        spec,
        diagnosis,
        evidence,
        spec.training,
        training_capabilities(),
        [],
        1,
        8,
        800,
        {},
        gateway=gateway,
    )
    assert response is not None, metadata
    assert (
        len(
            compile_plan(
                spec.training, response, spec.policy, [1, 2], 8, {"h1"}
            ).candidates
        )
        == count
    )
    assert response.verification_epochs == 3
    sent = state["requests"][0][2]["response_format"]["json_schema"]["schema"]
    assert sent["properties"]["proposals"]["maxItems"] == 3


def test_patch_schema_does_not_offer_null_for_required_numeric_settings():
    from mas_demo.llm.gateway import response_schema
    from jsonschema import Draft202012Validator

    schema = response_schema("planner", ConfigPlanResponse, "local")
    patch = schema["$defs"]["TrainingPatch"]
    validator = Draft202012Validator(patch)
    assert not validator.is_valid({"learning_rate": None})
    assert validator.is_valid({"grad_clip_norm": None})
    assert validator.is_valid({})
