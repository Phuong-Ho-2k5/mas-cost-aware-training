import pytest

from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.configuration import config_hash, training_capabilities
from mas_demo.contracts import TaskSpec, EvidenceWindow


def diagnosis_payload(status="DIAGNOSED"):
    return {
        "status": status,
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "cause": "Learning has stalled",
                "evidence_epochs": [1, 2],
                "confidence": 0.6,
            }
        ],
        "reason": "Verify a possible training problem",
    }


def plan_payload(spec, horizon=3):
    return {
        "proposals": [
            {
                "proposal_id": "p1",
                "hypothesis_id": "h1",
                "patch": {
                    "base_config_hash": config_hash(spec.training),
                    "changes": {"learning_rate": 0.027, "weight_decay": 0.0037},
                    "reason": "Test observed instability",
                    "evidence_epochs": [1, 2],
                },
                "expected_effect": "Improve learning",
            }
        ],
        "verification_epochs": horizon,
        "reason": "Compare at shared horizon",
    }


def test_analysis_and_planner_do_not_receive_action_menus(provider):
    gateway, state = provider
    spec = TaskSpec()
    evidence = EvidenceWindow(epochs=[1, 2], branch_id="main", status="INSUFFICIENT")
    state["responses"]["training_diagnosis"] = diagnosis_payload()
    report, metadata = diagnose(
        spec, evidence, [], gateway=gateway, current_config=spec.training
    )
    assert report.status == "DIAGNOSED", metadata
    state["responses"]["training_plan"] = plan_payload(spec)
    response, metadata = plan_with_agent(
        spec,
        report,
        evidence,
        spec.training,
        training_capabilities(),
        [],
        1,
        8,
        800,
        {"epoch_seconds": 10},
        gateway=gateway,
    )
    assert response.verification_epochs == 3, metadata
    assert response.proposals[0].patch.changes.learning_rate == 0.027
    for _, payload, _ in state["requests"]:
        assert (
            not {"allowed_interventions", "eligible_trials", "lr_factors"}
            & payload.keys()
        )


@pytest.mark.parametrize("status,epochs", [("ABSTAIN", [1, 2]), ("DIAGNOSED", [999])])
def test_analysis_rejects_contradictory_status_and_fabricated_evidence(
    provider, status, epochs
):
    gateway, state = provider
    payload = diagnosis_payload(status)
    payload["hypotheses"][0]["evidence_epochs"] = epochs
    state["responses"]["training_diagnosis"] = payload
    report, metadata = diagnose(
        TaskSpec(),
        EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY"),
        [],
        gateway=gateway,
    )
    assert report.status == "ABSTAIN"
    assert metadata["validation_error"]


def test_planner_can_abstain_without_default_horizon(provider):
    gateway, state = provider
    state["responses"]["training_plan"] = {
        "proposals": [],
        "verification_epochs": None,
        "reason": "Insufficient basis",
    }
    response, metadata = plan_with_agent(
        TaskSpec(),
        None,
        EvidenceWindow(epochs=[], branch_id="main", status="INSUFFICIENT"),
        TaskSpec().training,
        training_capabilities(),
        [],
        1,
        3,
        100,
        {},
        gateway=gateway,
    )
    assert response.proposals == [] and response.verification_epochs is None, metadata
