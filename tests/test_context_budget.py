from mas_demo.contracts import TaskSpec, EvidenceWindow
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.configuration import training_capabilities
from mas_demo.llm.gateway import OpenAICompatibleGateway


def test_context_budget_drops_optional_old_history_but_preserves_evidence(provider):
    upstream, state = provider
    state["token_context"] = 2500
    state["responses"]["training_diagnosis"] = {
        "status": "ABSTAIN",
        "hypotheses": [],
        "reason": "Observe",
    }
    gateway = OpenAICompatibleGateway(
        "", base_url=upstream.base_url, mode="local", attempts=1, max_tokens=1024
    )
    evidence = EvidenceWindow(epochs=[1, 2, 3, 4, 5], branch_id="main", status="READY")
    history = [{"round_id": i, "config_hash": str(i)} for i in [1, 2, 3]]
    report, metadata = diagnose(TaskSpec(), evidence, history, gateway=gateway)
    assert report.status == "ABSTAIN" and "error" not in metadata
    final = state["requests"][-1][1]
    assert final["evidence"] == evidence.model_dump(exclude={"signals"})
    assert final["history"] == history[-1:]
    assert metadata["context_compaction"]["history_removed"] == 2
    assert len(history) == 3


def test_repair_can_reference_audited_previous_output_without_truncating_evidence(
    provider,
):
    upstream, state = provider
    state["token_context"] = 2500
    state["responses"]["training_plan"] = {
        "proposals": [],
        "verification_epochs": None,
        "reason": "Abstain",
    }
    gateway = OpenAICompatibleGateway(
        "", base_url=upstream.base_url, mode="local", attempts=1, max_tokens=1024
    )
    spec = TaskSpec()
    evidence = EvidenceWindow(epochs=[1, 2, 3, 4, 5], branch_id="main", status="READY")
    result, metadata = plan_with_agent(
        spec,
        None,
        evidence,
        spec.training,
        training_capabilities(),
        [],
        1,
        8,
        800,
        {},
        gateway=gateway,
        previous_response={"proposals": [{"proposal_id": "p1", "data": "large"}]},
        validation_errors=[{"code": "INACTIVE_FIELD", "fields": ["momentum"]}],
    )
    assert result is not None, metadata
    assert (
        state["requests"][-1][1]["previous_response"]["stored_in"]
        == "previous planning event"
    )
    assert state["requests"][-1][1]["evidence"]["epochs"] == [1, 2, 3, 4, 5]


def test_analysis_repair_reference_uses_analysis_instructions(provider, monkeypatch):
    upstream, state = provider
    state["token_context"] = 2500
    gateway = OpenAICompatibleGateway(
        "", base_url=upstream.base_url, mode="local", attempts=1, max_tokens=1024
    )
    original = gateway.complete

    def complete(payload):
        state["responses"]["training_diagnosis"] = {
            "status": "ABSTAIN",
            "hypotheses": (
                []
                if payload.get("repair_attempt")
                else [
                    {
                        "hypothesis_id": "h1",
                        "cause": "Loss unstable",
                        "evidence_epochs": [1, 2],
                        "confidence": 0.7,
                    }
                ]
            ),
            "reason": "Check observations",
        }
        return original(payload)

    monkeypatch.setattr(gateway, "complete", complete)
    evidence = EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY")
    report, meta = diagnose(TaskSpec(), evidence, [], gateway=gateway)
    assert meta["repair_succeeded"] and report.status == "ABSTAIN"
    repaired = state["requests"][-1][1]
    assert repaired["evidence"] == evidence.model_dump(exclude={"signals"})
    assert "analysis" in repaired["previous_response"]["stored_in"]
    assert "diagnosis" in repaired["previous_response"]["instruction"]
    assert repaired["validation_errors"][0]["code"] == "contradictory_status"
    assert meta["calls"][1]["context_compaction"]["previous_response_referenced"]
