from copy import deepcopy

import pytest

from mas_demo.agents.analysis import diagnose
from mas_demo.contracts import DiagnosisReport, EvidenceWindow, TaskSpec
from mas_demo.llm.gateway import GatewayError


def output(status="DIAGNOSED"):
    return {
        "status": status,
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "cause": "Loss oscillates",
                "evidence_epochs": [1, 2],
                "confidence": 0.7,
            }
        ],
        "reason": "Loss varies across available epochs",
    }


class ScriptedGateway:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, payload):
        self.requests.append(deepcopy(payload))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return DiagnosisReport(mode="local", **result), {
            "attempts": 1,
            "latency_seconds": 2.5,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "raw_output": str(result),
        }


def run(gateway):
    return diagnose(
        TaskSpec(),
        EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY"),
        [],
        gateway=gateway,
    )


@pytest.mark.parametrize(
    "corrected",
    [
        output(),
        {"status": "ABSTAIN", "hypotheses": [], "reason": "Insufficient evidence"},
    ],
)
def test_repairs_contradictory_output_and_accepts_model_chosen_status(corrected):
    invalid = output("ABSTAIN")
    gateway = ScriptedGateway([invalid, corrected])
    report, meta = run(gateway)
    assert report.status == corrected["status"]
    assert report.model_dump(exclude={"mode"}) == corrected
    assert len(gateway.requests) == 2
    first, second = gateway.requests
    assert second["evidence"] == first["evidence"]
    assert second["current_config"] == first["current_config"]
    assert second["previous_response"] == invalid
    assert second["validation_errors"][0]["code"] == "contradictory_status"
    assert second["repair_attempt"] == 1
    assert meta["repair_succeeded"] and meta["repair_attempted"]
    assert "error" not in meta and "validation_error" not in meta
    assert meta["attempts"] == 2 and meta["latency_seconds"] == 5
    assert meta["usage"]["total_tokens"] == 30
    assert meta["calls"][0]["validation_error"]["code"] == "contradictory_status"
    assert len(meta["calls"]) == 2


def test_stops_after_one_failed_repair_and_marks_fallback():
    gateway = ScriptedGateway([output("ABSTAIN"), output("ABSTAIN"), output()])
    report, meta = run(gateway)
    assert len(gateway.requests) == 2
    assert report.status == "ABSTAIN" and report.hypotheses == []
    assert "invalid" in report.reason.lower()
    assert meta["outcome"] == "invalid_output"
    assert meta["repair_attempted"] and not meta["repair_succeeded"]
    assert meta["error"] and len(meta["calls"]) == 2


def test_does_not_repair_transport_errors_or_retry_valid_abstention():
    error = GatewayError(
        "Upstream transport failure", {"attempts": 1, "latency_seconds": 3}
    )
    gateway = ScriptedGateway([error])
    report, meta = run(gateway)
    assert len(gateway.requests) == 1 and meta["outcome"] == "gateway_error"
    assert not meta["repair_attempted"]
    gateway = ScriptedGateway(
        [{"status": "ABSTAIN", "hypotheses": [], "reason": "Healthy"}]
    )
    report, meta = run(gateway)
    assert report.reason == "Healthy" and len(gateway.requests) == 1
    assert meta["outcome"] == "valid" and "error" not in meta


@pytest.mark.parametrize(
    "message,extra",
    [
        ("Invalid upstream response schema", {}),
        (
            "Upstream HTTP 422",
            {"upstream_errors": [{"code": "invalid_model_output", "status": 422}]},
        ),
    ],
)
def test_repairs_schema_failure_with_raw_response(message, extra):
    gateway = ScriptedGateway(
        [
            GatewayError(
                message,
                {
                    "raw_output": '{"status":"unknown"}',
                    "attempts": 1,
                    "latency_seconds": 1,
                    **extra,
                },
            ),
            output(),
        ]
    )
    report, meta = run(gateway)
    assert report.status == "DIAGNOSED" and meta["repair_succeeded"]
    assert gateway.requests[1]["previous_response"] == '{"status":"unknown"}'
    assert gateway.requests[1]["validation_errors"][0]["code"] == "invalid_schema"


def test_repair_feedback_reaches_real_gateway(provider, monkeypatch):
    gateway, state = provider
    original = gateway.complete

    def complete(payload):
        state["responses"]["training_diagnosis"] = (
            output() if "repair_attempt" in payload else output("ABSTAIN")
        )
        return original(payload)

    monkeypatch.setattr(gateway, "complete", complete)
    report, meta = run(gateway)
    assert report.status == "DIAGNOSED" and meta["repair_succeeded"]
    assert len(state["requests"]) == 2
    assert state["requests"][1][1]["repair_attempt"] == 1


@pytest.mark.parametrize("usage", [None, [], "unavailable"])
def test_optional_provider_usage_cannot_crash_valid_analysis(usage):
    class Gateway:
        def complete(self, payload):
            return DiagnosisReport(mode="local", **output()), {"usage": usage}

    report, meta = run(Gateway())
    assert report.status == "DIAGNOSED" and meta["outcome"] == "valid"
    assert meta["usage"] == {}
    assert meta["calls"][0]["usage"] == usage
