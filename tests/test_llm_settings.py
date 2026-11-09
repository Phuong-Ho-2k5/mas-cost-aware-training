import json

import pytest

from mas_demo.contracts import TaskSpec, EvidenceWindow
from mas_demo.agents.analysis import diagnose
from mas_demo.llm.gateway import GatewayError, OpenRouterGateway


def factory():
    from mas_demo.llm.settings import create_gateway

    return create_gateway


def test_local_config_needs_no_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    spec = TaskSpec(analysis_mode="local")
    gateway = factory()(spec)
    assert gateway.mode == "local"
    assert gateway.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert gateway.base_url == "http://127.0.0.1:8000/v1"
    assert gateway.attempts == 1
    assert gateway.timeout == 120 and gateway.deadline == 130
    assert "key" not in spec.model_dump()


def test_old_config_keeps_openrouter_and_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        factory()(TaskSpec(analysis_mode="openrouter"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-key")
    gateway = factory()(TaskSpec(analysis_mode="openrouter"))
    assert gateway.mode == "openrouter" and gateway.attempts == 2
    assert gateway.base_url == "https://openrouter.ai/api/v1"
    assert gateway.model == TaskSpec().model_id


def test_local_gateway_preserves_metadata_and_abstention(provider, monkeypatch):
    upstream, state = provider
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gateway = factory()(TaskSpec(analysis_mode="local"))
    state["responses"]["training_diagnosis"] = {
        "status": "ABSTAIN",
        "hypotheses": [],
        "reason": "No evidence",
    }
    report, metadata = diagnose(
        TaskSpec(analysis_mode="local"),
        EvidenceWindow(epochs=[], branch_id="main", status="INSUFFICIENT"),
        [],
        gateway,
    )
    assert report.mode == metadata["mode"] == "local"
    assert report.status == "ABSTAIN"
    state["statuses"]["training_diagnosis"] = 503
    report, metadata = diagnose(
        TaskSpec(analysis_mode="local"),
        EvidenceWindow(epochs=[], branch_id="main", status="INSUFFICIENT"),
        [],
        gateway,
    )
    assert report.mode == metadata["mode"] == "local"
    assert report.status == "ABSTAIN" and metadata["attempts"] == 1
    assert len(state["requests"]) == 2


def test_local_endpoint_validation(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "file:///tmp/model")
    with pytest.raises(ValueError, match="HTTP"):
        factory()(TaskSpec(analysis_mode="local"))


def test_explicit_model_and_config_snapshot_are_preserved(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "server-secret")
    spec = TaskSpec(analysis_mode="local", model_id="served-qwen")
    gateway = factory()(spec)
    assert gateway.model == "served-qwen"
    saved = spec.model_dump_json()
    assert "server-secret" not in saved
    assert TaskSpec.model_validate_json(saved) == spec


def test_local_cli_passes_key_gate_without_calling_cloud(tmp_path, monkeypatch, capsys):
    from mas_demo import cli

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / "local.yaml"
    path.write_text("dataset: synthetic\ndevice: cpu\nanalysis_mode: local\n")
    monkeypatch.setattr(
        "sys.argv",
        ["mas-demo", "run", "--config", str(path), "--output", str(tmp_path / "run")],
    )
    monkeypatch.setattr(cli, "run", lambda spec, output: {"status": "COMPLETED"})
    cli.main()
    assert "COMPLETED" in capsys.readouterr().out


def test_truncated_output_never_becomes_a_decision(provider, monkeypatch):
    upstream, state = provider
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    gateway = factory()(TaskSpec(analysis_mode="local"))
    state["finish_reason"] = "length"
    state["responses"]["training_diagnosis"] = {
        "status": "ABSTAIN",
        "hypotheses": [],
        "reason": "A valid object but truncated completion",
    }
    with pytest.raises(GatewayError, match="schema") as caught:
        gateway.complete({})
    assert caught.value.metadata["finish_reason"] == "length"
    assert json.loads(caught.value.metadata["raw_output"])["status"] == "ABSTAIN"


def test_direct_analysis_factory_uses_local_backend(provider, monkeypatch):
    upstream, state = provider
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    state["responses"]["training_diagnosis"] = {
        "status": "ABSTAIN",
        "hypotheses": [],
        "reason": "No evidence",
    }
    report, meta = diagnose(
        TaskSpec(analysis_mode="local"),
        EvidenceWindow(epochs=[], branch_id="main", status="INSUFFICIENT"),
        [],
    )
    assert report.mode == meta["mode"] == "local"


def test_readiness_checks_served_model_before_benchmark(provider):
    from mas_demo.llm.health import wait_for_server

    gateway, state = provider
    state["served_models"] = [gateway.model]
    assert wait_for_server(gateway, timeout=0.2)["data"][0]["id"] == gateway.model
    state["served_models"] = ["different"]
    with pytest.raises(RuntimeError, match="not serve"):
        wait_for_server(gateway, timeout=0.2)


def test_readiness_fails_with_bounded_wait_when_server_unavailable(provider):
    from mas_demo.llm.health import wait_for_server

    gateway, state = provider
    state["health_status"] = 503
    with pytest.raises(TimeoutError, match="not ready"):
        wait_for_server(gateway, timeout=0.1, interval=0.01)


def test_local_generation_has_explicit_short_output_schema(provider, monkeypatch):
    upstream, state = provider
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    gateway = factory()(TaskSpec(analysis_mode="local"))
    gateway.complete({})
    schema = state["requests"][0][2]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["reason"]["maxLength"] == 160
    assert schema["properties"]["hypotheses"]["maxItems"] == 1
