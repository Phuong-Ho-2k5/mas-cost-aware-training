"""Real gateway/HTTP checks; only the WSL process-observation boundary is stubbed."""

import asyncio
from copy import deepcopy
import json
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from mas_demo.contracts import EvidenceWindow, TaskSpec
from mas_demo.llm import health
from mas_demo.llm.context import fit_local_context
from mas_demo.llm.gateway import GatewayError, VLLMGateway
from mas_demo.llm.settings import create_gateway

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


@pytest.fixture
def live_manifest(tmp_path):
    profile = {
        "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
        "dtype": "float16", "quantization": None, "language_model_only": True,
        "max_model_len": 4096, "max_num_seqs": 1, "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": 0.65,
        "generation_config": "vllm", "chat_template_kwargs": {"enable_thinking": False},
        "host": "127.0.0.1", "port": 8001,
        "structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True},
    }
    argv = ["/opt/mas-vllm/venv/bin/python", "/opt/mas-vllm/venv/bin/vllm", "serve", MODEL,
            "--revision", REVISION, "--tokenizer-revision", REVISION,
            "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", "8001",
            "--dtype", "float16", "--language-model-only", "--max-model-len", "4096",
            "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
            "--tensor-parallel-size", "1", "--enforce-eager",
            "--gpu-memory-utilization", "0.65", "--generation-config", "vllm",
            "--default-chat-template-kwargs", '{"enable_thinking":false}',
            "--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":true}']
    process = {"pid": 1234, "process_group": 1234, "create_time": 1794700000.25,
               "effective_argv": argv}
    versions = {"vllm": "0.33.0", "torch": "2.14.0", "transformers": "5.20.2", "triton": "3.8.0"}
    manifest = {"schema_version": 1, "runtime_status": "ready", "ready": True,
                "distro": "MAS-vLLM", "profile": profile, "process": process,
                "versions": versions, "torch_cuda": "13.0"}
    observation = {**deepcopy(process), "versions": deepcopy(versions), "torch_cuda": "13.0",
                   "listeners": [{"pid": 1234, "process_group": 1234,
                                  "create_time": 1794700000.25}]}
    path = tmp_path / "runtime-manifest.json"

    def save():
        path.write_text(json.dumps(manifest), encoding="utf-8")

    save()
    return path, manifest, observation, save


@pytest.fixture
def candidate(provider, live_manifest, monkeypatch):
    upstream, state = provider
    path, manifest, observation, save = live_manifest
    port = urlsplit(upstream.base_url).port
    manifest["profile"]["port"] = port
    for args in (manifest["process"]["effective_argv"], observation["effective_argv"]):
        args[args.index("--port") + 1] = str(port)
    save()
    monkeypatch.setattr(health, "_inspect_vllm_process", lambda pid, port: observation, raising=False)
    gateway = VLLMGateway("", MODEL, upstream.base_url, mode="vllm",
        attempts=1, local_backend="vllm", runtime_manifest=str(path), max_tokens=1024)
    state.update(token_context=4096, response_model=MODEL)
    state["responses"]["training_diagnosis"] = {"status": "ABSTAIN", "hypotheses": [], "reason": "Observe"}
    return gateway, state, manifest, observation, save


def test_vllm_uses_shared_gateway_without_cloud_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "vllm")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("LLM_RUNTIME_MANIFEST", "D:/runtime.json")
    spec = TaskSpec(analysis_mode="vllm", model_id=MODEL)
    gateway = create_gateway(spec)
    assert type(gateway) is VLLMGateway
    assert (gateway.mode, gateway.local_backend, gateway.model) == ("vllm", "vllm", MODEL)
    assert gateway.base_url == "http://127.0.0.1:8001/v1"
    assert gateway.runtime_manifest == "D:/runtime.json"
    assert (gateway.attempts, gateway.timeout, gateway.deadline) == (1, 120, 130)
    assert "runtime.json" not in spec.model_dump_json()


def test_unknown_local_backend_rejected(monkeypatch):
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "typo")
    with pytest.raises(ValueError, match="backend|BACKEND"):
        create_gateway(TaskSpec(analysis_mode="vllm"))
    with pytest.raises(ValueError, match="backend|BACKEND"):
        VLLMGateway("", local_backend="typo")


def test_only_vllm_receives_non_thinking_kwargs(candidate, provider):
    gateway, state, *_ = candidate
    gateway.complete({})
    chat = state["requests"][-1][2]
    token = state["tokenize_requests"][-1]
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    assert token == {**{k: chat[k] for k in ("model", "messages", "response_format", "chat_template_kwargs")},
                     "add_generation_prompt": True, "add_special_tokens": False}


@pytest.mark.parametrize("status,counts", [
    (404, {}), (401, {}), (200, {"count": True, "max_model_len": 4096}),
    (200, {"count": -1, "max_model_len": 4096}),
    (200, {"count": "900", "max_model_len": 4096}),
    (200, {"count": 900, "max_model_len": 0}), (200, []),
])
def test_vllm_token_preflight_fails_closed(candidate, status, counts):
    gateway, state, *_ = candidate
    state.update(tokenize_status=status, tokenize_response=counts)
    with pytest.raises(GatewayError, match="preflight"):
        gateway.complete({})
    assert not state["requests"]


@pytest.mark.parametrize("error", [httpx.ReadTimeout("late"), httpx.ConnectError("offline")])
def test_vllm_token_transport_failure_cannot_reach_chat(candidate, monkeypatch, error):
    gateway, state, *_ = candidate
    original = httpx.AsyncClient.post

    async def fail(client, url, **kwargs):
        if url.endswith("/tokenize"):
            raise error
        return await original(client, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    with pytest.raises(GatewayError, match="preflight"):
        gateway.complete({})
    assert not state["requests"]


def test_vllm_count_must_match_verified_context(candidate):
    gateway, state, *_ = candidate
    state["token_context"] = 32768
    with pytest.raises(GatewayError, match="context|preflight"):
        gateway.complete({})
    assert not state["requests"]


def test_vllm_compaction_preserves_evidence_and_output_reserve(candidate):
    gateway, state, *_ = candidate
    # 900 + 400*4 + 2048 > 4096; after two removals it fits exactly below the cap.
    gateway.max_tokens = 2048
    evidence = {"epochs": [1, 2], "metrics": [{"loss": 1.5}]}
    payload = {"evidence": evidence, "history": [{"round_id": i} for i in range(4)]}
    _, meta = gateway.complete(payload)
    final = state["requests"][-1][1]
    assert final["evidence"] == evidence and len(final["history"]) == 2
    assert len(payload["history"]) == 4
    assert meta["context_compaction"]["history_removed"] == 2
    assert meta["context_compaction"]["prompt_tokens"] == 1700


@pytest.mark.parametrize("role", ["training_diagnosis", "training_plan"])
def test_vllm_repair_compaction_keeps_role_specific_instruction(candidate, role):
    gateway, state, *_ = candidate
    state["token_context"] = 4096
    body = {"model": MODEL, "max_tokens": 2048,
            "messages": [{"role": "system", "content": "schema"},
                         {"role": "user", "content": json.dumps({"evidence": {"epochs": [1, 2]},
                          "previous_response": {"proposals": [{"proposal_id": "p1"}]},
                          "validation_errors": ["invalid"]})}],
            "response_format": {"json_schema": {"name": role}},
            "chat_template_kwargs": {"enable_thinking": False}}
    async def fit():
        async with httpx.AsyncClient(trust_env=False) as client:
            await fit_local_context(client, gateway.base_url, body, {}, {}, backend="vllm")
    asyncio.run(fit())
    final = json.loads(body["messages"][1]["content"])
    assert final["evidence"] == {"epochs": [1, 2]}
    assert final["validation_errors"] == ["invalid"]
    expected = "diagnosis" if role == "training_diagnosis" else "plan"
    assert expected in final["previous_response"]["instruction"]


def test_vllm_oversize_evidence_never_reaches_generation(candidate):
    gateway, state, *_ = candidate
    state["tokenize_response"] = {"count": 3500, "max_model_len": 4096}
    with pytest.raises(GatewayError, match="evidence was kept intact"):
        gateway.complete({"evidence": {"epochs": list(range(100))}})
    assert not state["requests"]


def test_verified_vllm_provenance_is_audited(candidate):
    gateway, state, *_ = candidate
    _, meta = gateway.complete({})
    server = meta["server"]
    assert server["revision"] == server["tokenizer_revision"] == REVISION
    assert server["dtype"] == "float16" and server["context_tokens"] == 4096
    assert server["chat_template_kwargs"] == {"enable_thinking": False}
    assert server["versions"]["vllm"] == "0.33.0" and server["torch_cuda"] == "13.0"
    assert meta["finish_reason"] == "stop" and meta["usage"]


@pytest.mark.parametrize("fault", ["stopped", "dead", "reused_pid", "foreign_listener",
    "missing_revision", "wrong_revision", "wrong_tokenizer", "wrong_version", "missing_manifest"])
def test_vllm_invalid_identity_rejected_before_any_http(candidate, monkeypatch, fault):
    gateway, state, manifest, observed, save = candidate
    if fault == "stopped":
        manifest.update(ready=False, runtime_status="stopped_after_probe")
    elif fault == "dead":
        def dead(*args):
            raise RuntimeError("process not alive")
        monkeypatch.setattr(health, "_inspect_vllm_process", dead, raising=False)
    elif fault == "reused_pid":
        observed["create_time"] += 1
    elif fault == "foreign_listener":
        observed["listeners"][0]["process_group"] += 1
    elif fault in ("missing_revision", "wrong_revision", "wrong_tokenizer"):
        args = observed["effective_argv"]
        flag = "--tokenizer-revision" if fault == "wrong_tokenizer" else "--revision"
        i = args.index(flag)
        if fault == "missing_revision":
            del args[i:i+2]
        else:
            args[i+1] = "wrong"
        # Tampering both manifest argv and observed argv must still be rejected.
        manifest["process"]["effective_argv"] = deepcopy(args)
    elif fault == "wrong_version":
        observed["versions"]["vllm"] = "0.0.0"
    else:
        gateway.runtime_manifest = None
    save()
    with pytest.raises(GatewayError, match="runtime|manifest|identity|process"):
        gateway.complete({})
    assert not state.get("tokenize_requests") and not state["requests"]


def test_readiness_cannot_trust_models_without_live_provenance(candidate):
    gateway, state, manifest, _, save = candidate
    state["served_models"] = [MODEL]
    assert health.wait_for_server(gateway, timeout=1)["data"][0]["id"] == MODEL
    manifest["ready"] = False
    save()
    with pytest.raises(RuntimeError, match="manifest|ready|runtime"):
        health.wait_for_server(gateway, timeout=1)


def test_gateway_revalidates_identity_on_later_agent_calls(candidate):
    gateway, state, _, observed, _ = candidate
    gateway.complete({})
    observed["create_time"] += 1
    with pytest.raises(GatewayError):
        gateway.complete({})
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("finish", ["length", "error", None])
def test_vllm_incomplete_json_never_becomes_valid_abstention(candidate, finish):
    gateway, state, *_ = candidate
    state.update(finish_reason=finish)
    state["responses"]["training_diagnosis"] = {"status": "ABSTAIN", "hypotheses": [], "reason": "Observe"}
    with pytest.raises(GatewayError) as error:
        gateway.complete({})
    assert error.value.metadata["finish_reason"] == finish
    assert error.value.metadata["raw_output"] and error.value.metadata["usage"]


def test_vllm_response_model_mismatch_rejected(candidate):
    gateway, state, *_ = candidate
    state["response_model"] = "different-model"
    with pytest.raises(GatewayError, match="model"):
        gateway.complete({})


def test_runtime_inspection_uses_fixed_structured_read_only_command(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"pid":1234}')

    monkeypatch.setattr(health.subprocess, "run", run)
    assert health._inspect_vllm_process(1234, 8001) == {"pid": 1234}
    command, kwargs = calls[0]
    assert command[-2:] == ["-c", health._INSPECT_VLLM]
    assert json.loads(kwargs["input"]) == {"pid": 1234, "port": 8001}
    assert not kwargs.get("shell") and 0 < kwargs["timeout"] <= 5


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("wsl", 5), OSError("unavailable")])
def test_runtime_inspection_unavailable_fails_closed(monkeypatch, failure):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(health.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="identity"):
        health._inspect_vllm_process(1234, 8001)


def test_runtime_verification_obeys_gateway_absolute_deadline(candidate, monkeypatch):
    gateway, state, *_ = candidate
    def slow(*args):
        time.sleep(0.8)
        raise RuntimeError("offline")
    monkeypatch.setattr(health, "verify_vllm_runtime", slow)
    gateway.deadline = 0.1
    started = time.monotonic()
    with pytest.raises(GatewayError, match="deadline"):
        gateway.complete({})
    assert time.monotonic() - started < 0.5
    assert not state["requests"] and not state.get("tokenize_requests")


def test_three_agents_use_one_verified_vllm_gateway(candidate):
    from mas_demo.agents.analysis import diagnose
    from mas_demo.agents.planner import plan_with_agent
    from mas_demo.agents.evaluation import evaluate_with_agent
    from mas_demo.configuration import training_capabilities
    from mas_demo.contracts import EpochEvent

    gateway, state, *_ = candidate
    spec = TaskSpec(analysis_mode="vllm", model_id=MODEL)
    evidence = EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY")
    state["responses"]["training_diagnosis"] = {"status": "DIAGNOSED", "hypotheses": [
        {"hypothesis_id": "h1", "cause": "Loss unstable", "confidence": 0.6, "evidence_epochs": [1, 2]}], "reason": "Test"}
    diagnosis, analysis_meta = diagnose(spec, evidence, [], gateway)
    plan, plan_meta = plan_with_agent(spec, diagnosis, evidence, spec.training,
        training_capabilities(), [], 1, 8, 800, {}, gateway=gateway)
    def event(accuracy):
        return EpochEvent(epoch=2, branch_id="main", train_loss=1, val_loss=1,
                          val_accuracy=accuracy, learning_rate=0.1, grad_norm=1)
    verdict, eval_meta = evaluate_with_agent(spec, [event(0.5)], {"A": [event(0.6)]}, 2, gateway)
    assert diagnosis.status == "DIAGNOSED" and plan.proposals == [] and verdict.decision == "PROMOTE"
    assert {row[0] for row in state["requests"]} == {"training_diagnosis", "training_plan", "training_evaluation"}
    assert len(state["tokenize_requests"]) == 3
    for meta in (analysis_meta, plan_meta, eval_meta):
        assert "error" not in meta and meta["server"]["revision"] == REVISION


def test_live_kernel_birth_identity_handles_wall_clock_shift(candidate):
    gateway, state, manifest, observed, save = candidate
    manifest["process"].update(start_ticks=123456, boot_id="boot-A")
    observed.update(start_ticks=123456, boot_id="boot-A")
    observed["create_time"] += 2
    observed["listeners"][0].update(start_ticks=123456, boot_id="boot-A")
    save()
    gateway.complete({})
    observed["start_ticks"] += 1
    with pytest.raises(GatewayError, match="runtime"):
        gateway.complete({})
    assert len(state["requests"]) == 1


def test_vllm_audits_effective_schema_without_expanding_system_prompt(candidate):
    gateway, state, *_ = candidate
    _, metadata = gateway.complete({"evidence": {"epochs": [3, 4]}})
    body = state["requests"][-1][2]
    assert metadata["response_schema"] == body["response_format"]["json_schema"]["schema"]
    assert "anyOf" in metadata["response_schema"]
    assert json.dumps(metadata["response_schema"]) not in body["messages"][0]["content"]
    assert json.loads(body["messages"][1]["content"]) == {"evidence": {"epochs": [3, 4]}}


def test_planner_request_excludes_evaluation_only_instructions(candidate):
    from mas_demo.contracts import ConfigPlanResponse
    from mas_demo.prompts import load_prompt

    gateway, state, *_ = candidate
    state["responses"]["training_plan"] = {
        "proposals": [], "verification_epochs": None, "reason": "No feasible trial"}
    gateway.structured("planner", {}, ConfigPlanResponse, load_prompt("planner"))
    system = state["requests"][-1][2]["messages"][0]["content"]
    assert "INCONCLUSIVE" not in system
    assert "PROMOTE" not in system
    assert "eligible_branches" not in system
    assert "response_format" in system


def test_vllm_planner_skips_the_generic_multi_role_wrapper(candidate, monkeypatch):
    from mas_demo.contracts import ConfigPlanResponse
    from mas_demo.llm import gateway as gateway_module
    from mas_demo.prompts import load_prompt

    gateway, state, *_ = candidate
    original = gateway_module.load_prompt
    monkeypatch.setattr(gateway_module, "load_prompt",
                        lambda name: "GENERIC_MULTI_ROLE_WRAPPER" if name == "local" else original(name))
    state["responses"]["training_plan"] = {
        "proposals": [], "verification_epochs": None, "reason": "No feasible trial"}
    gateway.structured("planner", {}, ConfigPlanResponse, load_prompt("planner"))
    assert "GENERIC_MULTI_ROLE_WRAPPER" not in state["requests"][-1][2]["messages"][0]["content"]
    gateway.complete({"evidence": {"epochs": []}})
    assert "GENERIC_MULTI_ROLE_WRAPPER" in state["requests"][-1][2]["messages"][0]["content"]


def test_only_vllm_requests_compact_json_to_prevent_whitespace_loops(candidate, provider):
    gateway, state, *_ = candidate
    _, metadata = gateway.complete({})
    assert "structured_outputs" not in state["requests"][-1][2]
    assert metadata["server"]["structured_outputs_config"] == {"backend": "xgrammar", "disable_any_whitespace": True}
