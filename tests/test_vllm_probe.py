"""Probe failures must not be reported as hardware/model feasibility success."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_vllm_qwen.py"


def probe_module():
    assert SCRIPT.exists(), "Task 2 probe script has not been implemented"
    spec = importlib.util.spec_from_file_location("vllm_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenization_uses_same_chat_template_and_schema():
    module = probe_module()
    body = module.chat_body("test-model", "hello")
    token_body = module.token_body(body)
    assert token_body["model"] == "test-model"
    assert token_body["messages"] == body["messages"]
    assert token_body["response_format"] == body["response_format"]
    assert token_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert token_body["add_generation_prompt"] is True
    assert token_body["add_special_tokens"] is False


def test_completion_gate_rejects_length_and_token_count_mismatch():
    module = probe_module()
    body = module.chat_body("test-model", "hello")
    response = {"model": "test-model", "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"hello"}'}}], "usage": {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25}}
    assert module.validate_completion(body, response, 17)["answer"] == "hello"
    with pytest.raises(ValueError, match="token"):
        module.validate_completion(body, response, 18)
    response["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="finish"):
        module.validate_completion(body, response, 17)


def test_completion_gate_checks_full_schema_and_model_identity():
    module = probe_module()
    body = module.chat_body("test-model", "hello")
    response = {"model": "wrong-model", "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"hello"}'}}], "usage": {"prompt_tokens": 1}}
    with pytest.raises(ValueError, match="model"):
        module.validate_completion(body, response, 1)
    response["model"] = "test-model"
    response["choices"][0]["message"]["content"] = '{"answer":123}'
    with pytest.raises(ValueError):
        module.validate_completion(body, response, 1)


def test_scheduler_metrics_sum_all_engines_and_require_presence():
    module = probe_module()
    metrics = 'vllm:num_requests_running{model_name="x",engine="0"} 1\nvllm:num_requests_running{model_name="x",engine="1"} 2\nvllm:num_requests_waiting{model_name="x"} 0\n'
    assert module.scheduler_counts(metrics) == {"running": 3, "waiting": 0}
    with pytest.raises(ValueError, match="metrics"):
        module.scheduler_counts("# no scheduler samples\n")


def test_probe_cannot_pass_without_actual_cancellation_and_long_context():
    module = probe_module()
    checks = {"json_requests": True, "long_context": True, "context_rejected": True, "truncation_rejected": True, "cancellation": False, "resource_headroom": True}
    assert module.hardware_gate(checks, provenance_verified=True) is False
    checks["cancellation"] = True
    assert module.hardware_gate(checks, provenance_verified=True) is True
    assert module.hardware_gate(checks) is False


def test_cancellation_does_not_accept_idle_observed_after_deadline():
    module = probe_module()
    elapsed = [0.0]
    def late_response(remaining):
        elapsed[0] += 6
        return {"running": 0, "waiting": 0}
    with pytest.raises(TimeoutError):
        module.wait_for_idle(late_response, timeout=5, clock=lambda: elapsed[0])


def test_headroom_cannot_use_startup_only_or_missing_inference_samples():
    module = probe_module()
    samples = [{"phase": "startup", "device_vram_used_mib": 100, "device_vram_free_mib": 3800},
               {"phase": "inference", "gpu_measurement_unavailable": "timeout"}]
    summary = module.summarize_resources(samples)
    assert module.resource_headroom(summary) is False
    samples.append({"phase": "inference", "device_vram_used_mib": 2900, "device_vram_free_mib": 1000})
    assert module.resource_headroom(module.summarize_resources(samples)) is False
    assert module.resource_headroom(module.summarize_resources([samples[0], samples[2]])) is True


def test_final_report_survives_resource_cleanup_error(tmp_path):
    module = probe_module()
    class BrokenResources:
        def finish(self):
            raise RuntimeError("sampler broke")
    report = {"checks": {name: False for name in module.CHECKS}, "provenance_verified": False}
    module.finalize_probe(report, BrokenResources(), None, tmp_path, module.time.monotonic())
    import json
    saved = json.loads((tmp_path / "probe.json").read_text())
    assert saved["hardware_gate_passed"] is False
    assert saved["cleanup_errors"]


@pytest.mark.parametrize("c_free,d_free", [(3.9, 28), (15, 9.9), (float("nan"), 28)])
def test_host_storage_guard_rejects_low_or_unknown_physical_capacity(c_free, d_free):
    module = probe_module()
    with pytest.raises(ValueError, match="storage"):
        module.validate_host_storage({"c_free_gib": c_free, "d_free_gib": d_free})


def test_host_storage_guard_accepts_reserved_capacity():
    module = probe_module()
    module.validate_host_storage({"c_free_gib": 15.5, "d_free_gib": 28})
