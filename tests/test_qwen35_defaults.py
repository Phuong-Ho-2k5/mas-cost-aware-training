import importlib.util
import json
from pathlib import Path
import sys

from dotenv import dotenv_values
import pytest

from mas_demo.contracts import TaskSpec
from mas_demo.cli import load_spec
from mas_demo.llm.settings import create_gateway
from mas_demo.llm.health import wait_for_server

ROOT = Path(__file__).resolve().parents[1]
NEW = "Qwen/Qwen3.5-0.8B"
OLD = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIGS = ["autonomous_smoke", "autonomous_healthy", "autonomous_mixed", "autotrainer_oscillating_loss"]


def test_new_task_and_all_shipped_configs_select_qwen35():
    assert TaskSpec().model_id == NEW
    for name in CONFIGS:
        assert load_spec(ROOT / f"configs/{name}.yaml").model_id == NEW


def test_local_factory_defaults_to_verified_vllm_without_cloud_key(monkeypatch):
    for key in ["LLM_BASE_URL", "LLM_LOCAL_BACKEND", "LLM_RUNTIME_MANIFEST", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    gateway = create_gateway(TaskSpec())
    assert gateway.model == NEW and gateway.local_backend == "vllm"
    assert gateway.base_url == "http://127.0.0.1:8000/v1"
    assert Path(gateway.runtime_manifest).resolve() == ROOT / "tmp/vllm/runtime-manifest.json"


def test_saved_old_model_remains_explicit_and_legacy_backend_supported(monkeypatch):
    saved = json.dumps({"analysis_mode": "local", "model_id": OLD})
    spec = TaskSpec.model_validate_json(saved)
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "bundled")
    gateway = create_gateway(spec)
    assert spec.model_id == gateway.model == OLD
    assert gateway.local_backend == "bundled"
    assert TaskSpec.model_validate_json(spec.model_dump_json()).model_id == OLD


def test_old_run_readiness_cannot_silently_use_new_model(provider, monkeypatch):
    upstream, state = provider
    state["served_models"] = [NEW]
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "bundled")
    monkeypatch.setenv("LLM_BASE_URL", upstream.base_url)
    gateway = create_gateway(TaskSpec(model_id=OLD))
    with pytest.raises(RuntimeError, match="not serve"):
        wait_for_server(gateway, timeout=0.2)
    assert gateway.model == OLD


@pytest.mark.parametrize("script", ["check_local_agents.py", "benchmark_local_agents.py"])
@pytest.mark.parametrize("override", [None, OLD])
def test_probe_and_benchmark_model_defaults_and_overrides(monkeypatch, tmp_path, script, override):
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location("default_cli_" + script[:-3], path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_environment", lambda: None)
    args = [str(path), "--output", str(tmp_path / "unused")]
    if override:args += ["--model", override]
    monkeypatch.setattr(sys, "argv", args)
    captured = []
    class Captured(RuntimeError):pass
    def capture(spec):
        captured.append(spec);raise Captured()
    monkeypatch.setattr(module, "create_gateway", capture)
    with pytest.raises(Captured):module.main()
    assert captured[0].model_id == (override or NEW)


def test_env_example_uses_vllm_but_keeps_legacy_launcher_model():
    values = dotenv_values(ROOT / ".env.example")
    assert values["LLM_LOCAL_BACKEND"] == "vllm"
    assert values["LLM_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert values["LLM_RUNTIME_MANIFEST"] == "tmp/vllm/runtime-manifest.json"
    assert values["LOCAL_LLM_MODEL"] == OLD
