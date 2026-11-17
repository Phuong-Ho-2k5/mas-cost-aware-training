"""The application must never select a removed inference backend."""

import pytest

from mas_demo.contracts import TaskSpec
from mas_demo.llm import gateway as gateway_module
from mas_demo.llm.settings import create_gateway


def test_default_task_selects_vllm():
    assert TaskSpec().analysis_mode == "vllm"


@pytest.mark.parametrize("mode", ["openrouter", "local", "bundled"])
def test_removed_analysis_modes_are_rejected(mode):
    with pytest.raises(ValueError):
        TaskSpec(analysis_mode=mode)


def test_factory_cannot_bypass_removed_mode_with_unvalidated_spec(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-old-key")
    with pytest.raises(ValueError, match="vLLM|vllm"):
        create_gateway(TaskSpec.model_construct(analysis_mode="openrouter"))


def test_public_gateway_is_vllm_only():
    assert not hasattr(gateway_module, "OpenRouterGateway")
    gateway_type = getattr(gateway_module, "VLLMGateway", None)
    assert gateway_type is not None
    gateway = gateway_type("")
    assert gateway.mode == gateway.local_backend == "vllm"
    assert gateway.base_url == "http://127.0.0.1:8000/v1"
    assert gateway.model == "Qwen/Qwen3.5-0.8B"
    assert gateway.attempts == 1


def test_factory_rejects_removed_bundled_backend(monkeypatch):
    monkeypatch.setenv("LLM_LOCAL_BACKEND", "bundled")
    with pytest.raises(ValueError, match="vLLM|vllm"):
        create_gateway(TaskSpec())


def test_vllm_factory_needs_no_cloud_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_LOCAL_BACKEND", raising=False)
    monkeypatch.delenv("LLM_RUNTIME_MANIFEST", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    gateway = create_gateway(TaskSpec())
    assert gateway.mode == "vllm"
    assert gateway.local_backend == "vllm"
    assert gateway.attempts == 1


@pytest.mark.parametrize("old_mode", ["local", "openrouter"])
@pytest.mark.parametrize("version", [2, 3])
def test_explicit_config_migration_selects_current_vllm_mode(old_mode, version):
    from mas_demo.cli import migrate_legacy_config

    source = {"schema_version": version, "analysis_mode": old_mode,
              "model_id": "old-provider-model"}
    migrated = migrate_legacy_config(source)
    assert migrated.analysis_mode == "vllm"
    assert migrated.model_id == TaskSpec().model_id
    assert source["analysis_mode"] == old_mode
    assert source["model_id"] == "old-provider-model"
