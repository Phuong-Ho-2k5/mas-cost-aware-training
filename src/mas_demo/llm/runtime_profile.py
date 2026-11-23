"""Pinned serving profiles and evidence bound to their exact configuration."""

from copy import deepcopy
import hashlib
import json
import re

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
COMPACT_DECODER = {"backend": "xgrammar", "disable_any_whitespace": True}
CHECKS = ("json_requests", "long_context", "context_rejected", "truncation_rejected",
          "cancellation", "resource_headroom")
PINNED = {
    "model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
    "dtype": "float16", "quantization": None, "language_model_only": True,
    "max_num_seqs": 1, "max_num_batched_tokens": 1024, "tensor_parallel_size": 1,
    "enforce_eager": True, "gpu_memory_utilization": 0.65,
    "generation_config": "vllm", "chat_template_kwargs": {"enable_thinking": False},
    "host": "127.0.0.1",
}


def validate_serving_profile(profile):
    if (not isinstance(profile, dict)
        or any(profile.get(k) != v or type(profile.get(k)) is not type(v) for k, v in PINNED.items())
        or type(profile.get("max_model_len")) is not int or profile["max_model_len"] not in (4096, 8192)
        or type(profile.get("port")) is not int or not 1 <= profile["port"] <= 65535):
        raise ValueError("Unqualified vLLM runtime profile")
    decoder = profile.get("structured_outputs_config")
    if decoder is not None and decoder != COMPACT_DECODER:
        raise ValueError("Unqualified structured output configuration")
    if profile["max_model_len"] == 8192 and decoder != COMPACT_DECODER:
        raise ValueError("8192 profile requires compact structured output configuration")
    reasoning_parser = profile.get("reasoning_parser")
    if reasoning_parser is not None and reasoning_parser != "qwen3":
        raise ValueError("Unqualified reasoning parser")
    return deepcopy(profile)


def profile_fingerprint(profile):
    canonical = validate_serving_profile(profile)
    canonical.pop("port")
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def expected_serving_argv(profile):
    profile = validate_serving_profile(profile)
    argv = ["/opt/mas-vllm/venv/bin/python", "/opt/mas-vllm/venv/bin/vllm", "serve", profile["model"]]
    for flag, value in (
        ("--revision", profile["revision"]), ("--tokenizer-revision", profile["tokenizer_revision"]),
        ("--served-model-name", profile["model"]), ("--host", profile["host"]),
        ("--port", profile["port"]), ("--dtype", profile["dtype"]), ("--language-model-only", None),
        ("--max-model-len", profile["max_model_len"]), ("--max-num-seqs", profile["max_num_seqs"]),
        ("--max-num-batched-tokens", profile["max_num_batched_tokens"]),
        ("--tensor-parallel-size", profile["tensor_parallel_size"]), ("--enforce-eager", None),
        ("--gpu-memory-utilization", profile["gpu_memory_utilization"]),
        ("--generation-config", profile["generation_config"]),
        ("--default-chat-template-kwargs", json.dumps(profile["chat_template_kwargs"], separators=(",", ":"))),
    ):
        argv.append(flag)
        if value is not None:
            argv.append(str(value))
    if profile.get("structured_outputs_config") is not None:
        argv.extend(["--structured-outputs-config", json.dumps(profile["structured_outputs_config"], separators=(",", ":"))])
    if profile.get("reasoning_parser") is not None:
        argv.extend(["--reasoning-parser", profile["reasoning_parser"]])
    return argv


def validate_runtime_qualification(profile, evidence, lock_sha256):
    fingerprint = profile_fingerprint(profile)
    if (not isinstance(evidence, dict)
        or evidence.get("hardware_gate_passed") is not True
        or evidence.get("provenance_verified") is not True
        or evidence.get("error") or evidence.get("cleanup_errors")
        or any(evidence.get("checks", {}).get(k) is not True for k in CHECKS)):
        raise ValueError("Incomplete runtime qualification")
    report_ref = evidence.get("probe_report", {})
    if (evidence.get("cleanup", {}).get("remaining_pids") != []
        or not isinstance(report_ref.get("path"), str) or not report_ref["path"]
        or not isinstance(report_ref.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", report_ref["sha256"])):
        raise ValueError("Runtime qualification is missing cleanup or original report evidence")
    if (evidence.get("profile_fingerprint") != fingerprint
        or profile_fingerprint(evidence.get("profile")) != fingerprint):
        raise ValueError("Runtime qualification profile mismatch")
    if evidence.get("launch_identity", {}).get("effective_argv") != expected_serving_argv(evidence["profile"]):
        raise ValueError("Runtime qualification launch command differs from profile")
    if evidence.get("requirements_lock", {}).get("sha256") != lock_sha256:
        raise ValueError("Runtime qualification lock mismatch")
    versions = evidence.get("versions", {})
    if (not isinstance(versions, dict)
        or any(not isinstance(versions.get(k), str) or not versions[k]
               for k in ("vllm", "torch", "transformers", "triton"))
        or not isinstance(evidence.get("torch_cuda"), str) or not evidence["torch_cuda"]):
        raise ValueError("Runtime qualification package metadata missing")
    return evidence
