"""Bounded readiness check for an OpenAI-compatible serving process."""
import time
import json
import math
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import httpx
from mas_demo.llm.runtime_profile import (
    MODEL, profile_fingerprint, validate_serving_profile, validate_runtime_qualification,
)


# Fixed read-only program. Manifest values travel as JSON stdin, never shell code.
_INSPECT_VLLM = r'''
import ast
from importlib import metadata
import json
import os
import sys
import psutil
from pathlib import Path

def birth(pid):
    ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    return {"start_ticks": ticks, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}

request = json.load(sys.stdin)
process = psutil.Process(request["pid"])
if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
    raise RuntimeError("vLLM process is not alive")
listeners = []
for connection in psutil.net_connections(kind="tcp"):
    if connection.status == psutil.CONN_LISTEN and connection.laddr.port == request["port"]:
        if connection.pid is None:
            raise RuntimeError("Unknown listener owner")
        owner = psutil.Process(connection.pid)
        listeners.append({"pid": owner.pid, "process_group": os.getpgid(owner.pid),
                          "create_time": owner.create_time(), "address": connection.laddr.ip, **birth(owner.pid)})
cuda = None
version_file = metadata.distribution("torch").locate_file("torch/version.py")
for node in ast.parse(version_file.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cuda" for t in node.targets):
        cuda = ast.literal_eval(node.value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "cuda":
        cuda = ast.literal_eval(node.value)
print(json.dumps({"pid": process.pid, "process_group": os.getpgid(process.pid),
    "create_time": process.create_time(), "effective_argv": process.cmdline(), "listeners": listeners,
    "versions": {name: metadata.version(name) for name in ("vllm", "torch", "transformers", "triton")},
    "torch_cuda": cuda, **birth(process.pid)}))
'''


def kernel_birth(pid):
    """Linux birth identity unaffected by wall-clock/WSL boot-time correction."""
    ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    return {"start_ticks": ticks, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def same_process_birth(expected, observed):
    if "start_ticks" in expected or "boot_id" in expected:
        return (type(expected.get("start_ticks")) is int and expected["start_ticks"] > 0
                and isinstance(expected.get("boot_id"), str) and bool(expected["boot_id"])
                and observed.get("start_ticks") == expected["start_ticks"]
                and observed.get("boot_id") == expected["boot_id"])
    return observed.get("create_time") == expected.get("create_time")


def _inspect_vllm_process(pid, port):
    command = ["/opt/mas-vllm/venv/bin/python", "-c", _INSPECT_VLLM]
    if os.name == "nt":
        command = ["wsl.exe", "--distribution", "MAS-vLLM", "--user", "root", "--exec", *command]
    try:
        result = subprocess.run(command, input=json.dumps({"pid": pid, "port": port}),
            capture_output=True, text=True, encoding="utf-8", timeout=5, check=True,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        observation = json.loads(result.stdout)
        if not isinstance(observation, dict):
            raise ValueError("Invalid process observation")
        return observation
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("Cannot verify live vLLM process identity") from None


def _validated_profile(profile, gateway):
    """Only the Task2-qualified checkpoint/profile can confer provenance."""
    profile = validate_serving_profile(profile)
    endpoint = urlsplit(gateway.base_url)
    if (gateway.model != MODEL or endpoint.scheme != "http"
        or endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.path != "/v1"
        or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
        or type(profile.get("port")) is not int or not 1 <= profile["port"] <= 65535
        or (endpoint.port or 80) != profile["port"]):
        raise ValueError("vLLM runtime endpoint/model identity mismatch")
    return profile


def _validated_argv(argv, profile):
    expected = ["/opt/mas-vllm/venv/bin/python", "/opt/mas-vllm/venv/bin/vllm", "serve", profile["model"]]
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
        expected.append(flag)
        if value is not None:
            expected.append(str(value))
    if profile.get("structured_outputs_config") is not None:
        expected.extend(["--structured-outputs-config", json.dumps(profile["structured_outputs_config"], separators=(",", ":"))])
    if argv != expected:
        raise ValueError("Effective vLLM argv differs from the pinned runtime profile")


def verify_vllm_runtime(gateway):
    """Revalidate a launcher's ready manifest against the live D-backed WSL runtime.

    A Task2 measurement manifest (stopped_after_probe) deliberately cannot pass.
    The future lifecycle launcher writes profile/versions plus process identity.
    """
    try:
        if not gateway.runtime_manifest:
            raise ValueError("LLM_RUNTIME_MANIFEST is required for vLLM")
        manifest = json.loads(Path(gateway.runtime_manifest).read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("ready") is not True or manifest.get("runtime_status") != "ready"
            or manifest.get("distro") != "MAS-vLLM"):
            raise ValueError("vLLM runtime manifest is not ready")
        profile = _validated_profile(manifest["profile"], gateway)
        qualification = None
        if profile["max_model_len"] == 8192 or manifest.get("qualification"):
            import hashlib
            root = Path(__file__).resolve().parents[3]
            fingerprint = profile_fingerprint(profile)
            proof_path = root / "tmp/vllm/qualifications" / f"{fingerprint}.json"
            proof_bytes = proof_path.read_bytes()
            reference = manifest.get("qualification") or {}
            if (reference.get("profile_fingerprint") != fingerprint
                or reference.get("sha256") != hashlib.sha256(proof_bytes).hexdigest()):
                raise ValueError("Runtime qualification reference mismatch")
            qualification = validate_runtime_qualification(
                profile, json.loads(proof_bytes),
                hashlib.sha256((root / "requirements-vllm.lock").read_bytes()).hexdigest())
        identity = manifest["process"]
        if (not isinstance(identity, dict) or type(identity.get("pid")) is not int
            or identity["pid"] <= 1 or identity.get("process_group") != identity["pid"]
            or isinstance(identity.get("create_time"), bool)
            or not isinstance(identity.get("create_time"), (int, float))
            or not math.isfinite(identity["create_time"]) or identity["create_time"] <= 0):
            raise ValueError("Invalid vLLM runtime process identity")
        _validated_argv(identity.get("effective_argv"), profile)
        observed = _inspect_vllm_process(identity["pid"], profile["port"])
        if not same_process_birth(identity, observed):
            raise ValueError("Live vLLM process birth identity changed")
        for key in ("pid", "process_group", "effective_argv"):
            if observed.get(key) != identity[key]:
                raise ValueError("Live vLLM process identity changed")
        listeners = observed["listeners"]
        if (not isinstance(listeners, list) or not listeners
            or any(type(item.get("pid")) is not int or item["pid"] <= 1
                   or item.get("process_group") != identity["process_group"]
                   or ((item.get("boot_id") != identity["boot_id"]
                        or type(item.get("start_ticks")) is not int
                        or item["start_ticks"] < identity["start_ticks"])
                       if "start_ticks" in identity else item.get("create_time", 0) < identity["create_time"])
                   or item.get("address", "127.0.0.1") not in ("127.0.0.1", "::1")
                   for item in listeners)):
            raise ValueError("vLLM port is not owned by the verified process group")
        versions = manifest["versions"]
        if qualification and (versions != qualification["versions"]
                              or manifest.get("torch_cuda") != qualification["torch_cuda"]):
            raise ValueError("Runtime packages differ from qualified profile")
        if (not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) or not versions[k]
              for k in ("vllm", "torch", "transformers", "triton"))
            or versions != observed.get("versions") or not manifest.get("torch_cuda")
            or manifest["torch_cuda"] != observed.get("torch_cuda")):
            raise ValueError("vLLM runtime package identity differs from manifest")
        return {"engine": "vllm", "revision": profile["revision"],
                "tokenizer_revision": profile["tokenizer_revision"], "dtype": profile["dtype"],
                "context_tokens": profile["max_model_len"], "chat_template_kwargs": profile["chat_template_kwargs"],
                "versions": versions, "torch_cuda": manifest["torch_cuda"],
                "structured_outputs_config": profile.get("structured_outputs_config"),
                "process": {k: observed[k] for k in ("pid", "process_group", "create_time", "start_ticks", "boot_id") if k in observed},
                "effective_argv": observed["effective_argv"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise RuntimeError("Invalid or stale vLLM runtime manifest/process identity") from None


def wait_for_server(gateway, timeout=180, interval=1):
    deadline = time.monotonic() + timeout
    headers = {"Authorization": "Bearer " + gateway.key} if gateway.key else {}
    with httpx.Client(trust_env=False, headers=headers) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(gateway.base_url + "/models", timeout=min(5, max(0.001, deadline - time.monotonic())))
                if response.status_code in (401, 403):
                    raise RuntimeError("Local server rejected credentials; check LLM_API_KEY")
                if response.status_code == 200:
                    data = response.json()
                    if gateway.model not in {item["id"] for item in data["data"]}:
                        raise RuntimeError(f"Local endpoint does not serve model {gateway.model}")
                    verify_vllm_runtime(gateway)
                    return data
            except (httpx.TransportError, ValueError, KeyError, TypeError):
                pass
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    raise TimeoutError("vLLM server is not ready; start scripts/start_vllm.ps1 and inspect its log")
