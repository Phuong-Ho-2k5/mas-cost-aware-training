"""Windows lifecycle controller and fixed WSL worker for the D-backed vLLM service."""
import argparse
import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mas_demo.llm.health import (_inspect_vllm_process, _validated_argv, _validated_profile,
                                 verify_vllm_runtime, kernel_birth, same_process_birth)
from mas_demo.llm.runtime_profile import profile_fingerprint, validate_runtime_qualification

DATA = ROOT / "tmp/vllm"
CURRENT = DATA / "runtime-manifest.json"
LINUX_RUNS = Path("/opt/mas-vllm/runs")


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_directory(run_id):
    if not isinstance(run_id, str) or not re.fullmatch("[0-9a-f]{32}", run_id):
        raise ValueError("Invalid task-owned run ID")
    return LINUX_RUNS / run_id


def validate_profile(profile, port):
    if type(port) is not int or port not in (8000, 8001):
        raise ValueError("Only candidate port8001 or final port8000 is allowed")
    result = deepcopy(profile)
    result["port"] = port
    _validated_profile(result, SimpleNamespace(model="Qwen/Qwen3.5-0.8B", base_url=f"http://127.0.0.1:{port}/v1"))
    return result


def assert_port_clear(port):
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            if family == socket.AF_INET6:
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                probe.bind((address, port))
            except OSError:
                raise RuntimeError(f"Port {port} is occupied; refusing to touch its owner") from None


def authorize_stop(owned, current, members, listeners, expected=None):
    """Pure ownership gate; no signals until every identity check succeeds."""
    if expected is not None and expected != owned:
        raise RuntimeError("Manifest identity differs from the task-owned receipt")
    if current is not None and (not same_process_birth(owned, current)
        or any(current.get(k) != owned.get(k) for k in ("pid", "process_group", "effective_argv"))):
        raise RuntimeError("Refusing stale/reused PID or changed command")
    group = owned["process_group"]
    if group != owned["pid"] or group <= 1:
        raise RuntimeError("Invalid task-owned group")
    if any(p["process_group"] != group or p["session"] != group
           or ((p.get("boot_id") != owned["boot_id"] or p.get("start_ticks", -1) < owned["start_ticks"])
               if "start_ticks" in owned else p["create_time"] < owned["create_time"]) for p in members):
        raise RuntimeError("Group membership identity mismatch")
    if any(p["process_group"] != group for p in listeners):
        raise RuntimeError("Port belongs to another process group")
    return group if members else None


def cleanup_failed_start(run_id, call_linux):
    return call_linux("stop", {"run_id": run_id, "cleanup": True})


def process_identity(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return None
        return {"pid": pid, "process_group": os.getpgid(pid),
                "create_time": p.create_time(), "effective_argv": p.cmdline(), **kernel_birth(pid)}
    except psutil.NoSuchProcess:
        return None


def group_members(group):
    import psutil
    result = []
    for p in psutil.process_iter(["pid", "create_time", "status"]):
        try:
            if p.info["status"] != psutil.STATUS_ZOMBIE and os.getpgid(p.pid) == group:
                result.append({"pid": p.pid, "process_group": group,
                               "session": os.getsid(p.pid), "create_time": p.info["create_time"], **kernel_birth(p.pid)})
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return result


def listeners(port):
    import psutil
    return [{"pid": c.pid, "process_group": os.getpgid(c.pid)}
            for c in psutil.net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN and c.laddr.port == port and c.pid is not None]


def linux_action(action, payload):
    directory = run_directory(payload["run_id"])
    receipt_path = directory / "owner.json"
    if action == "start":
        profile = validate_profile(payload["profile"], payload["profile"]["port"])
        assert_port_clear(profile["port"])
        directory.mkdir(parents=True, exist_ok=False)
        command = ["/bin/bash", str(ROOT / "scripts/wsl/start_vllm.sh"), str(profile["port"]), "1024", "0.65",
                   "true" if profile.get("structured_outputs_config") else "false", str(profile["max_model_len"]),
                   profile.get("reasoning_parser") or "none"]
        with (directory / "server.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        try:
            import psutil
            identity = {"pid": child.pid, "process_group": child.pid,
                        "create_time": psutil.Process(child.pid).create_time(), "effective_argv": command, **kernel_birth(child.pid)}
            receipt = {"run_id": payload["run_id"], "profile": profile, "process": identity,
                       "bootstrap_argv": command, "runtime_status": "starting", "server_log": str(directory / "server.log")}
            save(receipt_path, receipt)
            return receipt
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    if not receipt_path.exists():
        if action == "stop" and payload.get("cleanup"):
            return {"remaining_pids": [], "note": "No server receipt was created"}
        raise RuntimeError("No task-owned launch receipt")
    receipt = json.loads(receipt_path.read_text())
    profile = validate_profile(receipt["profile"], receipt["profile"]["port"])
    owned = receipt["process"]
    current = process_identity(owned["pid"])
    if action == "status":
        return {"alive": current is not None and same_process_birth(owned, current),
                "process": current}
    if action == "observe":
        if current is None or not same_process_birth(owned, current) or any(current[k] != owned[k] for k in ("pid", "process_group")):
            raise RuntimeError("Owned vLLM process exited or changed identity")
        _validated_argv(current["effective_argv"], profile)
        observed = _inspect_vllm_process(owned["pid"], profile["port"])
        receipt["process"] = current
        receipt["members"] = group_members(owned["pid"])
        save(receipt_path, receipt)
        return {**receipt, "versions": observed["versions"], "torch_cuda": observed["torch_cuda"]}
    if action == "challenge":
        nonce = payload["nonce"]
        if not isinstance(nonce, str) or not re.fullmatch("[0-9a-f]{32}", nonce):
            raise ValueError("Invalid endpoint nonce")
        expected = f'"GET /health?mas_identity={nonce} HTTP/1.1" 200'
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if expected in (directory / "server.log").read_text(errors="replace"):
                return {"nonce": nonce, "observed_in_owned_log": True}
            time.sleep(0.1)
        raise RuntimeError("Windows request did not reach the owned vLLM listener")
    if action != "stop":
        raise ValueError("Unknown Linux lifecycle action")
    # Verify both the selected receipt and any caller's manifest before allowing exec's argv transition.
    if payload.get("expected") is not None and payload["expected"] != owned:
        raise RuntimeError("Manifest identity differs from the task-owned receipt")
    if current is not None and current["effective_argv"] != owned["effective_argv"]:
        _validated_argv(current["effective_argv"], profile)
        owned = {**owned, "effective_argv": current["effective_argv"]}
    members = group_members(owned["process_group"])
    port_owners = [] if payload.get("cleanup") else listeners(profile["port"])
    group = authorize_stop(owned, current, members, port_owners)
    escalated = False
    if group is not None:
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while group_members(group) and time.monotonic() < deadline:
            time.sleep(0.2)
        if group_members(group):
            escalated = True
            os.killpg(group, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while group_members(group) and time.monotonic() < deadline:
                time.sleep(0.1)
    remaining = group_members(owned["process_group"])
    receipt.update(runtime_status="stopped" if not remaining else "stop_failed",
                   cleanup={"remaining_pids": [p["pid"] for p in remaining], "escalated": escalated})
    save(receipt_path, receipt)
    if remaining:
        raise RuntimeError("Owned workers did not exit")
    return receipt["cleanup"]


def assert_data_path(path):
    path = Path(path).absolute()
    if path.drive.upper() != "D:" or path == Path("D:/"):
        raise ValueError("Lifecycle data must stay on D")
    for item in (path, *path.parents):
        if item.exists() and getattr(item.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400:
            raise ValueError("Lifecycle data cannot traverse a reparse point")
    return path


def assert_storage():
    if shutil.disk_usage("C:/").free < 4 * 1024**3 or shutil.disk_usage("D:/").free < 10 * 1024**3:
        raise RuntimeError("Host storage reserve breached: need C4GiB/D10GiB")


def call_linux(action, payload):
    script = "/mnt/d/" + str(Path(__file__).resolve())[3:].replace("\\", "/")
    command = ["wsl.exe", "--distribution", "MAS-vLLM", "--user", "root", "--exec",
               "/opt/mas-vllm/venv/bin/python", script, "--linux-action", action]
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True,
        text=True, encoding="utf-8", timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("WSL lifecycle operation failed: " + result.stderr.strip()[-1500:])
    return json.loads(result.stdout)


def endpoint_challenge(client, port, run_id):
    nonce = uuid.uuid4().hex
    response = client.get(f"http://127.0.0.1:{port}/health", params={"mas_identity": nonce}, timeout=5)
    response.raise_for_status()
    return call_linux("challenge", {"run_id": run_id, "nonce": nonce})


def windows_listeners(port):
    import psutil
    result = []
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port:
            if connection.pid is None:
                raise RuntimeError("Cannot identify Windows port owner")
            owner = psutil.Process(connection.pid)
            result.append({"pid": owner.pid, "create_time": owner.create_time(),
                           "exe": owner.exe(), "address": connection.laddr.ip})
    return sorted(result, key=lambda item: (item["pid"], item["address"]))


def remember_windows_relay(manifest, port):
    owners = windows_listeners(port)
    approved = os.path.normcase(str(Path(os.environ["ProgramFiles"]) / "WSL/wslrelay.exe"))
    if owners and any(os.path.normcase(item["exe"]) != approved for item in owners):
        raise RuntimeError("Windows port belongs to an unrelated application")
    if owners:
        manifest["windows_listeners"] = owners


def assert_windows_stop_owner(manifest, port):
    owners = windows_listeners(port)
    if owners and owners != manifest.get("windows_listeners"):
        raise RuntimeError("Windows port owner changed or has no task-owned proof; refusing stop")


def wait_for_port_release(manifest, port, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        assert_windows_stop_owner(manifest, port)
        try:
            assert_port_clear(port)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Owned workers exited but Windows relay has not released the port") from None
            time.sleep(0.1)


def qualified_runtime(profile):
    lock_hash = hashlib.sha256((ROOT / "requirements-vllm.lock").read_bytes()).hexdigest()
    fingerprint = profile_fingerprint(profile)
    proof_path = DATA / "qualifications" / f"{fingerprint}.json"
    if proof_path.exists():
        proof_bytes = proof_path.read_bytes()
        evidence = validate_runtime_qualification(profile, json.loads(proof_bytes), lock_hash)
        return {**evidence, "qualification_reference": {
            "profile_fingerprint": fingerprint, "sha256": hashlib.sha256(proof_bytes).hexdigest()}}
    if profile["max_model_len"] != 4096:
        raise RuntimeError("Profile-specific qualification is missing; run the context probe first")
    evidence_file = DATA / "task2-qualification.json"
    source = evidence_file if evidence_file.exists() else CURRENT
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if (evidence.get("hardware_gate_passed") is not True
        or lock_hash != evidence.get("requirements_lock", {}).get("sha256")):
        raise RuntimeError("Task2 qualification/lock is missing or changed")
    if profile_fingerprint(evidence.get("profile", {})) != profile_fingerprint(profile):
        raise RuntimeError("Legacy qualification profile mismatch")
    if profile.get("structured_outputs_config") is not None:
        raise RuntimeError("New compact starts require profile-specific probe qualification")
    if not evidence_file.exists():
        save(evidence_file, evidence)
    return evidence


def start(port, timeout):
    import httpx
    assert_storage()
    assert_port_clear(port)
    profile = validate_profile(json.loads((DATA / "launch-profile.json").read_text(encoding="utf-8")), port)
    profile["structured_outputs_config"] = {"backend": "xgrammar", "disable_any_whitespace": True}
    qualification = qualified_runtime(profile)
    if CURRENT.exists():
        previous = json.loads(CURRENT.read_text(encoding="utf-8"))
        if previous.get("ready") is True or previous.get("runtime_status") in ("starting", "stop_failed"):
            raise RuntimeError("An active/unfinished manifest exists; use stop or inspect it first")
    run_id = uuid.uuid4().hex
    directory = assert_data_path(DATA / "runs" / run_id)
    directory.mkdir(parents=True)
    if CURRENT.exists():
        shutil.copyfile(CURRENT, directory / "previous-runtime-manifest.json")
    manifest = {"schema_version": 1, "task": 4, "distro": "MAS-vLLM", "run_id": run_id,
                "runtime_status": "starting", "ready": False, "profile": profile}
    if qualification.get("qualification_reference"):
        manifest["qualification"] = qualification["qualification_reference"]
    save(CURRENT, manifest)
    started = time.monotonic()
    try:
        call_linux("start", {"run_id": run_id, "profile": profile})
        deadline = started + timeout
        last_status = 0
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() < deadline:
                assert_storage()
                if time.monotonic() - last_status >= 5:
                    if not call_linux("status", {"run_id": run_id})["alive"]:
                        raise RuntimeError("Owned vLLM exited during startup; inspect the run server.log")
                    remember_windows_relay(manifest, port)
                    save(CURRENT, manifest)
                    last_status = time.monotonic()
                try:
                    health = client.get(f"http://127.0.0.1:{port}/health", timeout=1)
                    models = client.get(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                    if health.status_code == 200 and models.status_code == 200:
                        if profile["model"] not in {m["id"] for m in models.json()["data"]}:
                            raise RuntimeError("Windows endpoint model mismatch")
                        observed = call_linux("observe", {"run_id": run_id})
                        if (observed["versions"] != qualification["versions"]
                            or observed["torch_cuda"] != qualification["torch_cuda"]):
                            raise RuntimeError("Live packages differ from the Task2-qualified runtime")
                        remember_windows_relay(manifest, port)
                        manifest.update(process=observed["process"], versions=observed["versions"],
                                        torch_cuda=observed["torch_cuda"], server_log=observed["server_log"],
                                        windows_endpoint_proof=endpoint_challenge(client, port, run_id),
                                        ready=True, runtime_status="ready", startup_seconds=time.monotonic() - started)
                        candidate = directory / "runtime-manifest.json"
                        save(candidate, manifest)
                        verify_vllm_runtime(SimpleNamespace(model=profile["model"],
                            base_url=f"http://127.0.0.1:{port}/v1", runtime_manifest=str(candidate)))
                        save(CURRENT, manifest)
                        return manifest
                except httpx.TransportError:
                    pass
                time.sleep(1)
        raise TimeoutError("vLLM startup deadline exceeded")
    except BaseException as error:
        manifest.update(ready=False, runtime_status="failed", error=str(error))
        try:
            remember_windows_relay(manifest, port)
        except Exception as owner_error:
            manifest["windows_owner_error"] = str(owner_error)
        try:
            manifest["cleanup"] = cleanup_failed_start(run_id, call_linux)
        except Exception as cleanup_error:
            manifest.update(runtime_status="stop_failed", cleanup_error=str(cleanup_error))
        save(CURRENT, manifest)
        save(directory / "runtime-manifest.json", manifest)
        raise


def stop(port):
    manifest = json.loads(CURRENT.read_text(encoding="utf-8"))
    if (manifest.get("task") != 4 or manifest.get("distro") != "MAS-vLLM"
        or manifest.get("profile", {}).get("port") != port):
        raise RuntimeError("No matching task-owned runtime manifest; refusing stop")
    if manifest["runtime_status"] == "stopped":
        assert_port_clear(port)
        return manifest
    # Health is not required to recover a hung server. The saved Windows relay
    # birth identity and the Linux receipt/group checks independently authorize stop.
    assert_windows_stop_owner(manifest, port)
    cleanup = call_linux("stop", {"run_id": manifest["run_id"], "expected": manifest.get("process")})
    manifest.update(ready=False, runtime_status="stopped", cleanup=cleanup)
    save(CURRENT, manifest)
    try:
        wait_for_port_release(manifest, port)
    except RuntimeError as error:
        manifest.update(runtime_status="stop_failed", error=str(error))
        save(CURRENT, manifest)
        raise
    save(assert_data_path(DATA / "runs" / manifest["run_id"] / "runtime-manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("start", "stop"))
    parser.add_argument("--port", type=int, choices=(8000, 8001), default=8000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--linux-action", choices=("start", "status", "observe", "challenge", "stop"))
    args = parser.parse_args()
    if args.linux_action:
        if os.name != "posix":
            raise RuntimeError("Linux worker requires MAS-vLLM")
        result = linux_action(args.linux_action, json.load(sys.stdin))
    else:
        if os.name != "nt" or not args.action or not 1 <= args.timeout <= 600:
            raise ValueError("Windows controller needs an action and timeout1..600s")
        assert_data_path(DATA)
        for name in (CURRENT, DATA / "lifecycle.lock", DATA / "runs", DATA / "launch-profile.json", DATA / "task2-qualification.json"):
            assert_data_path(name)
        from filelock import FileLock
        with FileLock(str(DATA / "lifecycle.lock"), timeout=0):
            result = start(args.port, args.timeout) if args.action == "start" else stop(args.port)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
