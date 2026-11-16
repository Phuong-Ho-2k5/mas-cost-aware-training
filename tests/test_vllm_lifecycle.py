"""Lifecycle ownership gates without loading a model or signaling real processes."""
import importlib.util
import json
from pathlib import Path
import socket

import pytest

ROOT = Path(__file__).resolve().parents[1]


def service():
    path = ROOT / "scripts/vllm_service.py"
    assert path.exists(), "Task4 lifecycle controller is missing"
    spec = importlib.util.spec_from_file_location("vllm_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    return {"pid": 123, "process_group": 123, "create_time": 1000.25,
            "effective_argv": ["vllm", "serve", "pinned-model"]}


def test_windows_port_conflict_refuses_start_without_touching_listener():
    module = service()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="port|Port"):
            module.assert_port_clear(listener.getsockname()[1])
        assert listener.fileno() != -1


def test_wsl_port_conflict_rejected_before_spawn():
    module = service()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError):
            module.assert_port_clear(listener.getsockname()[1])


@pytest.mark.parametrize("fault", ["reused_pid", "wrong_command", "foreign_port", "foreign_group", "wrong_manifest"])
def test_stop_rejects_stale_or_foreign_identity(fault):
    module = service()
    owned, current = identity(), identity()
    members = [{"pid": 123, "process_group": 123, "session": 123, "create_time": 1000.25}]
    listeners = [{"pid": 123, "process_group": 123}]
    expected = dict(owned)
    if fault == "reused_pid":
        current["create_time"] += 1
    elif fault == "wrong_command":
        current["effective_argv"] = ["unrelated-server"]
    elif fault == "foreign_port":
        listeners[0]["process_group"] = 999
    elif fault == "foreign_group":
        members[0]["session"] = 999
    else:
        expected["pid"] = 999
    with pytest.raises(RuntimeError):
        module.authorize_stop(owned, current, members, listeners, expected=expected)


def test_stop_authorizes_only_owned_group_and_retains_orphan_workers():
    module = service()
    owned = identity()
    worker = {"pid": 124, "process_group": 123, "session": 123, "create_time": 1001}
    assert module.authorize_stop(owned, None, [worker], []) == 123
    assert module.authorize_stop(owned, None, [], []) is None


def test_failed_start_cleanup_uses_only_new_run_id():
    module = service()
    calls = []
    module.cleanup_failed_start("a" * 32, lambda action, payload: calls.append((action, payload)))
    assert calls == [("stop", {"run_id": "a" * 32, "cleanup": True})]


@pytest.mark.parametrize("bad", ["../Ubuntu", "Ubuntu", "docker-desktop", "a;shutdown", "a" * 33])
def test_run_id_cannot_select_other_distro_or_escape_directory(bad):
    with pytest.raises(ValueError):
        service().run_directory(bad)


def test_locked_profile_allows_only_candidate_and_final_ports():
    module = service()
    profile = json.loads((ROOT / "tmp/vllm/launch-profile.json").read_text(encoding="utf-8"))
    assert module.validate_profile(profile, 8001)["port"] == 8001
    assert module.validate_profile(profile, 8000)["port"] == 8000
    with pytest.raises(ValueError):
        module.validate_profile(profile, 9000)
    profile["revision"] = "unverified"
    with pytest.raises(ValueError):
        module.validate_profile(profile, 8001)


def test_package_or_lock_change_cannot_reuse_task2_qualification(tmp_path, monkeypatch):
    import hashlib
    module = service()
    lock = tmp_path / "requirements-vllm.lock"
    lock.write_text("pinned wheel hashes")
    evidence = {"hardware_gate_passed": True, "versions": {"vllm": "0.33.0"},
                "requirements_lock": {"sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}}
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CURRENT", tmp_path / "runtime-manifest.json")
    module.CURRENT.write_text(json.dumps(evidence))
    assert module.qualified_runtime()["versions"] == evidence["versions"]
    lock.write_text("changed")
    with pytest.raises(RuntimeError, match="lock|qualification"):
        module.qualified_runtime()


def test_startup_failure_cleans_new_group_and_writes_failed_state(tmp_path, monkeypatch):
    module = service()
    profile = json.loads((ROOT / "tmp/vllm/launch-profile.json").read_text(encoding="utf-8"))
    (tmp_path / "launch-profile.json").write_text(json.dumps(profile))
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "CURRENT", tmp_path / "runtime-manifest.json")
    monkeypatch.setattr(module, "assert_storage", lambda: None)
    monkeypatch.setattr(module, "assert_port_clear", lambda port: None)
    monkeypatch.setattr(module, "assert_data_path", lambda path: path)
    monkeypatch.setattr(module, "qualified_runtime", lambda: {"versions": {}}, raising=False)
    calls = []
    def call(action, payload):
        calls.append((action, payload))
        if action == "start":
            raise RuntimeError("startup failed")
        return {"remaining_pids": []}
    monkeypatch.setattr(module, "call_linux", call)
    with pytest.raises(RuntimeError, match="startup failed"):
        module.start(8001, 1)
    status = json.loads(module.CURRENT.read_text())
    assert not status["ready"] and status["runtime_status"] == "failed"
    assert calls[-1] == ("stop", {"run_id": status["run_id"], "cleanup": True})


def test_kernel_birth_identity_survives_clock_shift_but_not_pid_reuse():
    module = service()
    owned = {**identity(), "start_ticks": 123456, "boot_id": "boot-A"}
    shifted = {**owned, "create_time": owned["create_time"] + 2}
    members = [{"pid": 123, "process_group": 123, "session": 123, "create_time": 1002.25,
                "start_ticks": 123456, "boot_id": "boot-A"}]
    assert module.authorize_stop(owned, shifted, members, []) == 123
    for altered in ({**shifted, "start_ticks": 999999}, {**shifted, "boot_id": "boot-B"}):
        with pytest.raises(RuntimeError):
            module.authorize_stop(owned, altered, members, [])


def test_dead_service_stop_refuses_a_new_windows_port_owner(tmp_path, monkeypatch):
    module = service()
    run_id = "b" * 32
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "CURRENT", tmp_path / "runtime-manifest.json")
    monkeypatch.setattr(module, "assert_data_path", lambda p: p)
    (tmp_path / "runs" / run_id).mkdir(parents=True)
    calls = []
    def call(action, payload):
        calls.append(action)
        return {"alive": False} if action == "status" else {"remaining_pids": []}
    monkeypatch.setattr(module, "call_linux", call)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0)); server.listen()
        port = server.getsockname()[1]
        module.CURRENT.write_text(json.dumps({"task": 4, "ready": True, "runtime_status": "ready",
            "distro": "MAS-vLLM", "run_id": run_id, "process": identity(), "profile": {"port": port}}))
        with pytest.raises(RuntimeError, match="Port|port"):
            module.stop(port)
        assert "stop" not in calls


@pytest.mark.parametrize("state", ["ready", "stop_failed", "starting"])
def test_stop_unhealthy_owned_server_without_http(tmp_path, monkeypatch, state):
    module = service()
    run_id = "c" * 32
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "CURRENT", tmp_path / "runtime-manifest.json")
    monkeypatch.setattr(module, "assert_data_path", lambda p: p)
    (tmp_path / "runs" / run_id).mkdir(parents=True)
    windows = [{"pid": 234, "create_time": 500, "exe": "wslrelay.exe", "address": "127.0.0.1"}]
    monkeypatch.setattr(module, "windows_listeners", lambda port: windows, raising=False)
    def unhealthy(*args):
        raise RuntimeError("health is hung")
    monkeypatch.setattr(module, "endpoint_challenge", unhealthy)
    monkeypatch.setattr(module, "assert_port_clear", lambda port: None if "stop" in calls else (_ for _ in ()).throw(RuntimeError("owned occupied port")))
    calls = []
    def call(action, payload):
        calls.append(action)
        return {"alive": True} if action == "status" else {"remaining_pids": []}
    monkeypatch.setattr(module, "call_linux", call)
    module.CURRENT.write_text(json.dumps({"task": 4, "ready": state=="ready", "runtime_status": state,
        "distro": "MAS-vLLM", "run_id": run_id, "process": identity(), "profile": {"port": 8001},
        "windows_listeners": windows}))
    result = module.stop(8001)
    assert result["runtime_status"] == "stopped" and calls == ["stop"]


def test_stop_rejects_changed_windows_listener_identity(tmp_path, monkeypatch):
    module = service()
    monkeypatch.setattr(module, "CURRENT", tmp_path / "runtime-manifest.json")
    windows = [{"pid": 234, "create_time": 500, "exe": "wslrelay.exe", "address": "127.0.0.1"}]
    monkeypatch.setattr(module, "windows_listeners", lambda port: [{**windows[0], "create_time": 501}], raising=False)
    monkeypatch.setattr(module, "call_linux", lambda *args: pytest.fail("Must not signal Linux on foreign Windows port"))
    module.CURRENT.write_text(json.dumps({"task": 4, "ready": True, "runtime_status": "ready",
        "distro": "MAS-vLLM", "run_id": "d"*32, "profile": {"port": 8001}, "windows_listeners": windows}))
    with pytest.raises(RuntimeError):
        module.stop(8001)


def test_stop_waits_for_owned_windows_relay_to_release_port(monkeypatch):
    module = service()
    checks = []
    monkeypatch.setattr(module, "assert_windows_stop_owner", lambda m, p: None)
    def port_clear(port):
        checks.append(port)
        if len(checks) == 1:
            raise RuntimeError("relay still draining")
    monkeypatch.setattr(module, "assert_port_clear", port_clear)
    module.wait_for_port_release({}, 8001, timeout=1)
    assert checks == [8001, 8001]


def test_compact_decoder_configuration_is_verified_and_legacy_profile_can_stop():
    from mas_demo.llm.health import _validated_argv
    module = service()
    profile = json.loads((ROOT / "tmp/vllm/launch-profile.json").read_text(encoding="utf-8"))
    profile["structured_outputs_config"] = {"backend": "xgrammar", "disable_any_whitespace": True}
    module.validate_profile(profile, 8001)
    # Existing Task2 process argv remains valid for its original stop/provenance path.
    old = json.loads((ROOT / "tmp/vllm/task2-qualification.json").read_text(encoding="utf-8"))
    args = next(iter(old["verified_launches"].values()))["identity"]["effective_argv"]
    _validated_argv(args, old["profile"])
    with pytest.raises(ValueError):
        _validated_argv(args, profile)
    _validated_argv(args + ["--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":true}'], profile)
