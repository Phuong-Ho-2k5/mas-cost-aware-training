import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    file = ROOT / "scripts" / filename
    assert file.exists(), "Owned simulation process container is missing"
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name != "nt", reason="Windows process Job Object")
def test_job_cleans_mock_worker_after_launcher_exits(tmp_path):
    module = load("sim_process", "vllm_simulation_process.py")
    output = tmp_path / "child.txt"
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(p.pid,flush=True)"
    owned = module.OwnedSimulation()
    try:
        with output.open("wb") as stdout:
            parent = owned.start([sys.executable, "-c", code], stdout=stdout, stderr=subprocess.DEVNULL, cwd=ROOT)
        parent.wait(5)
        child_pid = int(output.read_text().strip())
        child = psutil.Process(child_pid)
        assert child.is_running()
        assert owned.stop()["active_processes"] == 0
        assert child.wait(timeout=3) == 1
    finally:
        owned.stop()


@pytest.mark.skipif(os.name != "nt", reason="Windows process Job Object")
def test_job_cleanup_reports_unconfirmed_survivors(monkeypatch):
    module = load("sim_process_survivor", "vllm_simulation_process.py")
    owned = module.OwnedSimulation()
    monkeypatch.setattr(owned, "active_processes", lambda: 1)
    with pytest.raises(RuntimeError, match="surviv|active|exit"):
        owned.stop(timeout=0.01)


def test_simulation_report_cannot_pass_without_events_or_clean_shutdown():
    module = load("sim_report", "check_vllm_simulation.py")
    assert module.finalize_report({"passed": True}, [], {"active_processes": 0})["passed"] is False
    assert module.finalize_report({"passed": True}, [{}], {"active_processes": 2})["passed"] is False
    assert module.finalize_report({"passed": True}, [{}], {"active_processes": 0})["passed"] is True


def test_job_setup_failure_still_saves_failed_report(tmp_path, monkeypatch):
    module = load("sim_setup_failure", "check_vllm_simulation.py")
    output = tmp_path / "failed-simulation"
    monkeypatch.setattr(sys, "argv", ["check_vllm_simulation.py", "--output", str(output)])
    def unavailable():
        raise RuntimeError("Job setup unavailable")
    monkeypatch.setattr(module, "OwnedSimulation", unavailable)
    with pytest.raises(RuntimeError, match="Job setup unavailable"):
        module.main()
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False and report["error"] == "Job setup unavailable"
