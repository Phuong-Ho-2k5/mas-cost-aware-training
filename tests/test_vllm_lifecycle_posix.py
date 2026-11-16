"""Real owned-group stop/refusal tests in MAS-vLLM; no GPU/model involved."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
import uuid


@unittest.skipUnless(os.name == "posix", "Run inside MAS-vLLM")
class LifecycleGroups(unittest.TestCase):
    def test_linux_conflicting_listener_is_untouched(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("lifecycle_conflict", root / "scripts/vllm_service.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profile = module.validate_profile(json.loads((root / "tmp/vllm/launch-profile.json").read_text()), 8000)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 8000)); listener.listen()
            with self.assertRaises(RuntimeError):
                module.linux_action("start", {"run_id": uuid.uuid4().hex, "profile": profile})
            self.assertGreaterEqual(listener.fileno(), 0)

    def test_stop_preserves_unrelated_group_and_stale_pid_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("lifecycle", root / "scripts/vllm_service.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir="/opt/mas-vllm/tmp") as directory:
            module.LINUX_RUNS = Path(directory)
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
            owned = subprocess.Popen(command, start_new_session=True)
            other = subprocess.Popen(command, start_new_session=True)
            try:
                run_id = uuid.uuid4().hex
                run = module.run_directory(run_id)
                run.mkdir()
                profile = module.validate_profile(json.loads((root / "tmp/vllm/launch-profile.json").read_text()), 8000)
                original = module.process_identity(owned.pid)
                stale = {**original, "create_time": original["create_time"] - 100, "start_ticks": original["start_ticks"] - 1}
                module.save(run / "owner.json", {"profile": profile, "process": stale})
                with self.assertRaises(RuntimeError):
                    module.linux_action("stop", {"run_id": run_id})
                self.assertIsNone(owned.poll())
                self.assertIsNone(other.poll())
                module.save(run / "owner.json", {"profile": profile, "process": original})
                result = module.linux_action("stop", {"run_id": run_id})
                self.assertEqual(result["remaining_pids"], [])
                owned.wait(3)
                self.assertIsNone(other.poll())
            finally:
                for child in (owned, other):
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(3)


if __name__ == "__main__":
    unittest.main()
