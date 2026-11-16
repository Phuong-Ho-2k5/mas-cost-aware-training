"""Real POSIX worker cleanup; runnable with stdlib unittest inside MAS-vLLM."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(os.name == "posix", "POSIX process groups")
class ProbeCleanupTest(unittest.TestCase):
    def test_cleanup_includes_workers_after_leader_exit(self):
        path = Path(__file__).resolve().parents[1] / "scripts/check_vllm_qwen.py"
        spec = importlib.util.spec_from_file_location("probe_cleanup", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "cleanup_owned_group"))
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True)"
        parent = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(parent.stdout.readline())
            parent.wait(3)
            self.assertNotIn(child_pid, module.cleanup_owned_group(parent, timeout=1)["remaining_pids"])
            import psutil
            if psutil.pid_exists(child_pid):
                self.assertEqual(psutil.Process(child_pid).status(), psutil.STATUS_ZOMBIE)
        finally:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            parent.stdout.close()


if __name__ == "__main__":
    unittest.main()
