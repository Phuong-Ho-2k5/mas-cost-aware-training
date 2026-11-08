import os
import subprocess
import sys
import time
from pathlib import Path
from mas_demo.contracts import SegmentResult
from mas_demo.io import write_json


class LocalBackend:
    """Synchronous child process; CUDA context is released before the next branch."""

    def run(self, spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor=None, intervention=None):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        request = output_dir / "request.json"
        write_json(request, {"spec": spec, "start": start, "epochs": epochs, "branch_id": branch_id,
                             "output_dir": str(output_dir), "budget_seconds": budget_seconds,
                             "lr_factor": lr_factor, "intervention": intervention})
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        began = time.monotonic()
        with (output_dir / "worker.log").open("w", encoding="utf-8") as log:
            try:
                subprocess.run([sys.executable, "-m", "mas_demo.training.worker", str(request)],
                               stdout=log, stderr=log, env=env, timeout=max(0.01, budget_seconds), check=True)
                result = SegmentResult.model_validate_json((output_dir / "result.json").read_text(encoding="utf-8"))
            except subprocess.TimeoutExpired:
                result = SegmentResult(status="BUDGET_EXHAUSTED", checkpoint=start,
                                       error="Worker watchdog expired; partial work not comparable")
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                result = SegmentResult(status="FAILED", checkpoint=start, error=f"Worker failed: {type(error).__name__}; see worker.log")
        result.seconds = time.monotonic() - began
        write_json(output_dir / "result.json", result)
        return result
