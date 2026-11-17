"""Immutable local experiment captures; no credentials or approximate token counts."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[3]


def fingerprint(paths, root=ROOT):
    files = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(set(map(Path, paths)))}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def new_capture_directory(path):
    path = Path(path).resolve()
    if os.name == "nt" and path.drive.upper() != "D:":
        raise ValueError("Experiment captures must be on D")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class Capture:
    def __init__(self, directory, gateway, fixture=None, settings=None):
        self.directory = new_capture_directory(directory)
        self.gateway = gateway
        fixture = Path(fixture) if fixture else ROOT / "tests/fixtures/config_agent_cases.json"
        sources = list((ROOT / "src/mas_demo").rglob("*.py")) + list((ROOT / "src/mas_demo/prompts").glob("*.txt"))
        sources += [ROOT / "scripts" / name for name in ("benchmark_local_agents.py", "benchmark_config_agent.py", "check_local_agents.py", "compare_local_training.py")]
        self.manifest = {"schema_version": 1, "state": "running", "model": gateway.model,
            "backend": gateway.local_backend, "endpoint": gateway.base_url,
            "gateway": {name: getattr(gateway, name) for name in ("timeout", "deadline", "attempts", "max_tokens")},
            "settings": settings or {}, "code_fingerprint": fingerprint(sources),
            "fixture_fingerprint": fingerprint([fixture], fixture.parent), "python": platform.python_version(),
            "client_packages": {name: metadata.version(name) for name in ("httpx", "pydantic", "torch", "numpy")},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with httpx.Client(trust_env=False) as client:
            response = client.get(gateway.base_url.removesuffix("/v1") + "/health", timeout=5)
            response.raise_for_status()
            self.health = response.json() if response.content else {}
        self.manifest["health"] = self.health
        from mas_demo.llm.health import verify_vllm_runtime
        self.manifest["verified_runtime"] = verify_vllm_runtime(gateway)
        self.runtime = json.loads(Path(gateway.runtime_manifest).read_text(encoding="utf-8"))
        write_json(self.directory / "runtime-manifest.json", self.runtime)
        write_json(self.directory / "manifest.json", self.manifest)
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def append(self, filename, value):
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def sample(self):
        import psutil
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        while not self.stop.is_set():
            row = {"monotonic": time.monotonic()}
            try:
                values = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=5, **flags).strip().splitlines()[0].split(",")
                row["device_vram_used_mib"], row["device_vram_free_mib"] = map(float, values)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row["gpu_unavailable"] = type(error).__name__
            try:
                if self.runtime:
                    code = '''import json,sys,psutil
from pathlib import Path
m=json.load(sys.stdin);p=psutil.Process(m['pid'])
ticks=int(Path(f"/proc/{p.pid}/stat").read_text().rsplit(')',1)[1].split()[19])
assert ticks==m['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==m['boot_id']
print(json.dumps({'rss_mib':sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])/1024**2}))'''
                    result = subprocess.run(["wsl.exe", "-d", "MAS-vLLM", "-u", "root", "--exec", "/opt/mas-vllm/venv/bin/python", "-c", code], input=json.dumps(self.runtime["process"]), capture_output=True, text=True, timeout=5, check=True, **flags)
                    row["server_tree_rss_mib"] = json.loads(result.stdout)["rss_mib"]
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error):
                row["rss_unavailable"] = True
            self.samples.append(row)
            self.append("resources.jsonl", row)
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, kind, value, trace):
        self.stop.set()
        self.thread.join(12)
        resources = {"samples": len(self.samples), "note": "NVML device memory, not CUDA allocator. RSS tree sums may double-count shared pages; Windows and WSL RSS have different accounting."}
        for key in ("device_vram_used_mib", "server_tree_rss_mib"):
            values = [row[key] for row in self.samples if key in row]
            resources["peak_" + key] = max(values) if values else None
        write_json(self.directory / "resource-summary.json", resources)
        self.manifest.update(state="failed" if kind else "completed", resource_profile=resources)
        if kind:
            self.manifest["error_type"] = kind.__name__
        write_json(self.directory / "manifest.json", self.manifest)
