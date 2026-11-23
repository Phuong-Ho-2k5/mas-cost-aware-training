"""Task 2: measure a pinned Qwen/vLLM candidate without changing MAS settings.

Run --launch inside the D-backed MAS-vLLM distro. The owned server is always
stopped on exit. Requests, errors, resource samples and the gate result persist.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator, ValidationError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mas_demo.llm.runtime_profile import (
    PINNED, COMPACT_DECODER, profile_fingerprint, expected_serving_argv, validate_runtime_qualification,
)


MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CHECKS = ("json_requests", "long_context", "context_rejected", "truncation_rejected", "cancellation", "resource_headroom")


def chat_body(model, prompt, max_tokens=128):
    return {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": "Return only JSON matching the schema. No explanation."},
                     {"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "probe", "strict": False,
            "schema": {"type": "object", "properties": {"answer": {"type": "string", "maxLength": 40}},
                       "required": ["answer"], "additionalProperties": False}}},
    }


def token_body(body):
    return {**{key: body[key] for key in ("model", "messages", "response_format", "chat_template_kwargs")},
            "add_generation_prompt": True, "add_special_tokens": False}


def validate_completion(body, response, expected_prompt_tokens):
    if response.get("model") != body["model"]:
        raise ValueError("Served model identity mismatch")
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete finish_reason")
    if response.get("usage", {}).get("prompt_tokens") != expected_prompt_tokens:
        raise ValueError("Prompt token count differs from tokenizer preflight")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    value = json.loads(choice["message"]["content"], parse_constant=reject_constant)
    try:
        Draft202012Validator(body["response_format"]["json_schema"]["schema"]).validate(value)
    except ValidationError as error:
        raise ValueError("Full JSON Schema validation failed") from error
    return value


def scheduler_counts(metrics):
    counts, seen = {"running": 0, "waiting": 0}, set()
    for line in metrics.splitlines():
        match = re.match(r'^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([-+0-9.eE]+)', line)
        if match:
            name, value = match.groups()
            number = float(value)
            if not math.isfinite(number) or number < 0 or number != int(number):
                raise ValueError("Invalid scheduler metrics")
            counts[name] += int(number)
            seen.add(name)
    if seen != {"running", "waiting"}:
        raise ValueError("Scheduler metrics are unavailable")
    return counts


def hardware_gate(checks, provenance_verified=False):
    return provenance_verified and all(checks.get(name) is True for name in CHECKS)


def wait_for_idle(fetch_counts, timeout=5, clock=time.monotonic):
    started = clock()
    while True:
        remaining = timeout - (clock() - started)
        if remaining <= 0:
            raise TimeoutError("Scheduler did not become idle within five seconds")
        counts = fetch_counts(remaining)
        elapsed = clock() - started
        if elapsed > timeout:
            raise TimeoutError("Idle was observed after the cancellation deadline")
        if counts == {"running": 0, "waiting": 0}:
            return elapsed
        time.sleep(min(0.1, max(0, timeout - elapsed)))


def summarize_resources(samples):
    gpu = [s for s in samples if "device_vram_free_mib" in s]
    inference = [s for s in samples if s.get("phase") == "inference"]
    valid = [s for s in inference if "device_vram_free_mib" in s]
    rss = [s["server_tree_rss_mib"] for s in samples if s.get("server_tree_rss_mib") is not None]
    return {"samples": len(samples), "inference_samples": len(inference), "valid_inference_gpu_samples": len(valid),
            "peak_device_vram_used_mib": max((s["device_vram_used_mib"] for s in gpu), default=None),
            "minimum_device_vram_free_mib": min((s["device_vram_free_mib"] for s in gpu), default=None),
            "minimum_inference_vram_free_mib": min((s["device_vram_free_mib"] for s in valid), default=None),
            "peak_server_tree_rss_mib": max(rss, default=None),
            "note": "NVML measures device memory, not the CUDA allocator; WSL reporting can differ. RSS sums server/worker RSS and can count shared pages more than once."}


def resource_headroom(summary):
    return (summary.get("inference_samples", 0) > 0
            and summary.get("valid_inference_gpu_samples") == summary["inference_samples"]
            and summary.get("minimum_inference_vram_free_mib", -1) >= 512)


def host_storage():
    roots = {"c": "C:/", "d": "D:/"} if os.name == "nt" else {"c": "/mnt/c", "d": "/mnt/d"}
    return {drive + "_free_gib": shutil.disk_usage(path).free / 1024**3 for drive, path in roots.items()}


def validate_host_storage(storage):
    for key, minimum in (("c_free_gib", 4), ("d_free_gib", 10)):
        value = storage.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"Host storage reserve failed: {key}={value}, minimum={minimum} GiB")


def owned_group_members(group):
    import psutil
    members = []
    for process in psutil.process_iter(["pid", "status"]):
        try:
            if process.info["status"] != psutil.STATUS_ZOMBIE and os.getpgid(process.pid) == group:
                members.append(process.pid)
        except (ProcessLookupError, PermissionError, psutil.Error):
            pass
    return members


def cleanup_owned_group(process, timeout=15):
    group = process.pid  # Popen(start_new_session=True) established ownership.
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while owned_group_members(group) and time.monotonic() < deadline:
        time.sleep(0.1)
    escalated = bool(owned_group_members(group))
    if escalated:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3
        while owned_group_members(group) and time.monotonic() < deadline:
            time.sleep(0.1)
    process.wait(timeout=1)
    remaining = owned_group_members(group)
    if remaining:
        raise RuntimeError(f"Owned worker group did not exit: {remaining}")
    return {"remaining_pids": remaining, "escalated": escalated}


def finalize_probe(report, resources, process, output, started):
    errors = report.setdefault("cleanup_errors", [])
    if getattr(resources, "storage_error", None):
        report.setdefault("error", {"type": "HostStorageError", "message": resources.storage_error})
    try:
        report["resources"] = resources.finish()
    except Exception as error:
        errors.append(f"resources: {type(error).__name__}: {error}")
    try:
        if process:
            report["cleanup"] = cleanup_owned_group(process)
    except Exception as error:
        errors.append(f"server: {type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["checks"]["resource_headroom"] = resource_headroom(report.get("resources", {}))
        report["hardware_gate_passed"] = (not report.get("error") and not errors
            and hardware_gate(report["checks"], report.get("provenance_verified", False)))
        save(output / "probe.json", report)


def verify_launch_identity(process, batch_tokens, memory_fraction, context_tokens=4096, reasoning_parser=None):
    import psutil
    argv = psutil.Process(process.pid).cmdline()
    if batch_tokens == 1024 and float(memory_fraction) == 0.65:
        declared = {**PINNED, "port": 8001, "max_model_len": context_tokens,
                    "structured_outputs_config": COMPACT_DECODER}
        if reasoning_parser:
            declared["reasoning_parser"] = reasoning_parser
        if argv != expected_serving_argv(declared):
            raise ValueError("Effective launch command differs from the complete declared profile")
    expected = {"--revision": REVISION, "--tokenizer-revision": REVISION, "--served-model-name": MODEL,
                "--host": "127.0.0.1", "--port": "8001", "--dtype": "float16", "--max-model-len": str(context_tokens),
                "--max-num-seqs": "1", "--max-num-batched-tokens": str(batch_tokens), "--gpu-memory-utilization": memory_fraction}
    for flag, value in expected.items():
        if flag not in argv or argv[argv.index(flag) + 1] != value:
            raise ValueError(f"Effective launch identity differs at {flag}")
    kwargs = json.loads(argv[argv.index("--default-chat-template-kwargs") + 1])
    if kwargs.get("enable_thinking") is not False:
        raise ValueError("Effective thinking mode is not disabled")
    if ("--structured-outputs-config" not in argv
        or json.loads(argv[argv.index("--structured-outputs-config") + 1]) != COMPACT_DECODER):
        raise ValueError("Effective compact JSON decoder differs from the target profile")
    members = owned_group_members(process.pid)
    listeners = {c.pid for c in psutil.net_connections(kind="tcp") if c.status == psutil.CONN_LISTEN and c.laddr.port == 8001}
    if not listeners or not listeners.issubset(set(members)):
        raise ValueError("Candidate listener does not belong to the owned launch group")
    return {"effective_argv": argv, "listener_pids": sorted(listeners), "create_time": psutil.Process(process.pid).create_time()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class Resources:
    def __init__(self, output, server_pid=None, enforce_storage=False):
        self.output, self.server_pid = output, server_pid
        self.enforce_storage, self.storage_error = enforce_storage, None
        self.samples, self.stop = [], threading.Event()
        self.phase = "startup"
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        import psutil
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        while not self.stop.is_set():
            sample = {"monotonic": time.monotonic(), "phase": self.phase}
            if self.enforce_storage:
                try:
                    sample["host_storage"] = host_storage()
                    validate_host_storage(sample["host_storage"])
                except (OSError, ValueError) as error:
                    self.storage_error = str(error)
                    sample["storage_error"] = self.storage_error
                    self.samples.append(sample)
                    with self.output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(sample) + "\n")
                    if self.server_pid:
                        try:
                            os.killpg(self.server_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    return
            try:
                row = subprocess.check_output([smi, "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
                used, free = map(float, row.split(","))
                sample.update(device_vram_used_mib=used, device_vram_free_mib=free)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                sample["gpu_measurement_unavailable"] = type(error).__name__
            if self.server_pid:
                try:
                    root = psutil.Process(self.server_pid)
                    sample["server_tree_rss_mib"] = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)]) / 1024**2
                except (psutil.Error, OSError):
                    sample["server_tree_rss_mib"] = None
            self.samples.append(sample)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample) + "\n")
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(7)
        return summarize_resources(self.samples)


class Probe:
    def __init__(self, url, model, output, context_tokens=4096):
        self.url, self.model, self.output = url.rstrip("/"), model, output
        self.context_tokens = context_tokens
        self.root = self.url.removesuffix("/v1")
        self.client = httpx.Client(trust_env=False, timeout=120)

    def request(self, method, path, body=None, timeout=120):
        started = time.monotonic()
        record = {"method": method, "path": path, "body": body}
        try:
            response = self.client.request(method, self.root + path, json=body, timeout=timeout)
            record.update(status=response.status_code, raw_response=response.text)
            return response
        except httpx.TransportError as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            with (self.output / "responses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def count(self, body):
        response = self.request("POST", "/tokenize", token_body(body))
        response.raise_for_status()
        data = response.json()
        if type(data.get("count")) is not int or data.get("max_model_len") != self.context_tokens:
            raise ValueError(f"Tokenizer response does not establish the {self.context_tokens}-token budget")
        return data["count"]

    def completion(self, body):
        count = self.count(body)
        response = self.request("POST", "/v1/chat/completions", body)
        response.raise_for_status()
        return validate_completion(body, response.json(), count)

    def metrics(self, timeout=5):
        response = self.request("GET", "/metrics", timeout=timeout)
        response.raise_for_status()
        return scheduler_counts(response.text)

    def run(self, checks):
        for i in range(3):
            self.completion(chat_body(self.model, f"Say hello in at most five words. Probe {i + 1}."))
        checks["json_requests"] = True

        low, high, best = 1, self.context_tokens * 2, None
        prompt_target = self.context_tokens - 512 - 34
        while low <= high:
            size = (low + high) // 2
            body = chat_body(self.model, "Read the reference, then answer hello. Reference:\n" + "padding " * size, 512)
            count = self.count(body)
            if count <= prompt_target:
                best = (body, count)
                low = size + 1
            else:
                high = size - 1
        if best is None or best[1] < prompt_target - 250:
            raise ValueError("Long-context input was not near the requested context boundary")
        self.completion(best[0])
        checks["long_context"] = True

        oversized = chat_body(self.model, "padding " * (self.context_tokens * 3))
        if self.count(oversized) <= self.context_tokens:
            raise ValueError("Oversized prompt did not exceed the context limit")
        response = self.request("POST", "/v1/chat/completions", oversized)
        checks["context_rejected"] = (response.status_code in (400, 413, 422)
            and bool(re.search(r"context|maximum.{0,30}token|too many tokens|length", response.text, re.I)))
        if not checks["context_rejected"]:
            raise ValueError("Over-context input was not explicitly rejected")

        short = chat_body(self.model, "Return a long greeting containing at least forty characters.", 1)
        short["response_format"]["json_schema"]["schema"]["properties"]["answer"]["minLength"] = 40
        count = self.count(short)
        response = self.request("POST", "/v1/chat/completions", short)
        response.raise_for_status()
        if response.json()["choices"][0].get("finish_reason") != "length":
            raise ValueError("Truncation probe did not produce a real token-limited output")
        try:
            validate_completion(short, response.json(), count)
        except ValueError:
            checks["truncation_rejected"] = True

        long = chat_body(self.model, "Write a very long greeting in the answer field.", 2048)
        long["response_format"]["json_schema"]["schema"]["properties"]["answer"] = {"type": "string", "minLength": 20000, "maxLength": 20000}
        long.update(stream=True, stream_options={"include_usage": True})
        cancellation = {"request": long, "events": [], "observed_running": False, "first_content": None}
        try:
            with self.client.stream("POST", self.url + "/chat/completions", json=long) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    cancellation["events"].append(line)
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    choices = event.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        cancellation["first_content"] = choices[0]["delta"]["content"]
                        cancellation["observed_running"] = self.metrics()["running"] > 0
                        if choices[0].get("finish_reason") is not None:
                            raise ValueError("Cancellation request completed naturally before interruption")
                        break
            if not cancellation["observed_running"]:
                raise ValueError("Cancellation did not observe active generation before disconnect")
            cancellation["idle_after_seconds"] = wait_for_idle(lambda remaining: self.metrics(timeout=min(1, remaining)))
            self.completion(chat_body(self.model, "Say hello after cancellation."))
            checks["cancellation"] = True
        except Exception as error:
            cancellation["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save(self.output / "cancellation.json", cancellation)


def prepare_snapshot(output):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    if snapshot.name != REVISION:
        raise ValueError("Snapshot revision does not match the pinned checkpoint")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                    digest.update(chunk)
            files.append({"name": str(path.relative_to(snapshot)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    if not any(f["name"].endswith(".safetensors") for f in files):
        raise ValueError("Pinned snapshot has no downloaded weights")
    report = {"model": MODEL, "revision": REVISION, "tokenizer_revision": REVISION,
              "snapshot_path": str(snapshot), "total_bytes": sum(f["bytes"] for f in files), "files": files}
    save(output / "snapshot.json", report)
    return report


def publish_qualification(report, output, root):
    lock_hash = hashlib.sha256((root / "requirements-vllm.lock").read_bytes()).hexdigest()
    if report.get("cleanup", {}).get("remaining_pids") != []:
        raise ValueError("Probe cleanup must be verified before publication")
    report_path = output / "probe.json"
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    if on_disk != report:
        raise ValueError("Probe report differs from persisted evidence")
    fingerprint = profile_fingerprint(report["profile"])
    proof = {**report, "probe_report": {"path": str(report_path),
             "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}}
    validate_runtime_qualification(proof.get("profile"), proof, lock_hash)
    path = root / "tmp/vllm/qualifications" / f"{fingerprint}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.new")
    save(temporary, proof)
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--batch-tokens", choices=(512, 1024), type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", choices=("0.65", "0.80", "0.85"), default="0.65")
    parser.add_argument("--context-tokens", choices=(4096, 8192), type=int, default=4096)
    parser.add_argument("--reasoning-parser", choices=("none", "qwen3"), default="none")
    parser.add_argument("--publish-qualification", action="store_true")
    args = parser.parse_args()
    if args.publish_qualification and not args.launch:
        parser.error("Publishing qualification requires an owned --launch")
    parsed = urlsplit(args.url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != 8001 or parsed.path != "/v1":
        parser.error("Task 2 targets only http://127.0.0.1:8001/v1")
    output = Path(args.output).resolve()
    if os.name == "nt":
        if output.drive.upper() != "D:":
            parser.error("Windows probe artifacts must be on D")
    elif args.launch and not output.is_relative_to(Path("/opt/mas-vllm")):
        parser.error("Launch artifacts must be within /opt/mas-vllm in the D-backed distro")
    if output.exists():
        parser.error("Choose a new output directory; existing probe evidence is immutable")
    output.mkdir(parents=True)
    report = {"task": 2, "model": args.model, "revision": REVISION, "checks": {name: False for name in CHECKS},
              "hardware_gate_passed": False, "provenance_verified": False,
              "qualification": "owned launch required" if not args.launch else "pending",
              "batch_tokens": args.batch_tokens, "gpu_memory_utilization": args.gpu_memory_utilization}
    process = None
    resources = Resources(output / "resources.jsonl", enforce_storage=args.launch)
    probe = Probe(args.url, args.model, output, context_tokens=args.context_tokens)
    root = Path(__file__).resolve().parents[1]
    reasoning_parser = None if args.reasoning_parser == "none" else args.reasoning_parser
    report["profile"] = {**PINNED, "port": 8001, "max_model_len": args.context_tokens,
                         "max_num_batched_tokens": args.batch_tokens,
                         "gpu_memory_utilization": float(args.gpu_memory_utilization),
                         "structured_outputs_config": COMPACT_DECODER}
    if reasoning_parser:
        report["profile"]["reasoning_parser"] = reasoning_parser
    report["profile_fingerprint"] = profile_fingerprint(report["profile"]) if args.publish_qualification else None
    report["requirements_lock"] = {"sha256": hashlib.sha256((root / "requirements-vllm.lock").read_bytes()).hexdigest()}
    started = time.monotonic()
    if args.launch and os.name == "posix":
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Probe received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.launch:
            if os.name == "nt" or args.model != MODEL:
                raise ValueError("Launch mode requires Linux and the pinned original model")
            report["host_storage_before"] = host_storage()
            validate_host_storage(report["host_storage_before"])
            with socket.socket() as check:
                if check.connect_ex(("127.0.0.1", 8001)) == 0:
                    raise ValueError("Candidate port is already owned by another process")
            report["snapshot"] = prepare_snapshot(output)
            launcher = Path(__file__).resolve().parent / "wsl/start_vllm.sh"
            command = ["/bin/bash", str(launcher), "8001", str(args.batch_tokens), args.gpu_memory_utilization,
                       "true", str(args.context_tokens), args.reasoning_parser]
            report["launch_command"] = command
            with (output / "server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report["owned_process"] = {"pid": process.pid, "process_group": process.pid}
            resources.server_pid = process.pid
        resources.thread.start()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup with code {process.returncode}; inspect server.log")
            try:
                response = probe.client.get(probe.root + "/health", timeout=2)
                if response.status_code == 200:
                    models = probe.request("GET", "/v1/models").json()
                    if args.model not in {item["id"] for item in models["data"]}:
                        raise ValueError("Readiness serves another model")
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("Startup exceeded 600 seconds")
        report["startup_seconds"] = time.monotonic() - started
        if process:
            report["launch_identity"] = verify_launch_identity(process, args.batch_tokens, args.gpu_memory_utilization,
                                                               args.context_tokens, reasoning_parser=reasoning_parser)
            from importlib import metadata
            import torch
            report["versions"] = {name: metadata.version(name) for name in ("vllm", "torch", "transformers", "triton")}
            report["torch_cuda"] = torch.version.cuda
            report["provenance_verified"] = True
            report["qualification"] = "owned pinned launch verified"
        resources.phase = "inference"
        probe.run(report["checks"])
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if args.launch and os.name == "posix":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            probe.client.close()
        except Exception as error:
            report.setdefault("cleanup_errors", []).append(f"client: {error}")
        finalize_probe(report, resources, process, output, started)
        if args.publish_qualification and report["hardware_gate_passed"]:
            publish_qualification(report, output, root)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["hardware_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
