"""Exercise the real vLLM gateway through an owned local Postman simulation.

Only process provenance is stubbed: the simulation deliberately is not vLLM.
HTTP requests, serialization, deadlines and agent error handling run for real.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import httpx
import psutil
from mas_demo.agents.analysis import diagnose
from mas_demo.agents.planner import plan_with_agent
from mas_demo.agents.evaluation import evaluate_with_agent
from mas_demo.configuration import training_capabilities
from mas_demo.contracts import EvidenceWindow, EpochEvent, TaskSpec
from mas_demo.llm.gateway import GatewayError, VLLMGateway
from mas_demo.llm import health
from vllm_simulation_process import OwnedSimulation


def finalize_report(report, events, cleanup):
    report.update(request_events=len(events), cleanup=cleanup)
    if not events or cleanup.get("active_processes") != 0 or cleanup.get("error"):
        report.update(passed=False, error=report.get("error") or "Request evidence or clean shutdown was not established")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if os.name == "nt" and output.drive.upper() != "D:":
        raise ValueError("Save simulation evidence on D")
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [shutil.which("postman"), "simulation", "run", "postman/simulations/vllm.sim.yaml",
               "--port", str(port), "--no-history", "--output", "ndjson"]
    records, report = [], {"scope": "real HTTP caller; process provenance stubbed; no live inference"}
    process, container = None, None
    client_type, async_type = httpx.Client, httpx.AsyncClient
    verifier = health.verify_vllm_runtime
    fault = ["success"]
    try:
        with (output / "simulation.ndjson").open("wb") as stdout, (output / "simulation.err").open("wb") as stderr:
            container = OwnedSimulation()
            process = container.start(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        owned = psutil.Process(process.pid)
        report["simulation_process"] = {"pid": owned.pid, "create_time": owned.create_time(), "argv": owned.cmdline()}
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 45
        with client_type(trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Simulation exited; inspect simulation.err")
                try:
                    response = client.get(url + "/health", headers={"x-mock-slug": "vllm"}, timeout=1)
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.3)
            else:
                raise TimeoutError("Simulation did not listen")

        def client_factory(original):
            def create(*args, **kwargs):
                kwargs["headers"] = {**kwargs.get("headers", {}), "x-mock-slug": "vllm", "x-mas-fault": fault[0]}
                return original(*args, **kwargs)
            return create

        httpx.Client, httpx.AsyncClient = client_factory(client_type), client_factory(async_type)
        health.verify_vllm_runtime = lambda gateway: {"engine": "stand-in", "context_tokens": 4096,
                                                    "structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True},
                                                    "provenance": "deliberately stubbed for HTTP fault injection"}
        model = "Qwen/Qwen3.5-0.8B"
        gateway = VLLMGateway("", model, url + "/v1", mode="vllm", local_backend="vllm",
            attempts=1, timeout=0.25, deadline=2, max_tokens=64)
        spec = TaskSpec(analysis_mode="vllm", model_id=model)
        evidence = EvidenceWindow(epochs=[], branch_id="main", status="INSUFFICIENT")
        health.wait_for_server(gateway, timeout=1)
        result, meta = gateway.complete({})
        assert result.status == "ABSTAIN" and meta["finish_reason"] == "stop"
        records.append({"case": "success", "metadata": meta})
        result, meta = diagnose(spec, evidence, [], gateway)
        assert result is not None and result.status == "ABSTAIN" and not meta.get("error")
        records.append({"case": "two-stage-abstain", "metadata": meta})
        fault[0] = "diagnosed"
        observed = EvidenceWindow(epochs=[1, 2], branch_id="main", status="READY", recent_events=[
            EpochEvent(epoch=i, branch_id="main", train_loss=float(i), val_loss=float(i),
                       val_accuracy=.8 / i, grad_norm=float(i), learning_rate=.01) for i in (1, 2)])
        result, meta = diagnose(spec, observed, [], gateway)
        assert result is not None and result.status == "DIAGNOSED" and len(result.hypotheses) == 3
        assert not meta.get("error") and len(meta["calls"]) == 2
        records.append({"case": "two-stage-diagnosed", "metadata": meta})
        fault[0] = "models-mismatch"
        try:
            health.wait_for_server(gateway, timeout=1)
        except RuntimeError as error:
            records.append({"case": fault[0], "error": str(error)})
        else:
            raise AssertionError("Readiness accepted a different model")

        for case in ("429", "503", "timeout", "malformed", "length", "model-mismatch", "token404"):
            fault[0] = case
            try:
                gateway.complete({})
            except GatewayError as error:
                records.append({"case": case, "error": str(error), "metadata": error.metadata})
                assert error.metadata["attempts"] <= 1
            else:
                raise AssertionError(f"Fault {case} did not fail closed")
            diagnosis, analysis_meta = diagnose(spec, evidence, [], gateway)
            plan, plan_meta = plan_with_agent(spec, None, evidence, spec.training,
                training_capabilities(), [], 1, 8,  gateway=gateway)
            verdict, eval_meta = evaluate_with_agent(spec, [], {}, 4, gateway)
            assert diagnosis is None and analysis_meta.get("error")
            assert plan is None and plan_meta.get("error")
            assert verdict.decision == "INCONCLUSIVE" and eval_meta.get("error")
            records[-1]["agent_fallbacks"] = {"analysis": analysis_meta, "planner": plan_meta, "evaluation": eval_meta}
        report.update(passed=True, cases=records)
    except BaseException as error:
        report.update(passed=False, error=str(error), cases=records)
        raise
    finally:
        httpx.Client, httpx.AsyncClient = client_type, async_type
        health.verify_vllm_runtime = verifier
        try:
            cleanup = container.stop() if container is not None else {"active_processes": 0}
        except Exception as error:
            cleanup = {"active_processes": None, "error": str(error)}
        time.sleep(0.2)
        events = []
        for line in (output / "simulation.ndjson").read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
                if event.get("event") == "request" or event.get("type") == "request":
                    events.append(event)
            except ValueError:
                pass
        finalize_report(report, events, cleanup)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise AssertionError(report["error"])
    print(json.dumps({"passed": report["passed"], "cases": len(records), "request_events": len(events)}))


if __name__ == "__main__":
    main()
