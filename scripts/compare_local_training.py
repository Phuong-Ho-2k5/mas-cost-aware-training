"""Paired multi-seed real-training pilot; run separately from agent benchmarks."""
import argparse
import json
from pathlib import Path
import time

from mas_demo.cli import load_environment, load_spec
from mas_demo.contracts import DiagnosisReport
from mas_demo.llm.settings import create_gateway
from mas_demo.llm.health import wait_for_server
from mas_demo.orchestration import run


class NoInterventionBaseline:
    """Explicit experimental baseline, never an error fallback for an LLM run."""
    def complete(self, evidence):
        return DiagnosisReport(status="ABSTAIN", mode="local", reason="Explicit no-intervention baseline"), {
            "agent": "analysis", "mode": "disabled_baseline", "model": None, "attempts": 0,
            "latency_seconds": 0, "usage": {}}


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_qwen_cifar.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output", default="artifacts/local-qwen-training-pilot")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "pilot.json").exists():
        parser.error("Pilot report already exists; choose a new output directory")
    base = load_spec(args.config)
    gateway = create_gateway(base)
    wait_for_server(gateway)
    rows = []
    for seed in args.seeds:
        for policy in ("no_intervention", "local_llm"):
            spec = base.model_copy(update={"seed": seed})
            directory = root / f"seed-{seed}-{policy}"
            started = time.monotonic()
            state = run(spec, directory, gateway=NoInterventionBaseline() if policy == "no_intervention" else gateway)
            wall = time.monotonic() - started
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            agent_calls = [e["metadata"] for e in events if "metadata" in e and e["metadata"].get("attempts", 0)]
            row = {"seed": seed, "policy": policy, "status": state["status"], "epoch": state["epoch"],
                   "best_accuracy": state["best_accuracy"], "worker_seconds": state["budget"]["consumed"],
                   "llm_seconds": sum(m.get("latency_seconds", 0) for m in agent_calls),
                   "llm_calls": len(agent_calls), "wall_seconds": wall,
                   "model": spec.model_id if policy == "local_llm" else None,
                   "manifest": json.loads((directory / "dataset_manifest.json").read_text())}
            rows.append(row)
            report = {"runs": rows, "config": base.model_dump(), "seeds": args.seeds,
                "limitations": "Small paired pilot, not a statistical efficacy claim. Same per-run training-worker budget and epoch cap; LLM/control/trial wall times reported separately, not matched total-compute experiments. Cloud comparison not run."}
            (root / "pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in row.items() if k != "manifest"}), flush=True)


if __name__ == "__main__":
    main()
