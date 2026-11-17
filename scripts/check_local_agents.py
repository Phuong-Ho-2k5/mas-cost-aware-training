"""Probe all three roles directly even when a training smoke run abstains."""
import json
import argparse
from mas_demo.llm.capture import Capture, write_json
from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec
from mas_demo.llm.benchmark import load_cases, run_case, summarize
from mas_demo.llm.settings import create_gateway
from mas_demo.llm.health import wait_for_server


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = TaskSpec(analysis_mode="vllm", model_id=args.model)
    gateway = create_gateway(spec)
    wait_for_server(gateway)
    records = []
    with Capture(args.output, gateway, settings={"kind": "three_role_probe"}) as capture:
        for case in load_cases():
            if case["id"] in ("a-unstable", "p-one", "e-gain"):
                record = run_case(case, spec, gateway)
                records.append(record)
                capture.append("responses.jsonl", record)
        write_json(capture.directory / "report.json", summarize(records))
    print(json.dumps({"records": records, "summary": summarize(records)}, indent=2))
    if not all(r["semantic_valid"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
