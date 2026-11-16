"""Run 30 authored scenarios x 3 rounds against the real configured local server."""

import argparse
import json
import time
from pathlib import Path

from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec
from mas_demo.llm.benchmark import load_cases, run_case, summarize
from mas_demo.llm.settings import create_gateway
from mas_demo.llm.health import wait_for_server
from mas_demo.llm.capture import Capture, write_json


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="artifacts/local-qwen-benchmark")
    parser.add_argument("--cases")
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    spec = TaskSpec(analysis_mode="local", model_id=args.model)
    gateway = create_gateway(spec)
    wait_for_server(gateway)
    records = []
    started = time.monotonic()
    cases = [case for case in load_cases(args.cases) if not args.case_ids or case["id"] in args.case_ids]
    with Capture(args.output, gateway, args.cases, {"rounds": args.rounds, "case_ids": args.case_ids, "warmup": "first selected case per role, excluded from scored records"}) as capture:
        warmed = set()
        for case in cases:
            if case["role"] not in warmed:
                capture.append("warmup.jsonl", run_case(case, spec, gateway))
                warmed.add(case["role"])
        for iteration in range(args.rounds):
            for case in cases:
                record = {"iteration": iteration + 1, **run_case(case, spec, gateway)}
                records.append(record)
                capture.append("responses.jsonl", record)
                print(f"{len(records)} {case['id']}: schema={record['schema_valid']} correct={record['correct']}", flush=True)
        report = {
            "schema_version": 3,
            "model": args.model,
            "mode": "local",
            "rounds": args.rounds,
            "wall_seconds": time.monotonic() - started,
            **summarize(records),
        }
        (capture.directory / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
