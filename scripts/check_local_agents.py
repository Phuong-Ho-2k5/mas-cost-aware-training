"""Probe all three roles directly even when a training smoke run abstains."""
import json
from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec
from mas_demo.llm.benchmark import load_cases, run_case, summarize
from mas_demo.llm.settings import create_gateway
from mas_demo.llm.health import wait_for_server


def main():
    load_environment()
    spec = TaskSpec(analysis_mode="local")
    gateway = create_gateway(spec)
    wait_for_server(gateway)
    records = [run_case(case, spec, gateway) for case in load_cases() if case["id"] in ("a-unstable", "p-one", "e-gain")]
    print(json.dumps({"records": records, "summary": summarize(records)}, indent=2))
    if not all(r["semantic_valid"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
