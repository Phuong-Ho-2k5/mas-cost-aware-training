"""Probe a protocol-v3 HTTP stand-in; pytest's provider fixture needs no service."""

import argparse
import json
from mas_demo.contracts import TaskSpec
from mas_demo.llm.benchmark import load_cases, run_case
from mas_demo.llm.gateway import OpenRouterGateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4010")
    args = parser.parse_args()
    gateway = OpenRouterGateway("simulation-only", base_url=args.url, retry_delay=0)
    spec = TaskSpec(analysis_mode="openrouter")
    for case in load_cases():
        if case["id"] in ("a-unstable", "p-one", "e-gain"):
            record = run_case(case, spec, gateway)
            print(json.dumps(record, indent=2))
            if not record["semantic_valid"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
