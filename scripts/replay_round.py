"""Replay a saved Analysis/Planner round; never run training or edit the source."""
import argparse
import json
from pathlib import Path

from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec
from mas_demo.llm.capture import Capture, write_json
from mas_demo.llm.health import wait_for_server
from mas_demo.llm.round_replay import load_round, run_round_replay
from mas_demo.llm.settings import create_gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hypotheses", help="JSON with reviewed hypotheses: replay Planner only, explicitly labelled in the report")
    args = parser.parse_args()
    source, output = Path(args.run).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents:
        parser.error("Replay output must be outside the original run directory")
    case = load_round(source, args.round)
    if args.hypotheses:
        case["provided_hypotheses"] = json.loads(Path(args.hypotheses).read_text(encoding="utf-8"))
        case["provided_hypotheses_path"] = str(Path(args.hypotheses).resolve())
    load_environment()
    gateway = create_gateway(TaskSpec.model_validate(case["spec"]))
    wait_for_server(gateway)
    mode = "planner_only" if args.hypotheses else "pipeline"
    with Capture(output, gateway, source / "events.jsonl", {"kind": "round_replay", "round": args.round, "mode": mode}) as capture:
        write_json(output / "inputs.json", case)
        def emit(event):
            capture.append("responses.jsonl", event)
            print(f"{event['kind']} completed", flush=True)
        report = run_round_replay(case, gateway, emit=emit)
        write_json(output / "report.json", report)
    print(json.dumps(dict(mode=report["mode"], outcome=report["outcome"], grounding_preserved=report["grounding_preserved"],
                         manual_review_pending=True, report=str(output / "report.json")), indent=2))
    if report["outcome"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
