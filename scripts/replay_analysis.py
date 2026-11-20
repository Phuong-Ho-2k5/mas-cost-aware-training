"""Compare Analysis on immutable saved evidence, without running training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from mas_demo.agents.analysis import diagnose
from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec, EvidenceWindow, TrainingConfig
from mas_demo.llm.capture import Capture, write_json
from mas_demo.llm.health import wait_for_server
from mas_demo.llm.settings import create_gateway


def load_replay_cases(path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Replay needs a nonempty case list")
    seen = set()
    for case in cases:
        identifier = case["case_id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Duplicate or empty replay case ID")
        seen.add(identifier)
        TaskSpec.model_validate(case["spec"])
        EvidenceWindow.model_validate(case["evidence"])
        TrainingConfig.model_validate(case["current_config"])
        if case.get("expected_detection") not in (None, "DIAGNOSED", "ABSTAIN"):
            raise ValueError("Invalid expected detection")
    return cases


def run_replay_case(case, gateway):
    value = deepcopy(case)
    result, meta = diagnose(
        TaskSpec.model_validate(value["spec"]),
        EvidenceWindow.model_validate(value["evidence"]),
        value.get("history", []), gateway=gateway,
        current_config=TrainingConfig.model_validate(value["current_config"]),
    )
    detected = (meta.get("detection") or {}).get("status")
    if detected is None and result is not None and not meta.get("error"):
        detected = result.status
    expected = value.get("expected_detection")
    return dict(
        case_id=value["case_id"], expected_detection=expected,
        actual_detection=detected,
        detection_correct=(detected == expected if expected is not None else None),
        diagnosis_ready=result is not None and result.status == "DIAGNOSED",
        model_abstained=result is not None and result.status == "ABSTAIN" and not meta.get("error"),
        output=result.model_dump() if result is not None else None,
        metadata=meta,
    )


def summarize_replay(records):
    labelled = [r for r in records if r.get("detection_correct") is not None]
    detected = [r for r in records if r.get("actual_detection") == "DIAGNOSED"]
    tokens, stages = {}, {}
    def add_usage(target, usage):
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    target[key] = target.get(key, 0) + value
    for record in records:
        meta = record.get("metadata", {})
        add_usage(tokens, meta.get("usage"))
        for call in meta.get("calls") or [meta]:
            stage = call.get("stage", "legacy")
            row = stages.setdefault(stage, dict(calls=0, repairs=0, errors=0, latency_seconds=0, token_usage={}))
            row["calls"] += 1
            row["repairs"] += call.get("attempt_index") == 1
            row["errors"] += bool(call.get("error"))
            row["latency_seconds"] += call.get("latency_seconds", 0)
            add_usage(row["token_usage"], call.get("usage"))
    return dict(
        count=len(records), labelled_count=len(labelled),
        detection_correct=sum(r["detection_correct"] for r in labelled),
        diagnosis_ready=sum(r["diagnosis_ready"] for r in records),
        valid_abstentions=sum(r["model_abstained"] for r in records),
        execution_errors=sum(bool(r.get("metadata", {}).get("error")) for r in records),
        latency_seconds=sum(r.get("metadata", {}).get("latency_seconds", 0) for r in records),
        detected_count=len(detected),
        ready_given_detected=(sum(r["diagnosis_ready"] for r in detected) / len(detected) if detected else None),
        token_usage=tokens, stage_totals=stages,
        manual_review_pending=True,
    )


def execute_cases(cases, repeat, gateway, capture):
    capture.append("warmup.jsonl", run_replay_case(cases[0], gateway))
    records = []
    for iteration in range(repeat):
        for case in cases:
            record = dict(iteration=iteration + 1, **run_replay_case(case, gateway))
            capture.append("responses.jsonl", record)
            records.append(record)
            print(f"{len(records)} {case['case_id']}: {record.get('actual_detection')} ready={record.get('diagnosis_ready')}", flush=True)
    return records


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = load_replay_cases(args.input)
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.case_ids:
        available = {c["case_id"] for c in cases}
        if set(args.case_ids) - available:
            parser.error("Unknown case IDs")
        cases = [c for c in cases if c["case_id"] in args.case_ids]
    gateway = create_gateway(TaskSpec.model_validate(cases[0]["spec"]))
    if any(c["spec"]["model_id"] != gateway.model for c in cases):
        parser.error("All cases must use the same model")
    wait_for_server(gateway)
    with Capture(args.output, gateway, args.input, {"kind": "analysis_replay", "repeat": args.repeat}) as capture:
        write_json(capture.directory / "inputs.json", cases)
        records = execute_cases(cases, args.repeat, gateway, capture)
        report = summarize_replay(records)
        write_json(capture.directory / "report.json", report)
    print(json.dumps(report, indent=2))
    if report["execution_errors"] or report["detection_correct"] != report["labelled_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
