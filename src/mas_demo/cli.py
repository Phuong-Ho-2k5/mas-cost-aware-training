import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import yaml
from dotenv import load_dotenv
from mas_demo.contracts import TaskSpec
from mas_demo.data import prepare_data
from mas_demo.execution.local import LocalBackend
from mas_demo.io import read_json, write_json
from mas_demo.orchestration import run
from mas_demo.training.worker import resolve_device
from mas_demo.llm.settings import create_gateway


def load_spec(path):
    return TaskSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def load_environment(path=None):
    load_dotenv(dotenv_path=path if path is not None else Path(__file__).resolve().parents[2]/".env",
                override=False)


def main():
    load_environment()
    parser = argparse.ArgumentParser(description="MAS training demo — local, one worker at a time")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "profile", "prepare"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/high_lr.yaml")
        cmd.add_argument("--device", choices=["cpu", "cuda", "auto"])
        if name != "prepare":
            cmd.add_argument("--output")
        if name == "run":
            cmd.add_argument("--analysis", choices=["openrouter", "local"],
                             help="LLM backend shared by all three agents")
    resume = sub.add_parser("resume")
    resume.add_argument("output")
    stop = sub.add_parser("stop")
    stop.add_argument("output")
    args = parser.parse_args()
    if args.command == "stop":
        directory = Path(args.output)
        if not (directory/"state.json").exists():
            parser.error("Run state does not exist")
        (directory/"stop.flag").touch()
        print("Stop requested. Worker finishes current safe segment/verification round first.")
        return
    if args.command == "resume":
        spec = TaskSpec.model_validate(read_json(Path(args.output)/"config.json"))
        try:
            create_gateway(spec)
        except ValueError as error:
            parser.error(str(error))
        flag = Path(args.output)/"stop.flag"
        flag.unlink(missing_ok=True)
        result = run(spec, args.output, resume=True)
    else:
        spec = load_spec(args.config)
        if args.device:
            spec.device = args.device
        if args.command == "run" and args.analysis:
            spec.analysis_mode = args.analysis
            if args.analysis == "local" and spec.model_id == TaskSpec().model_id:
                spec.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        if args.command == "run":
            try:
                create_gateway(spec)
            except ValueError as error:
                parser.error(str(error))
        spec.device = resolve_device(spec.device)
        if args.command == "prepare":
            manifest = prepare_data(spec)
            print(json.dumps(manifest.model_dump(exclude={"train_indices", "val_indices"}), indent=2))
            return
        output = args.output or f"artifacts/{datetime.now():%Y%m%d-%H%M%S-%f}"
        if args.command == "profile":
            prepare_data(spec)
            result = LocalBackend().run(spec, None, 1, "profile", output, spec.budget_seconds)
            profile = {"status": result.status, "resource_type": spec.device,
                       "worker_seconds": result.seconds,
                       "hot_epoch_seconds": result.events[-1].seconds if result.events else None,
                       "peak_vram_mb": result.events[-1].peak_vram_mb if result.events else None,
                       "recommended_epoch_estimate": result.seconds,
                       "note": "One profile segment; benchmark all policies with a frozen config"}
            write_json(Path(output)/"profile.json", profile)
            print(json.dumps(profile, indent=2))
            if result.status != "COMPLETE":
                raise SystemExit(1)
            return
        print(f"Run: {Path(output).resolve()} | {spec.device} | agents={spec.analysis_mode}", flush=True)
        result = run(spec, output)
    print(json.dumps({k: result.get(k) for k in ["status", "epoch", "round", "reason", "budget", "resource_type", "gpu_hours", "best_accuracy"]}, indent=2))
    if result["status"] in ("FAILED", "INTERRUPTED", "NUMERICAL_ERROR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
