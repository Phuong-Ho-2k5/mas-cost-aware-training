"""CLI for autonomous trial generation. Legacy run artifacts are view-only."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv
from mas_demo.contracts import TaskSpec, TrainingConfig, ExperimentPolicy
from mas_demo.data import prepare_data
from mas_demo.execution.local import LocalBackend
from mas_demo.io import read_json, write_json
from mas_demo.orchestration import run
from mas_demo.training.worker import resolve_device
from mas_demo.llm.settings import create_gateway


def load_spec(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TaskSpec.model_validate(values)


def load_environment(path=None):
    load_dotenv(
        dotenv_path=(
            path if path is not None else Path(__file__).resolve().parents[2] / ".env"
        ),
        override=False,
    )


def migrate_legacy_config(values):
    if values.get("schema_version") == 3:
        return TaskSpec.model_validate(values)
    if values.get("fault_epoch") is not None:
        raise ValueError(
            "Legacy fault injection must be converted explicitly to initial training settings"
        )
    training = {
        key: values[key] for key in TrainingConfig.model_fields if key in values
    }
    policy = {
        key: values[key] for key in ExperimentPolicy.model_fields if key in values
    }
    run_fields = {
        key: values[key]
        for key in TaskSpec.model_fields
        if key in values
        and key
        not in (
            "schema_version",
            "training",
            "policy",
            "window",
            "observation_interval",
        )
    }
    return TaskSpec(
        **run_fields, training=training, policy=policy, observation_interval=1, window=5
    )


def main():
    load_environment()
    parser = argparse.ArgumentParser(
        description="MAS autonomous configuration trials (protocol v3)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "profile", "prepare"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", default="configs/autonomous_mixed.yaml")
        cmd.add_argument("--device", choices=["cpu", "cuda", "auto"])
        if name != "prepare":
            cmd.add_argument("--output")
        if name == "run":
            cmd.add_argument("--analysis", choices=["local", "openrouter"])
            cmd.add_argument("--model")
    for name in ("resume", "stop"):
        commands.add_parser(name).add_argument("output")
    migrate = commands.add_parser("migrate-config")
    migrate.add_argument("source")
    migrate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "migrate-config":
        target = Path(args.output)
        if target.exists():
            parser.error("Output already exists")
        values = yaml.safe_load(Path(args.source).read_text(encoding="utf-8")) or {}
        spec = migrate_legacy_config(values)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"Migrated initial settings to {target}; old action menus and horizons were discarded."
        )
        return
    if args.command == "stop":
        directory = Path(args.output)
        if not (directory / "state.json").exists():
            parser.error("Run state does not exist")
        (directory / "stop.flag").touch()
        print("Stop requested at next safe epoch/verification boundary")
        return
    try:
        if args.command == "resume":
            directory = Path(args.output)
            saved = read_json(directory / "state.json")
            if saved.get("schema_version") != 3:
                parser.error(
                    "Legacy run is view-only. Use migrate-config and start a new v3 run."
                )
            spec = TaskSpec.model_validate(read_json(directory / "config.json"))
            create_gateway(spec)
            (directory / "stop.flag").unlink(missing_ok=True)
            result = run(spec, directory, resume=True)
        else:
            spec = load_spec(args.config)
            if args.device:
                spec.device = args.device
            if args.command == "run" and args.analysis:
                spec.analysis_mode = args.analysis
            if args.command == "run" and args.model:
                spec.model_id = args.model
            spec.device = resolve_device(spec.device)
            if args.command == "prepare":
                manifest = prepare_data(spec)
                print(
                    json.dumps(
                        manifest.model_dump(exclude={"train_indices", "val_indices"}),
                        indent=2,
                    )
                )
                return
            output = args.output or f"artifacts/{datetime.now():%Y%m%d-%H%M%S-%f}"
            if args.command == "profile":
                prepare_data(spec)
                segment = LocalBackend().run(
                    spec, None, 1, "profile", output, spec.budget_seconds
                )
                result = {
                    "status": segment.status,
                    "worker_seconds": segment.seconds,
                    "recommended_epoch_estimate": segment.seconds,
                    "peak_vram_mb": (
                        segment.events[-1].peak_vram_mb if segment.events else None
                    ),
                    "note": "Worker cost includes startup/teardown; LLM not called.",
                }
                write_json(Path(output) / "profile.json", result)
            else:
                create_gateway(spec)
                print(
                    f"Run: {Path(output).resolve()} | {spec.device} | agents={spec.analysis_mode}",
                    flush=True,
                )
                result = run(spec, output)
    except ValueError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "epoch",
                    "round",
                    "reason",
                    "budget",
                    "active_config",
                    "active_config_hash",
                    "llm_seconds",
                    "wall_seconds",
                    "best_accuracy",
                )
                if key in result
            },
            indent=2,
        )
    )
    if result["status"] in ("FAILED", "INTERRUPTED", "NUMERICAL_ERROR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
