"""Resume-safe paired benchmark runner (Baseline vs MAS) for converted AUTOTRAINER cases."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import yaml

from mas_demo.cli import load_environment
from mas_demo.contracts import TaskSpec
from mas_demo.data import prepare_data
from mas_demo.llm.health import verify_vllm_runtime
from mas_demo.llm.settings import create_gateway
from mas_demo.orchestration import run as run_mas
from mas_demo.training.worker import resolve_device, train_segment


def select_pilot_10(cases: list[dict]) -> list[dict]:
    """Pick 2 cases per issue (explode, vanish, not_converge, unstable, healthy) balanced across datasets."""
    by_issue = defaultdict(list)
    for c in cases:
        by_issue[c["primary_issue"]].append(c)
    ds_pref = ["Blob", "Circle", "MNIST", "CIFAR-10"]
    picked = []
    for issue in ("explode", "vanish", "not_converge", "unstable", "healthy"):
        pool = by_issue.get(issue, [])
        # Prefer different datasets within the same issue bucket
        chosen = []
        seen_ds = set()
        for ds in ds_pref:
            match = next((x for x in pool if x["dataset_dir"] == ds and x["slug"] not in {c["slug"] for c in chosen}), None)
            if match:
                chosen.append(match)
                seen_ds.add(ds)
                if len(chosen) == 2:
                    break
        for x in pool:
            if len(chosen) == 2:
                break
            if x["slug"] not in {c["slug"] for c in chosen}:
                chosen.append(x)
        picked.extend(chosen)
    return picked


def check_storage(min_c_gib: float = 4.2, min_d_gib: float = 10.0):
    c_free = shutil.disk_usage("C:/").free / 1024**3
    d_free = shutil.disk_usage("D:/").free / 1024**3
    if c_free < min_c_gib or d_free < min_d_gib:
        raise RuntimeError(
            f"Storage guard triggered: C={c_free:.2f} GiB (min {min_c_gib}), D={d_free:.2f} GiB (min {min_d_gib})"
        )


def prune_intermediate_checkpoints(run_dir: Path, keep_path: str | None):
    keep = Path(keep_path).resolve() if keep_path else None
    for pt in run_dir.rglob("*.pt"):
        if keep is not None and pt.resolve() == keep:
            continue
        pt.unlink(missing_ok=True)


def run_baseline_case(spec: TaskSpec, out_dir: Path, prune: bool = False) -> dict:
    summary_path = out_dir / "baseline_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_data(spec)
    started = time.monotonic()
    result = train_segment(
        spec,
        None,
        spec.max_epochs,
        "baseline",
        out_dir,
        spec.budget_seconds,
        training_config=spec.training,
    )
    elapsed = time.monotonic() - started
    accs = [e.val_accuracy for e in result.events if e.val_accuracy is not None]
    losses = [e.val_loss for e in result.events if e.val_loss is not None]
    summary = {
        "mode": "baseline",
        "status": result.status,
        "epochs_completed": len(result.events),
        "best_accuracy": max(accs) if accs else None,
        "final_accuracy": accs[-1] if accs else None,
        "best_val_loss": min(losses) if losses else None,
        "final_val_loss": losses[-1] if losses else None,
        "seconds": round(elapsed, 3),
        "error": result.error,
        "best_checkpoint": result.best_checkpoint.path if result.best_checkpoint else None,
    }
    if prune:
        prune_intermediate_checkpoints(out_dir, summary["best_checkpoint"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_mas_case(spec: TaskSpec, out_dir: Path, prune: bool = False) -> dict:
    summary_path = out_dir / "mas_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_vllm_runtime(create_gateway(spec))
    started = time.monotonic()
    resume = (out_dir / "state.json").exists()
    state = run_mas(spec, out_dir, resume=resume)
    elapsed = time.monotonic() - started
    summary = {
        "mode": "mas",
        "status": state.get("status"),
        "reason": state.get("reason", ""),
        "epochs_completed": state.get("epoch", 0),
        "rounds": state.get("round", 0),
        "best_accuracy": state.get("best_accuracy") if state.get("best_accuracy", -1) >= 0 else None,
        "worker_seconds": round(state.get("budget", {}).get("consumed", 0.0), 3),
        "llm_seconds": round(state.get("llm_seconds", 0.0), 3),
        "wall_seconds": round(elapsed, 3),
        "best_checkpoint": (state.get("best_checkpoint") or {}).get("path"),
    }
    if prune:
        prune_intermediate_checkpoints(out_dir, summary["best_checkpoint"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=r"D:\HK261\autotrainer_raw\converted\suite_manifest.json")
    parser.add_argument("--out-dir", default="artifacts/autotrainer_bench/pilot")
    parser.add_argument("--mode", choices=("baseline", "mas", "both"), default="mas")
    parser.add_argument("--pilot", action="store_true", help="Run 10 stratified pilot cases (2 per issue)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--prune-checkpoints", action="store_true")
    args = parser.parse_args()

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    cases = select_pilot_10(suite["cases"]) if args.pilot else suite["cases"]
    cases = cases[args.start : (args.start + args.limit if args.limit is not None else None)]

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for idx, case in enumerate(cases, 1):
        check_storage()
        raw_spec = yaml.safe_load(Path(case["task_yaml"]).read_text(encoding="utf-8"))
        if args.max_epochs is not None:
            raw_spec["policy"]["max_epochs"] = args.max_epochs
        if args.budget_seconds is not None:
            raw_spec["policy"]["budget_seconds"] = args.budget_seconds
        spec = TaskSpec.model_validate(raw_spec)
        spec.device = resolve_device(spec.device)

        case_dir = out_root / case["slug"]
        row = {
            "slug": case["slug"],
            "case_id": case["case_id"],
            "dataset": case["dataset_dir"],
            "primary_issue": case["primary_issue"],
            "origin_best_acc": case.get("origin_best_acc"),
            "autotrainer_test_acc": case.get("autotrainer_test_acc"),
            "device": spec.device,
            "max_epochs": spec.max_epochs,
        }
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "

        def _delta(a, b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return f"{a - b:+.4f}"
            return "   -   "

        at_orig = row["origin_best_acc"]
        at_fix = row["autotrainer_test_acc"]
        print(f"[{idx}/{len(cases)}] {case['slug']} ({case['dataset_dir']}/{case['primary_issue']}, dev={spec.device})", flush=True)
        print(f"  AT repo:  origin={_fmt(at_orig)} -> repaired={_fmt(at_fix)} (delta={_delta(at_fix, at_orig)})", flush=True)
        if args.mode in ("baseline", "both"):
            row["baseline"] = run_baseline_case(spec, case_dir / "baseline", prune=args.prune_checkpoints)
            b = row["baseline"]
            print(f"  PT base:  best_acc={_fmt(b['best_accuracy'])} status={b['status']} ep={b['epochs_completed']} time={b['seconds']}s", flush=True)
        if args.mode in ("mas", "both"):
            row["mas"] = run_mas_case(spec, case_dir / "mas", prune=args.prune_checkpoints)
            m = row["mas"]
            print(
                f"  MAS:      best_acc={_fmt(m['best_accuracy'])} "
                f"(vs_origin={_delta(m['best_accuracy'], at_orig)}, vs_AT={_delta(m['best_accuracy'], at_fix)}) "
                f"status={m['status']} ep={m['epochs_completed']} rounds={m['rounds']} worker={m['worker_seconds']}s llm={m['llm_seconds']}s",
                flush=True,
            )
        report_rows.append(row)
        (out_root / "benchmark_progress.json").write_text(
            json.dumps({"mode": args.mode, "count": len(report_rows), "rows": report_rows}, indent=2) + "\n",
            encoding="utf-8",
        )

    print("\n=== COMPARISON SUMMARY (AUTOTRAINER vs MAS) ===")
    header = f"{'Case':36s} | {'Issue':12s} | {'AT_Orig':7s} | {'AT_Fix':7s} | {'MAS':7s} | {'MAS-Orig':8s} | {'MAS-AT':8s} | {'MAS_Status'}"
    print(header)
    print("-" * len(header))
    for r in report_rows:
        o_acc = r.get("origin_best_acc")
        f_acc = r.get("autotrainer_test_acc")
        m_acc = (r.get("mas") or {}).get("best_accuracy")
        m_st = (r.get("mas") or {}).get("status", "-")
        o_s = f"{o_acc:.4f}" if isinstance(o_acc, (int, float)) else "-"
        f_s = f"{f_acc:.4f}" if isinstance(f_acc, (int, float)) else "-"
        m_s = f"{m_acc:.4f}" if isinstance(m_acc, (int, float)) else "-"
        do_s = f"{m_acc - o_acc:+.4f}" if isinstance(m_acc, (int, float)) and isinstance(o_acc, (int, float)) else "-"
        df_s = f"{m_acc - f_acc:+.4f}" if isinstance(m_acc, (int, float)) and isinstance(f_acc, (int, float)) else "-"
        print(
            f"{r['slug']:36s} | {r['primary_issue']:12s} | {o_s:>7s} | {f_s:>7s} | {m_s:>7s} | {do_s:>8s} | {df_s:>8s} | {m_st}"
        )

    print(f"\nDone {len(report_rows)} cases -> {out_root / 'benchmark_progress.json'}")


if __name__ == "__main__":
    main()
