"""Export AUTOTRAINER vs MAS pilot/benchmark results with clickable links to Excel (.xlsx)."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_autotrainer_benchmark import select_pilot_10  # noqa: E402


AUTOTRAINER_REPO_URL = "https://github.com/shiningrain/AUTOTRAINER"


def _fault_source(case_id: str) -> str:
    name = case_id.split("/")[-1]
    if name.startswith("config_random"):
        return "config_random (Hyperparam)"
    if name.startswith("double_random"):
        return "double_random (Model+Hyperparam)"
    if name.startswith("model_"):
        return "model (Architecture)"
    return "normal (Healthy)"


def _extract_at_solution(raw_dir: Path) -> str:
    pkl = raw_dir / "result" / "autotrainer_result.pkl"
    if not pkl.exists():
        return "N/A (no repair)"
    try:
        obj = pickle.load(pkl.open("rb"))
        mp = str(obj.get("model_path", ""))
    except Exception:
        return "N/A"
    steps = re.findall(r"(?:vanish|explode|not_converge|unstable|oscillating|relu)_([a-z0-9]+)_\d+", mp.lower())
    if steps:
        return " -> ".join(dict.fromkeys(steps))
    return Path(mp).parent.name if mp else "N/A"


def _extract_mas_promotions(events_path: Path) -> tuple[int, str]:
    if not events_path.exists():
        return 0, ""
    promotions = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "evaluation":
            v = r.get("verdict") or {}
            if v.get("decision") == "PROMOTE":
                rnd = r.get("round")
                branch = v.get("selected_branch")
                gain = v.get("gain")
                # Pull diff from metadata candidate_configs or trial training_config
                diff = None
                trials = (r.get("result") or {}).get("trials") or {}
                ctrl = (r.get("result") or {}).get("control") or {}
                t_cfg = (trials.get(branch) or {}).get("training_config") or {}
                c_cfg = ctrl.get("training_config") or {}
                if t_cfg and c_cfg:
                    diff = {k: f"{c_cfg.get(k)}->{t_cfg.get(k)}" for k in t_cfg if t_cfg.get(k) != c_cfg.get(k)}
                gain_str = f"+{gain*100:.2f}%" if isinstance(gain, (int, float)) else ""
                diff_str = ", ".join(f"{k}:{val}" for k, val in (diff or {}).items())
                promotions.append(f"R{rnd}({diff_str} {gain_str})".strip())
    return len(promotions), " | ".join(promotions) if promotions else "None"


def _summarize_initial_config(task: dict, spec: dict) -> str:
    t = task.get("training", {})
    layers = spec.get("layers", [])
    param_idx = [i for i, e in enumerate(layers) if e["type"] in ("Dense", "Conv2D")]
    last_idx = param_idx[-1] if param_idx else len(layers) - 1
    acts = {}
    for i, e in enumerate(layers):
        if i < last_idx and e.get("activation") not in (None, "linear"):
            a = e["activation"]
            acts[a] = acts.get(a, 0) + 1
    act_str = ", ".join(f"{v}x{k}" for k, v in acts.items()) if acts else "linear"
    return f"opt={t.get('optimizer')}, lr={t.get('learning_rate')}, bs={t.get('batch_size')}, acts=[{act_str}]"


def _summarize_final_config(active: dict) -> str:
    if not active:
        return ""
    parts = [
        f"opt={active.get('optimizer')}",
        f"lr={active.get('learning_rate')}",
        f"bs={active.get('batch_size')}",
    ]
    if active.get("hidden_activation", "keep") != "keep":
        parts.append(f"act={active.get('hidden_activation')}")
    if active.get("use_batchnorm"):
        parts.append("bn=True")
    if active.get("weight_init", "keep") != "keep":
        parts.append(f"init={active.get('weight_init')}")
    if active.get("weight_decay", 0.0):
        parts.append(f"wd={active.get('weight_decay')}")
    if active.get("grad_clip_norm") is not None:
        parts.append(f"clip={active.get('grad_clip_norm')}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default="artifacts/autotrainer_bench/pilot_v1")
    parser.add_argument("--suite", default=r"D:\HK261\autotrainer_raw\converted\suite_manifest.json")
    parser.add_argument("--pilot", action="store_true", default=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else bench_dir / f"{bench_dir.name}_summary.xlsx"
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    cases = select_pilot_10(suite["cases"]) if args.pilot else suite["cases"]

    wb = Workbook()
    ws = wb.active
    ws.title = "Pilot_Results"

    headers = [
        "STT",
        "Case_Slug",
        "AUTOTRAINER_Case_ID",
        "Dataset",
        "Fault_Source",
        "Primary_Issue",
        "Initial_Config",
        "AT_Orig_Acc",
        "AT_Fix_Acc",
        "AT_Solution",
        "MAS_Best_Acc",
        "MAS_vs_AT_Orig",
        "MAS_vs_AT_Fix",
        "Promotions",
        "MAS_Final_Config",
        "Epochs_Done",
        "Max_Epochs",
        "Rounds",
        "MAS_Status",
        "Worker_Sec",
        "LLM_Sec",
        "Link_MAS_Artifacts",
        "Link_AUTOTRAINER_Raw",
        "Link_AUTOTRAINER_Repo",
    ]

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    link_font = Font(color="0563C1", underline="single")
    pos_fill = PatternFill("solid", fgColor="E2EFDA")
    neg_fill = PatternFill("solid", fgColor="FCE4D6")

    ws.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 28

    rows_data = []
    for idx, case in enumerate(cases, 1):
        slug = case["slug"]
        mas_dir = bench_dir / slug / "mas"
        st_path = mas_dir / "state.json"
        ev_path = mas_dir / "events.jsonl"
        conv_dir = Path(case["task_yaml"]).parent
        task = yaml.safe_load((conv_dir / "task.yaml").read_text(encoding="utf-8"))
        spec = json.loads((conv_dir / "model_spec.json").read_text(encoding="utf-8"))
        raw_dir = Path(case["source_h5"]).parent

        state = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {}
        at_orig = case.get("origin_best_acc")
        at_fix = case.get("autotrainer_test_acc")
        mas_acc = state.get("best_accuracy")
        if mas_acc is not None and mas_acc < 0:
            mas_acc = None
        vs_orig = (mas_acc - at_orig) if (mas_acc is not None and at_orig is not None) else None
        vs_fix = (mas_acc - at_fix) if (mas_acc is not None and at_fix is not None) else None
        n_prom, prom_str = _extract_mas_promotions(ev_path)

        row = [
            idx,
            slug,
            case["case_id"],
            case["dataset_dir"],
            _fault_source(case["case_id"]),
            case["primary_issue"],
            _summarize_initial_config(task, spec),
            at_orig,
            at_fix,
            _extract_at_solution(raw_dir),
            mas_acc,
            vs_orig,
            vs_fix,
            n_prom,
            _summarize_final_config(state.get("active_config") or {}),
            state.get("epoch"),
            task.get("policy", {}).get("max_epochs"),
            state.get("round"),
            state.get("status", "NOT_STARTED"),
            round(state.get("budget", {}).get("consumed", 0.0), 1) if state else None,
            round(state.get("llm_seconds", 0.0), 1) if state else None,
            str(mas_dir),
            str(raw_dir),
            AUTOTRAINER_REPO_URL,
        ]
        ws.append(row)
        r_idx = idx + 1

        # Percent formatting
        for col_i in (8, 9, 11, 12, 13):
            c = ws.cell(row=r_idx, column=col_i)
            if c.value is not None:
                c.number_format = "+0.00%;-0.00%;0.00%" if col_i in (12, 13) else "0.00%"
        if vs_fix is not None:
            ws.cell(row=r_idx, column=13).fill = pos_fill if vs_fix >= 0 else neg_fill

        # Clickable links
        c_mas = ws.cell(row=r_idx, column=22)
        c_mas.hyperlink = mas_dir.as_uri()
        c_mas.value = f"Open {slug}/mas"
        c_mas.font = link_font

        c_raw = ws.cell(row=r_idx, column=23)
        c_raw.hyperlink = raw_dir.as_uri()
        c_raw.value = f"Open AUTOTRAINER/{case['case_id']}"
        c_raw.font = link_font

        c_repo = ws.cell(row=r_idx, column=24)
        c_repo.hyperlink = AUTOTRAINER_REPO_URL
        c_repo.value = "shiningrain/AUTOTRAINER"
        c_repo.font = link_font

        rows_data.append({
            "issue": case["primary_issue"],
            "at_orig": at_orig,
            "at_fix": at_fix,
            "mas_acc": mas_acc,
            "vs_orig": vs_orig,
            "vs_fix": vs_fix,
        })

    # Auto-fit column widths (bounded)
    for col_idx in range(1, len(headers) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(1, len(cases) + 2))
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 48)
    ws.freeze_panes = "C2"

    # Sheet 2: Summary
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Metric / Group", "Cases", "Avg AT_Orig", "Avg AT_Fix", "Avg MAS (Lv2)", "Avg Gain vs Orig", "Avg Delta vs AT_Fix", "MAS >= AT_Fix"])
    for col_idx in range(1, 9):
        c = ws2.cell(row=1, column=col_idx)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    def _add_summary_row(label: str, subset: list[dict]):
        origs = [x["at_orig"] for x in subset if x["at_orig"] is not None]
        fixes = [x["at_fix"] for x in subset if x["at_fix"] is not None]
        mass = [x["mas_acc"] for x in subset if x["mas_acc"] is not None]
        v_orig = [x["vs_orig"] for x in subset if x["vs_orig"] is not None]
        v_fix = [x["vs_fix"] for x in subset if x["vs_fix"] is not None]
        wins = sum(1 for x in subset if x["vs_fix"] is not None and x["vs_fix"] >= 0)
        ws2.append([
            label,
            len(subset),
            sum(origs) / len(origs) if origs else None,
            sum(fixes) / len(fixes) if fixes else None,
            sum(mass) / len(mass) if mass else None,
            sum(v_orig) / len(v_orig) if v_orig else None,
            sum(v_fix) / len(v_fix) if v_fix else None,
            f"{wins}/{len(v_fix)}" if v_fix else "N/A",
        ])
        r_i = ws2.max_row
        for col_i in (3, 4, 5, 6, 7):
            c = ws2.cell(row=r_i, column=col_i)
            if c.value is not None:
                c.number_format = "+0.00%;-0.00%;0.00%" if col_i in (6, 7) else "0.00%"

    fault_rows = [x for x in rows_data if x["issue"] != "healthy"]
    _add_summary_row("ALL FAULT CASES (8)", fault_rows)
    for issue in ("explode", "vanish", "not_converge", "unstable", "healthy"):
        _add_summary_row(f"Issue: {issue}", [x for x in rows_data if x["issue"] == issue])
    _add_summary_row("ALL 10 PILOT CASES", rows_data)

    for col_idx in range(1, 9):
        col_letter = get_column_letter(col_idx)
        max_len = max(len(str(ws2.cell(row=r, column=col_idx).value or "")) for r in range(1, ws2.max_row + 1))
        ws2.column_dimensions[col_letter].width = max(max_len + 4, 16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"Saved Excel summary -> {out_path}")


if __name__ == "__main__":
    main()
