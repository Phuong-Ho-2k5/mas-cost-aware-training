"""Audit AUTOTRAINER raw models/configs without importing Keras or TensorFlow."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import random

import h5py


SUPPORTED_DATASETS = {"Blob", "Circle", "MNIST", "CIFAR-10"}
SUPPORTED_LAYERS = {
    "Dense",
    "Conv2D",
    "MaxPooling2D",
    "AveragePooling2D",
    "Flatten",
    "BatchNormalization",
    "Activation",
    "Dropout",
    "InputLayer",
}
SUPPORTED_LOSSES = {
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "binary_crossentropy",
}
SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw", "rmsprop"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def inspect_h5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            return {"error": "missing_model_config"}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        cfg = json.loads(raw)
    model_class = cfg.get("class_name", "Unknown")
    layers = cfg.get("config", {}).get("layers", []) if isinstance(cfg.get("config"), dict) else []
    layer_types = [layer.get("class_name", "Unknown") for layer in layers]
    activations = []
    for layer in layers:
        lcfg = layer.get("config", {})
        act = lcfg.get("activation")
        if act:
            activations.append(act)
    layer_set = set(layer_types)
    if layer_set & {"LSTM", "GRU", "SimpleRNN", "Bidirectional", "Embedding"}:
        arch = "RNN"
    elif layer_set & {"Conv2D", "Conv1D", "SeparableConv2D", "DepthwiseConv2D"}:
        arch = "CNN"
    elif "Dense" in layer_set:
        arch = "MLP"
    else:
        arch = "OTHER"
    is_linear = model_class == "Sequential"
    if not is_linear and model_class in ("Model", "Functional") and layers:
        # Verify a single-input, single-output linear chain (each non-input layer consumes the previous layer).
        is_linear = True
        prev_name = None
        for idx, layer in enumerate(layers):
            lname = layer.get("name") or layer.get("config", {}).get("name")
            inbound = layer.get("inbound_nodes") or []
            if idx == 0:
                prev_name = lname
                continue
            if not inbound or len(inbound) != 1 or len(inbound[0]) != 1:
                is_linear = False
                break
            src = inbound[0][0][0]
            if src != prev_name:
                is_linear = False
                break
            prev_name = lname
    convertible = is_linear and bool(layer_types) and layer_set <= SUPPORTED_LAYERS
    return {
        "model_class": model_class,
        "is_linear_chain": is_linear,
        "architecture_type": arch,
        "num_layers": len(layers),
        "layer_types": layer_types,
        "activations": activations,
        "convertible_sequential": convertible,
        "unsupported_layers": sorted(layer_set - SUPPORTED_LAYERS),
    }


def inspect_case(case_dir: Path, dataset_dir: str, status_dir: str, compute_hash: bool = False) -> dict:
    h5_path = case_dir / "model.h5"
    cfg_path = case_dir / "training_config.pkl"
    record = {
        "case_id": f"{dataset_dir}/{status_dir}/{case_dir.name}",
        "dataset_dir": dataset_dir,
        "status_dir": status_dir,
        "name": case_dir.name,
        "case_path": str(case_dir),
        "has_h5": h5_path.exists(),
        "has_config": cfg_path.exists(),
    }
    if cfg_path.exists():
        try:
            with cfg_path.open("rb") as f:
                cfg = to_builtin(pickle.load(f))
            record["config"] = cfg
            record["raw_dataset"] = str(cfg.get("dataset", ""))
            record["optimizer"] = str(cfg.get("optimizer", "")).lower()
            record["loss"] = str(cfg.get("loss", "")).lower()
            record["batch_size"] = cfg.get("batchsize")
            record["epochs"] = cfg.get("epoch")
            opt_kwargs = cfg.get("opt_kwargs") or {}
            record["learning_rate"] = opt_kwargs.get("lr", opt_kwargs.get("learning_rate"))
        except Exception as err:
            record["config_error"] = f"{type(err).__name__}: {err}"
    if h5_path.exists():
        try:
            record["model"] = inspect_h5(h5_path)
        except Exception as err:
            record["model_error"] = f"{type(err).__name__}: {err}"
    result_dir = case_dir / "result"
    if result_dir.exists():
        at_path = result_dir / "autotrainer_result.pkl"
        orig_path = result_dir / "origin_result.pkl"
        if at_path.exists():
            try:
                at = pickle.load(at_path.open("rb"))
                record["initial_issue"] = to_builtin(at.get("initial_issue", []))
                record["now_issue"] = to_builtin(at.get("now_issue", []))
                record["autotrainer_test_acc"] = to_builtin(at.get("test_acc"))
            except Exception as err:
                record["at_result_error"] = str(err)
        if orig_path.exists():
            try:
                orig = pickle.load(orig_path.open("rb"))
                record["origin_best_acc"] = to_builtin(orig.get("best_acc"))
            except Exception as err:
                record["orig_result_error"] = str(err)
    if compute_hash:
        if h5_path.exists():
            record["model_h5_sha256"] = sha256_file(h5_path)
        if cfg_path.exists():
            record["config_pkl_sha256"] = sha256_file(cfg_path)
    return record


def primary_issue(record: dict) -> str:
    if record["status_dir"] == "normal":
        return "healthy"
    issues = record.get("initial_issue") or []
    if not issues:
        return "unknown_fault"
    return str(issues[0]).lower()


def select_stratified_100(records: list[dict], seed: int = 42, total: int = 100) -> list[dict]:
    """Select a stratified 100-case benchmark across supported datasets & Level-1-relevant faults."""
    # Filter to cleanly portable Sequential MLP/CNN on Blob, Circle, MNIST, CIFAR-10
    eligible = []
    for r in records:
        if r["dataset_dir"] not in SUPPORTED_DATASETS:
            continue
        if not (r.get("has_h5") and r.get("has_config")):
            continue
        m = r.get("model") or {}
        if not m.get("convertible_sequential"):
            continue
        if r.get("loss") not in SUPPORTED_LOSSES:
            continue
        if r.get("optimizer") not in SUPPORTED_OPTIMIZERS:
            continue
        # Exclude non-hyperparameter/data-preprocessing-only faults from the Level-1 benchmark
        issue = primary_issue(r)
        if issue in ("abnormal_data", "activation_issue", "loss_issue", "relu"):
            continue
        r = dict(r)
        r["primary_issue"] = issue
        eligible.append(r)

    rng = random.Random(seed)
    by_issue = defaultdict(list)
    for r in eligible:
        by_issue[r["primary_issue"]].append(r)
    for k in by_issue:
        by_issue[k].sort(key=lambda x: x["case_id"])
        rng.shuffle(by_issue[k])

    # Target quotas for Level-1 hyperparameter-fixable faults + 15 healthy controls
    quotas = {
        "healthy": 15,
        "unstable": 15,       # Oscillating Loss (takes all available up to 15)
        "explode": 25,        # Exploding Gradient
        "vanish": 25,         # Vanishing Gradient
        "not_converge": 35,   # Slow Convergence (also backfills to reach 100)
    }

    selected = []
    used_ids = set()

    def pick_balanced_by_dataset(pool: list[dict], count: int) -> list[dict]:
        by_ds = defaultdict(list)
        for item in pool:
            if item["case_id"] not in used_ids:
                by_ds[item["dataset_dir"]].append(item)
        picked = []
        ds_order = ["Blob", "Circle", "MNIST", "CIFAR-10"]
        while len(picked) < count:
            added = False
            for ds in ds_order:
                if by_ds[ds] and len(picked) < count:
                    item = by_ds[ds].pop(0)
                    picked.append(item)
                    used_ids.add(item["case_id"])
                    added = True
            if not added:
                break
        return picked

    for bucket, target in quotas.items():
        remaining_slots = total - len(selected)
        if remaining_slots <= 0:
            break
        selected.extend(pick_balanced_by_dataset(by_issue.get(bucket, []), min(target, remaining_slots)))

    # Fill any remaining slots up to `total` from remaining eligible fault cases (balanced by dataset)
    if len(selected) < total:
        remaining_faults = [r for r in eligible if r["case_id"] not in used_ids and r["primary_issue"] != "healthy"]
        rng.shuffle(remaining_faults)
        selected.extend(pick_balanced_by_dataset(remaining_faults, total - len(selected)))

    if len(selected) < total:
        remaining_any = [r for r in eligible if r["case_id"] not in used_ids]
        rng.shuffle(remaining_any)
        selected.extend(pick_balanced_by_dataset(remaining_any, total - len(selected)))

    # Add SHA-256 provenance for all selected cases
    for item in selected:
        cdir = Path(item["case_path"])
        item["model_h5_sha256"] = sha256_file(cdir / "model.h5")
        item["config_pkl_sha256"] = sha256_file(cdir / "training_config.pkl")

    selected.sort(key=lambda x: (x["dataset_dir"], x["primary_issue"], x["case_id"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=r"D:\HK261\autotrainer_raw\extracted\2022_rawdata")
    parser.add_argument("--catalog-out", default=r"D:\HK261\autotrainer_raw\catalog_all.json")
    parser.add_argument("--select-100", default=None, help="Optional output path for stratified 100-case manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.raw_dir)
    records = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for status_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for case_dir in sorted(p for p in status_dir.iterdir() if p.is_dir()):
                records.append(inspect_case(case_dir, ds_dir.name, status_dir.name))

    Path(args.catalog_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.catalog_out).write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Total cases audited: {len(records)}")
    print("By dataset:", dict(Counter(r["dataset_dir"] for r in records)))
    print("By arch:", dict(Counter((r.get("model") or {}).get("architecture_type", "ERR") for r in records)))
    print("Convertible sequential:", sum(1 for r in records if (r.get("model") or {}).get("convertible_sequential")))
    print("By primary issue (all):", dict(Counter(primary_issue(r) for r in records)))

    supported = [
        r for r in records
        if r["dataset_dir"] in SUPPORTED_DATASETS
        and (r.get("model") or {}).get("convertible_sequential")
        and r.get("loss") in SUPPORTED_LOSSES
        and r.get("optimizer") in SUPPORTED_OPTIMIZERS
    ]
    print(f"\nEligible convertible (Blob/Circle/MNIST/CIFAR-10): {len(supported)}")
    print("  By dataset:", dict(Counter(r["dataset_dir"] for r in supported)))
    print("  By issue:", dict(Counter(primary_issue(r) for r in supported)))
    print("  By optimizer:", dict(Counter(r.get("optimizer") for r in supported)))
    print("  By loss:", dict(Counter(r.get("loss") for r in supported)))

    if args.select_100:
        selected = select_stratified_100(records, seed=args.seed, total=100)
        manifest = {
            "schema_version": 1,
            "source_repo": "https://github.com/shiningrain/AUTOTRAINER",
            "raw_root": str(root),
            "selection_seed": args.seed,
            "count": len(selected),
            "breakdown_by_dataset": dict(Counter(r["dataset_dir"] for r in selected)),
            "breakdown_by_issue": dict(Counter(r["primary_issue"] for r in selected)),
            "breakdown_by_arch": dict(Counter(r["model"]["architecture_type"] for r in selected)),
            "cases": selected,
        }
        out = Path(args.select_100)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSelected {len(selected)} cases -> {out}")
        print("  Selected by dataset:", manifest["breakdown_by_dataset"])
        print("  Selected by issue:", manifest["breakdown_by_issue"])
        print("  Selected by arch:", manifest["breakdown_by_arch"])


if __name__ == "__main__":
    main()
