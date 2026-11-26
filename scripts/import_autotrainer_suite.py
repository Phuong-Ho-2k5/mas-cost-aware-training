"""Convert the stratified 100-case AUTOTRAINER benchmark from Keras HDF5 to PyTorch specs + .npz + TaskSpec YAML."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re

import h5py
import numpy as np
import yaml


DATASET_MAP = {
    "Blob": ("blobs", 700, 300),
    "Circle": ("circle", 700, 300),
    "MNIST": ("mnist", 60000, 10000),
    "CIFAR-10": ("cifar10", 50000, 10000),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_layer_tensors(weights_group, layer_name: str) -> dict[str, np.ndarray]:
    if layer_name not in weights_group:
        raise KeyError(f"Layer {layer_name} missing in model_weights")
    found = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            leaf = name.split("/")[-1]
            if leaf.startswith("kernel"):
                found["kernel"] = np.asarray(obj, dtype=np.float32)
            elif leaf.startswith("bias"):
                found["bias"] = np.asarray(obj, dtype=np.float32)

    weights_group[layer_name].visititems(visitor)
    return found


def convert_case(case: dict, output_dir: Path, max_epochs: int | None = None, budget_seconds: float = 900.0) -> dict:
    case_dir = Path(case["case_path"])
    h5_path = case_dir / "model.h5"
    cfg_path = case_dir / "training_config.pkl"

    with cfg_path.open("rb") as f:
        raw_cfg = pickle.load(f)

    with h5py.File(h5_path, "r") as hf:
        raw_mc = hf.attrs["model_config"]
        if isinstance(raw_mc, bytes):
            raw_mc = raw_mc.decode("utf-8")
        mc = json.loads(raw_mc)
        keras_layers = mc["config"]["layers"]
        mw = hf["model_weights"]

        spec_layers = []
        weights_out = {}

        # First pass: collect layers (skipping InputLayer)
        non_input = [l for l in keras_layers if l["class_name"] != "InputLayer"]
        last_idx = len(non_input) - 1

        for idx, layer in enumerate(non_input):
            cls = layer["class_name"]
            lcfg = layer["config"]
            lname = layer.get("name") or lcfg.get("name") or f"layer_{idx}"
            is_last = idx == last_idx

            if cls == "Dense":
                tensors = _find_layer_tensors(mw, lname)
                kernel = tensors["kernel"]
                in_f, out_f = int(kernel.shape[0]), int(kernel.shape[1])
                key = f"l{idx:02d}_{lname}"
                weights_out[f"{key}__weight"] = kernel.T.copy()
                if "bias" in tensors:
                    weights_out[f"{key}__bias"] = tensors["bias"].copy()
                act = lcfg.get("activation")
                # Strip final softmax (CrossEntropyLoss) or 1-unit final sigmoid (BCEWithLogitsLoss)
                if is_last and (act == "softmax" or (act == "sigmoid" and out_f == 1)):
                    act = None
                spec_layers.append(
                    {
                        "type": "Dense",
                        "name": lname,
                        "weight_key": key,
                        "in_features": in_f,
                        "out_features": out_f,
                        "use_bias": bool(lcfg.get("use_bias", True)),
                        "activation": act,
                    }
                )
            elif cls == "Conv2D":
                tensors = _find_layer_tensors(mw, lname)
                kernel = tensors["kernel"]  # Keras HWIO
                kH, kW, in_c, out_c = (int(x) for x in kernel.shape)
                key = f"l{idx:02d}_{lname}"
                weights_out[f"{key}__weight"] = np.transpose(kernel, (3, 2, 0, 1)).copy()
                if "bias" in tensors:
                    weights_out[f"{key}__bias"] = tensors["bias"].copy()
                pad = lcfg.get("padding", "valid")
                if pad != "valid":
                    raise ValueError(f"Unsupported Conv2D padding {pad} in {case['case_id']}")
                spec_layers.append(
                    {
                        "type": "Conv2D",
                        "name": lname,
                        "weight_key": key,
                        "in_channels": in_c,
                        "out_channels": out_c,
                        "kernel_size": [kH, kW],
                        "strides": list(lcfg.get("strides", [1, 1])),
                        "padding": 0,
                        "use_bias": bool(lcfg.get("use_bias", True)),
                        "activation": lcfg.get("activation"),
                    }
                )
            elif cls in ("MaxPooling2D", "AveragePooling2D"):
                pool_size = list(lcfg.get("pool_size", [2, 2]))
                strides = list(lcfg.get("strides") or pool_size)
                spec_layers.append(
                    {
                        "type": cls,
                        "name": lname,
                        "pool_size": pool_size,
                        "strides": strides,
                    }
                )
            elif cls == "Flatten":
                spec_layers.append({"type": "Flatten", "name": lname})
            elif cls == "Dropout":
                spec_layers.append({"type": "Dropout", "name": lname, "rate": float(lcfg.get("rate", 0.0))})
            elif cls == "Activation":
                act = lcfg.get("activation")
                # Strip trailing softmax or 1-unit binary sigmoid
                prev_out = next(
                    (l["out_features"] for l in reversed(spec_layers) if l["type"] == "Dense"),
                    None,
                )
                if is_last and (act == "softmax" or (act == "sigmoid" and prev_out == 1)):
                    continue
                spec_layers.append({"type": "Activation", "name": lname, "activation": act})
            else:
                raise ValueError(f"Unsupported layer {cls} in {case['case_id']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "model_spec.json"
    weights_path = output_dir / "initial_weights.npz"
    task_path = output_dir / "task.yaml"
    prov_path = output_dir / "provenance.json"

    ds_name, train_samples, val_samples = DATASET_MAP[case["dataset_dir"]]
    out_units = next(l["out_features"] for l in reversed(spec_layers) if l["type"] == "Dense")
    arch = case["model"]["architecture_type"]
    spec_payload = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "dataset": ds_name,
        "architecture_type": arch,
        "output_units": out_units,
        "description": f"{ds_name} classification / AUTOTRAINER {arch} ({len(spec_layers)} layers, {out_units} outputs)",
        "layers": spec_layers,
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(weights_path, **weights_out)

    # Map training_config.pkl -> MAS TaskSpec YAML (matching AUTOTRAINER config)
    opt = str(raw_cfg.get("optimizer", "sgd")).lower()
    opt_kwargs = raw_cfg.get("opt_kwargs") or {}
    lr = float(opt_kwargs.get("lr", opt_kwargs.get("learning_rate", 0.01)))
    if not np.isfinite(lr) or lr <= 0:
        lr = 0.01
    momentum = float(opt_kwargs.get("momentum", 0.0 if opt == "rmsprop" else 0.0))
    momentum = min(max(momentum, 0.0), 0.999)
    beta1 = float(opt_kwargs.get("beta_1", opt_kwargs.get("beta1", 0.9)))
    beta1 = min(max(beta1, 0.0), 0.999)
    beta2 = float(opt_kwargs.get("beta_2", opt_kwargs.get("beta2", 0.999)))
    beta2 = min(max(beta2, 0.0), 0.9999)
    epsilon = float(opt_kwargs.get("epsilon", 1e-7))
    if not np.isfinite(epsilon) or epsilon <= 0:
        epsilon = 1e-7
    batch_size = max(1, int(raw_cfg.get("batchsize", 16)))
    orig_epochs = max(1, int(raw_cfg.get("epoch", 50)))
    resolved_max_epochs = int(max_epochs) if max_epochs is not None else orig_epochs

    task_yaml = {
        "schema_version": 3,
        "dataset": ds_name,
        "data_dir": "data",
        "device": "cpu" if ds_name in ("blobs", "circle") else "auto",
        "seed": 11,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "observation_interval": 2,
        "window": 5,
        "analysis_mode": "vllm",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_spec_path": str(spec_path.resolve()),
        "initial_weights_path": str(weights_path.resolve()),
        "training": {
            "optimizer": opt,
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": 0.0,
            "grad_clip_norm": None,
            "grad_clip_value": None,
            "label_smoothing": 0.0,
            "batch_size": batch_size,
            "gradient_accumulation": 1,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "scheduler": "none",
        },
        "policy": {
            "max_epochs": resolved_max_epochs,
            "max_trials": 3,
            "max_verification_epochs": 10 if resolved_max_epochs >= 50 else 4,
            "budget_seconds": budget_seconds,
            "epoch_seconds_estimate": 1.0 if ds_name in ("blobs", "circle") else 8.0,
            "worker_overhead_seconds": 10.0,
            "safety_margin": 0.3,
            "min_gain": 0.005,
            "cooldown_rounds": 2,
        },
    }
    task_path.write_text(yaml.safe_dump(task_yaml, sort_keys=False), encoding="utf-8")

    provenance = {
        "case_id": case["case_id"],
        "dataset_dir": case["dataset_dir"],
        "status_dir": case["status_dir"],
        "primary_issue": case["primary_issue"],
        "initial_issue": case.get("initial_issue", []),
        "origin_best_acc": case.get("origin_best_acc"),
        "autotrainer_test_acc": case.get("autotrainer_test_acc"),
        "source_h5": str(h5_path),
        "source_h5_sha256": case["model_h5_sha256"],
        "source_config_sha256": case["config_pkl_sha256"],
        "spec_sha256": sha256_file(spec_path),
        "weights_sha256": sha256_file(weights_path),
        "task_yaml": str(task_path.resolve()),
    }
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")[:60]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/autotrainer_benchmark_100.json")
    parser.add_argument("--out-root", default=r"D:\HK261\autotrainer_raw\converted")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override original AUTOTRAINER epoch count")
    parser.add_argument("--budget-seconds", type=float, default=900.0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    converted = []
    for idx, case in enumerate(manifest["cases"], 1):
        slug = f"{idx:03d}_{slugify(case['dataset_dir'])}_{slugify(case['primary_issue'])}_{slugify(case['name'][-12:])}"
        prov = convert_case(case, out_root / slug, max_epochs=args.max_epochs, budget_seconds=args.budget_seconds)
        prov["slug"] = slug
        converted.append(prov)

    suite_path = out_root / "suite_manifest.json"
    suite_path.write_text(
        json.dumps({"schema_version": 1, "count": len(converted), "cases": converted}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Converted {len(converted)}/{manifest['count']} cases -> {suite_path}")


if __name__ == "__main__":
    main()
