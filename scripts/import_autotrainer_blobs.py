"""Convert the pinned upstream HDF5's numeric tensors; never execute Keras models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "03a463ea05691a73403d7cff99a368fb4b6720cf"
H5_SHA256 = "059096a7e635729fd700b0d74268bfe22589f7a8ea693f17a6ca79953ab3bf75"


def convert(source, destination):
    import h5py  # Optional conversion tool; not needed to run the packaged demo.

    source, destination = Path(source), Path(destination)
    if destination.suffix != ".npz":
        raise ValueError("Output must have a .npz extension")
    if hashlib.sha256(source.read_bytes()).hexdigest() != H5_SHA256:
        raise ValueError("Input differs from the pinned AUTOTRAINER model.h5")
    if destination.exists():
        raise FileExistsError("Output already exists; choose another path")
    with h5py.File(source, "r") as model:
        values = {
            "hidden_weight": np.asarray(
                model["model_weights/dense_1/dense_1/kernel:0"]
            ).T.copy(),
            "hidden_bias": np.asarray(model["model_weights/dense_1/dense_1/bias:0"]),
            "output_weight": np.asarray(
                model["model_weights/dense_2/dense_2/kernel:0"]
            ).T.copy(),
            "output_bias": np.asarray(model["model_weights/dense_2/dense_2/bias:0"]),
        }
    for name, shape in {
        "hidden_weight": (50, 2),
        "hidden_bias": (50,),
        "output_weight": (3, 50),
        "output_bias": (3,),
    }.items():
        if (
            values[name].shape != shape
            or values[name].dtype != np.float32
            or not np.isfinite(values[name]).all()
        ):
            raise ValueError("Unexpected tensor format: " + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)
    provenance = {
        "source_repo": "https://github.com/shiningrain/AUTOTRAINER",
        "source_commit": UPSTREAM_COMMIT,
        "source_model": "AutoTrainer/demo_case/Oscillating_Loss_Case/model.h5",
        "source_h5_sha256": H5_SHA256,
        "asset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "architecture": "2 -> Dense(50, ReLU) -> Dense(3, softmax)",
        "torch_output": "Logits; CrossEntropyLoss includes log-softmax with integer class targets.",
        "original_config": {
            "dataset": "simplednn_blob",
            "optimizer": "SGD",
            "opt_kwargs": {"lr": 0.01, "momentum": 0.9},
            "batchsize": 1,
            "epoch": 200,
            "loss": "categorical_crossentropy",
        },
        "port_limitations": "Preserves data recipe, initial tensors and settings; PyTorch RNG/order and optimizer transitions are not a bitwise Keras reproduction.",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))
