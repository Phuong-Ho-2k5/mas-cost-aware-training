import os
from pathlib import Path
import torch
from mas_demo.contracts import CheckpointRef
from mas_demo.io import sha256


def validate_checkpoint(ref, dataset_version=None):
    if sha256(ref.path) != ref.sha256:
        raise ValueError("Checkpoint hash mismatch")
    if dataset_version is not None and ref.dataset_version != dataset_version:
        raise ValueError("Checkpoint dataset mismatch")


def save_checkpoint(path, state, branch_id, dataset_version, parent=None):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)
    return CheckpointRef(path=str(path), sha256=sha256(path), epoch=state["epoch"],
                         branch_id=branch_id, dataset_version=dataset_version,
                         parent_sha256=parent.sha256 if parent else None)
