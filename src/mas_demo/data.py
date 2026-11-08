import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Subset
from mas_demo.contracts import DatasetManifest
from mas_demo.io import write_json


def prepare_data(spec):
    """Download once in the parent; workers load an immutable manifest."""
    root = Path(spec.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if spec.dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        CIFAR10(str(root), train=True, download=True)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(spec.seed)).tolist()
        train_indices = order[:45000][:spec.train_samples]
        val_indices = order[45000:][:spec.val_samples]
        origin = "real_full" if len(train_indices) == 45000 and len(val_indices) == 5000 else "real_subset"
    else:
        train_indices = list(range(spec.train_samples))
        val_indices = list(range(spec.train_samples, spec.train_samples + spec.val_samples))
        origin = "synthetic"
    identity = {"dataset": spec.dataset, "seed": spec.seed, "train": train_indices,
                "val": val_indices, "transform": "normalize-0.5-v1"}
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = DatasetManifest(dataset=spec.dataset, seed=spec.seed, train_indices=train_indices,
                             val_indices=val_indices, origin=origin, version=version)
    write_json(root / f"manifest-{version}.json", result)
    return result


def get_manifest(spec):
    # No network in a worker. Same deterministic identity as preprocessing.
    candidates = Path(spec.data_dir).glob("manifest-*.json")
    for path in candidates:
        m = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if (m.dataset == spec.dataset and m.seed == spec.seed and
                len(m.train_indices) == spec.train_samples and len(m.val_indices) == spec.val_samples):
            return m
    raise ValueError("Dataset manifest missing; run preprocessing first")


def load_datasets(spec, manifest):
    if spec.dataset == "cifar10":
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        data = datasets.CIFAR10(spec.data_dir, train=True, download=False, transform=transform)
    else:
        # Explicit synthetic smoke data: real optimization, not research evidence.
        generator = torch.Generator().manual_seed(spec.seed + 123)
        size = spec.train_samples + spec.val_samples
        y = torch.arange(size) % 10
        x = torch.randn(size, 3, 32, 32, generator=generator) * 0.05
        for index in range(size):
            label = int(y[index])
            x[index, label % 3, (label // 3) * 6:(label // 3) * 6 + 6, :] += 1
        data = TensorDataset(x, y)
    return Subset(data, manifest.train_indices), Subset(data, manifest.val_indices)
