import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Subset
from mas_demo.contracts import DatasetManifest
from mas_demo.io import write_json


def _blobs_manifest(spec):
    import numpy as np
    import sklearn

    train_indices = list(range(spec.train_samples))
    val_indices = list(range(700, 700 + spec.val_samples))
    recipe = "sklearn.make_blobs:n1000:centers3:features2:std2:random_state2:first700-split:times100-div100:float32-v1"
    generator_version = f"scikit-learn={sklearn.__version__};numpy={np.__version__}"
    identity = dict(
        dataset="blobs",
        seed=spec.seed,
        train=train_indices,
        val=val_indices,
        transform=recipe,
        generator_version=generator_version,
    )
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return DatasetManifest(
        dataset="blobs",
        seed=spec.seed,
        train_indices=train_indices,
        val_indices=val_indices,
        origin="generated_toy",
        version=version,
        data_recipe=recipe,
        generator_version=generator_version,
    )


def _circle_manifest(spec):
    import numpy as np
    import sklearn

    train_indices = list(range(spec.train_samples))
    val_indices = list(range(700, 700 + spec.val_samples))
    recipe = "sklearn.make_circles:n1000:noise0.1:random_state1:first700-split:times100-div100:float32-v1"
    generator_version = f"scikit-learn={sklearn.__version__};numpy={np.__version__}"
    identity = dict(
        dataset="circle",
        seed=spec.seed,
        train=train_indices,
        val=val_indices,
        transform=recipe,
        generator_version=generator_version,
    )
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return DatasetManifest(
        dataset="circle",
        seed=spec.seed,
        train_indices=train_indices,
        val_indices=val_indices,
        origin="generated_toy",
        version=version,
        data_recipe=recipe,
        generator_version=generator_version,
    )


def prepare_data(spec):
    """Download once in the parent; workers load an immutable manifest."""
    root = Path(spec.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if spec.dataset == "blobs":
        manifest = _blobs_manifest(spec)
        write_json(root / f"manifest-{manifest.version}.json", manifest)
        return manifest
    if spec.dataset == "circle":
        manifest = _circle_manifest(spec)
        write_json(root / f"manifest-{manifest.version}.json", manifest)
        return manifest
    if spec.dataset == "mnist":
        from torchvision.datasets import MNIST

        MNIST(str(root), train=True, download=True)
        if spec.train_samples == 60000 and spec.val_samples == 10000:
            MNIST(str(root), train=False, download=True)
            train_indices = list(range(60000))
            val_indices = list(range(10000))
            origin = "real_full"
            transform_id = "to-tensor-0-1-official-split-v1"
        else:
            order = torch.randperm(
                60000, generator=torch.Generator().manual_seed(spec.seed)
            ).tolist()
            train_indices = order[:55000][: spec.train_samples]
            val_indices = order[55000:][: spec.val_samples]
            origin = (
                "real_full"
                if len(train_indices) == 55000 and len(val_indices) == 5000
                else "real_subset"
            )
            transform_id = "to-tensor-0-1-v1"
    elif spec.dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        CIFAR10(str(root), train=True, download=True)
        if spec.train_samples == 50000 and spec.val_samples == 10000:
            CIFAR10(str(root), train=False, download=True)
            train_indices = list(range(50000))
            val_indices = list(range(10000))
            origin = "real_full"
            transform_id = "autotrainer-cifar10-official-split-v1"
        else:
            order = torch.randperm(
                50000, generator=torch.Generator().manual_seed(spec.seed)
            ).tolist()
            train_indices = order[:45000][: spec.train_samples]
            val_indices = order[45000:][: spec.val_samples]
            origin = (
                "real_full"
                if len(train_indices) == 45000 and len(val_indices) == 5000
                else "real_subset"
            )
            transform_id = "normalize-0.5-v1"
    else:
        train_indices = list(range(spec.train_samples))
        val_indices = list(
            range(spec.train_samples, spec.train_samples + spec.val_samples)
        )
        origin = "synthetic"
        transform_id = "normalize-0.5-v1"
    identity = {
        "dataset": spec.dataset,
        "seed": spec.seed,
        "train": train_indices,
        "val": val_indices,
        "transform": transform_id,
    }
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = DatasetManifest(
        dataset=spec.dataset,
        seed=spec.seed,
        train_indices=train_indices,
        val_indices=val_indices,
        origin=origin,
        version=version,
    )
    write_json(root / f"manifest-{version}.json", result)
    return result


def get_manifest(spec):
    # No network in a worker. Same deterministic identity as preprocessing.
    if spec.dataset in ("blobs", "circle"):
        expected = _blobs_manifest(spec) if spec.dataset == "blobs" else _circle_manifest(spec)
        path = Path(spec.data_dir) / f"manifest-{expected.version}.json"
        if not path.exists():
            raise ValueError(
                f"{spec.dataset} manifest missing or generator version changed; prepare a new run"
            )
        actual = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"{spec.dataset} manifest does not match the pinned data recipe/split"
            )
        return actual
    candidates = Path(spec.data_dir).glob("manifest-*.json")
    for path in candidates:
        m = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            m.dataset == spec.dataset
            and m.seed == spec.seed
            and len(m.train_indices) == spec.train_samples
            and len(m.val_indices) == spec.val_samples
        ):
            return m
    raise ValueError("Dataset manifest missing; run preprocessing first")


def load_datasets(spec, manifest):
    if spec.dataset == "blobs":
        import numpy as np
        from sklearn.datasets import make_blobs

        x, y = make_blobs(
            n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2
        )
        # Preserve upstream's X*=100 followed by preprocess(X)=X/100.
        x = ((x * 100) / 100).astype(np.float32)
        data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    elif spec.dataset == "circle":
        import numpy as np
        from sklearn.datasets import make_circles

        x, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
        x = ((x * 100) / 100).astype(np.float32)
        data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    elif spec.dataset == "mnist":
        from torchvision import datasets, transforms

        if spec.train_samples == 60000 and spec.val_samples == 10000:
            train_ds = datasets.MNIST(
                spec.data_dir, train=True, download=False, transform=transforms.ToTensor()
            )
            val_ds = datasets.MNIST(
                spec.data_dir, train=False, download=False, transform=transforms.ToTensor()
            )
            return Subset(train_ds, manifest.train_indices), Subset(val_ds, manifest.val_indices)
        data = datasets.MNIST(
            spec.data_dir, train=True, download=False, transform=transforms.ToTensor()
        )
    elif spec.dataset == "cifar10":
        from torchvision import datasets, transforms

        if spec.train_samples == 50000 and spec.val_samples == 10000:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (125.307 / 255.0, 122.95 / 255.0, 113.865 / 255.0),
                        (62.9932 / 255.0, 62.0887 / 255.0, 66.7048 / 255.0),
                    ),
                ]
            )
            train_ds = datasets.CIFAR10(
                spec.data_dir, train=True, download=False, transform=transform
            )
            val_ds = datasets.CIFAR10(
                spec.data_dir, train=False, download=False, transform=transform
            )
            return Subset(train_ds, manifest.train_indices), Subset(val_ds, manifest.val_indices)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        data = datasets.CIFAR10(
            spec.data_dir, train=True, download=False, transform=transform
        )
    else:
        # Explicit synthetic smoke data: real optimization, not research evidence.
        generator = torch.Generator().manual_seed(spec.seed + 123)
        size = spec.train_samples + spec.val_samples
        y = torch.arange(size) % 10
        x = torch.randn(size, 3, 32, 32, generator=generator) * 0.05
        for index in range(size):
            label = int(y[index])
            x[index, label % 3, (label // 3) * 6 : (label // 3) * 6 + 6, :] += 1
        data = TensorDataset(x, y)
    return Subset(data, manifest.train_indices), Subset(data, manifest.val_indices)
