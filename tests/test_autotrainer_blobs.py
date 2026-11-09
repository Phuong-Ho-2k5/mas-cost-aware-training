import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs

from mas_demo.contracts import TaskSpec
from mas_demo.data import prepare_data, load_datasets
from mas_demo.training.model import create_model, task_description
from mas_demo.training.worker import train_segment
from mas_demo.training.checkpoint import save_checkpoint


def test_blobs_recipe_and_official_split_are_preserved(tmp_path):
    spec = TaskSpec(dataset="blobs", data_dir=str(tmp_path))
    manifest = prepare_data(spec)
    train, val = load_datasets(spec, manifest)
    expected_x, expected_y = make_blobs(
        n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2
    )
    expected_x = ((expected_x * 100) / 100).astype(np.float32)
    assert len(train) == 700 and len(val) == 300
    assert manifest.train_indices == list(range(700))
    assert manifest.val_indices == list(range(700, 1000))
    for index in (0, 16, 699):
        np.testing.assert_array_equal(train[index][0].numpy(), expected_x[index])
        assert train[index][1].item() == expected_y[index]
    np.testing.assert_array_equal(val[0][0].numpy(), expected_x[700])
    assert manifest.origin == "generated_toy"
    assert prepare_data(spec).version == manifest.version


def test_mlp_matches_original_h5_probabilities_without_tensorflow():
    spec = TaskSpec(dataset="blobs")
    reference = json.loads(
        (
            Path(__file__).parent / "fixtures/autotrainer_blobs_reference.json"
        ).read_text()
    )
    model = create_model(spec).eval()
    assert sum(p.numel() for p in model.parameters()) == reference["parameters"] == 303
    with torch.no_grad():
        logits = model(torch.tensor(reference["input"]))
        actual = torch.softmax(logits, dim=1).numpy()
    np.testing.assert_allclose(actual, reference["probabilities"], rtol=2e-5, atol=1e-7)
    assert "BlobsMLP" in task_description(spec) and "3 classes" in task_description(
        spec
    )


def test_small_subsets_keep_validation_in_original_validation_partition(tmp_path):
    spec = TaskSpec(
        dataset="blobs", data_dir=str(tmp_path), train_samples=12, val_samples=9
    )
    manifest = prepare_data(spec)
    assert manifest.val_indices == list(range(700, 709))
    with pytest.raises(ValueError):
        TaskSpec(dataset="blobs", train_samples=701)


def test_blobs_training_resume_matches_continuous_and_stores_model_identity(tmp_path):
    spec = TaskSpec(
        dataset="blobs",
        data_dir=str(tmp_path / "data"),
        train_samples=24,
        val_samples=12,
        training={"batch_size": 1, "learning_rate": 0.01, "momentum": 0.9},
    )
    prepare_data(spec)
    whole = train_segment(spec, None, 2, "main", tmp_path / "whole", 120)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    resumed = train_segment(
        spec, first.checkpoint, 1, "main", tmp_path / "resumed", 120
    )
    assert whole.status == resumed.status == "COMPLETE"
    a = torch.load(whole.checkpoint.path, weights_only=False)
    b = torch.load(resumed.checkpoint.path, weights_only=False)
    assert a["model_signature"] == b["model_signature"]
    assert "BlobsMLP" in a["model_signature"]
    assert whole.events[-1].val_loss == resumed.events[-1].val_loss
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    b["model_signature"] = "SmallCNN"
    wrong = save_checkpoint(
        tmp_path / "wrong.pt", b, "main", resumed.checkpoint.dataset_version
    )
    with pytest.raises(ValueError, match="model identity mismatch"):
        train_segment(spec, wrong, 1, "main", tmp_path / "wrong-model", 120)
