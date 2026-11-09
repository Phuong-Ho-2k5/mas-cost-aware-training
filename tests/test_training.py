import torch
from mas_demo.contracts import TaskSpec, TrainingConfig
from mas_demo.data import prepare_data, load_datasets
from mas_demo.training.worker import train_segment


def tiny(tmp_path):
    return TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        training=TrainingConfig(batch_size=4),
    )


def test_manifest_split_is_disjoint_and_reproducible(tmp_path):
    spec = tiny(tmp_path)
    a, b = prepare_data(spec), prepare_data(spec)
    assert a == b and not set(a.train_indices) & set(a.val_indices)
    x, y = load_datasets(spec, a)
    assert len(x) == 16 and len(y) == 8


def test_budget_exhausted_does_not_train(tmp_path):
    spec = tiny(tmp_path)
    prepare_data(spec)
    result = train_segment(spec, None, 2, "main", tmp_path / "empty", 0)
    assert result.status == "BUDGET_EXHAUSTED"
    assert not result.events and result.checkpoint is None


def test_changed_seed_cannot_resume_checkpoint(tmp_path):
    import pytest

    spec = tiny(tmp_path)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    spec.seed = 22
    prepare_data(spec)
    with pytest.raises(ValueError, match="dataset|seed"):
        train_segment(spec, first.checkpoint, 1, "main", tmp_path / "wrong", 120)
