import torch

from mas_demo.contracts import TaskSpec
from mas_demo.data import prepare_data, load_datasets
from mas_demo.training.worker import train_segment


def tiny(tmp_path, **kwargs):
    return TaskSpec(dataset="synthetic", train_samples=32, val_samples=16,
                    batch_size=8, device="cpu", data_dir=str(tmp_path / "data"), **kwargs)


def test_manifest_split_is_disjoint_and_reproducible(tmp_path):
    spec = tiny(tmp_path)
    a = prepare_data(spec)
    b = prepare_data(spec)
    assert a == b
    assert not set(a.train_indices) & set(a.val_indices)
    x, y = load_datasets(spec, a)
    assert len(x) == 32 and len(y) == 16


def test_full_state_resume_matches_continuous_training(tmp_path):
    spec = tiny(tmp_path)
    prepare_data(spec)
    whole = train_segment(spec, None, 2, "whole", tmp_path / "whole", 120)
    first = train_segment(spec, None, 1, "first", tmp_path / "first", 120)
    resumed = train_segment(spec, first.checkpoint, 1, "resumed", tmp_path / "resumed", 120)
    a = torch.load(whole.checkpoint.path, weights_only=False)
    b = torch.load(resumed.checkpoint.path, weights_only=False)
    assert whole.events[-1].val_loss == resumed.events[-1].val_loss
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert torch.equal(a["generator_state"], b["generator_state"])


def test_treatment_changes_lr_after_restore_without_changing_parent(tmp_path):
    spec = tiny(tmp_path)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "main", 120)
    branch = train_segment(spec, first.checkpoint, 1, "trial", tmp_path / "trial", 120, lr_factor=0.1)
    assert branch.events[-1].learning_rate == spec.learning_rate * 0.1
    assert torch.load(first.checkpoint.path, weights_only=False)["epoch"] == 1


def test_budget_exhausted_does_not_train(tmp_path):
    spec = tiny(tmp_path)
    prepare_data(spec)
    result = train_segment(spec, None, 2, "empty", tmp_path / "empty", 0)
    assert result.status == "BUDGET_EXHAUSTED"
    assert not result.events and result.checkpoint is None
