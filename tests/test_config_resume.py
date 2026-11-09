import torch
import pytest
from mas_demo.contracts import TaskSpec, TrainingConfig, ExperimentPolicy
from mas_demo.configuration import config_hash
from mas_demo.data import prepare_data
from mas_demo.training.worker import train_segment


def tiny(tmp_path, **training):
    return TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=20,
        val_samples=8,
        training=TrainingConfig(batch_size=4, **training),
        policy=ExperimentPolicy(max_epochs=8, budget_seconds=120),
    )


@pytest.mark.parametrize("lr", [0.027, 0.2])
def test_candidate_config_applied_after_restore_and_kept_on_resume(tmp_path, lr):
    spec = tiny(tmp_path, learning_rate=0.1, grad_clip_norm=1.7)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    candidate = spec.training.model_copy(
        update={"learning_rate": lr, "weight_decay": 0.0037, "grad_clip_norm": None}
    )
    trial = train_segment(
        spec,
        first.checkpoint,
        1,
        "trial",
        tmp_path / "trial",
        120,
        training_config=candidate,
    )
    continued = train_segment(
        spec, trial.checkpoint, 1, "main", tmp_path / "continued", 120
    )
    assert trial.events[-1].learning_rate == lr
    assert continued.events[-1].learning_rate == lr
    assert continued.training_config.grad_clip_norm is None
    assert trial.checkpoint.config_hash == config_hash(candidate)
    saved = torch.load(trial.checkpoint.path, weights_only=False)
    assert saved["training_config"] == candidate.model_dump()


def test_optimizer_switch_records_reset_and_batch_accumulation_changes(tmp_path):
    spec = tiny(tmp_path)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    config = spec.training.model_copy(
        update={"optimizer": "adamw", "batch_size": 3, "gradient_accumulation": 2}
    )
    result = train_segment(
        spec,
        first.checkpoint,
        1,
        "trial",
        tmp_path / "trial",
        120,
        training_config=config,
    )
    assert result.status == "COMPLETE"
    assert result.transition["optimizer_reset"] is True
    assert result.events[-1].samples_seen == 40
    assert result.events[-1].optimizer_steps == 9  # five old + ceil(7 new batches / 2)


@pytest.mark.parametrize("scheduler", ["none", "step", "cosine"])
def test_full_state_resume_matches_continuous_training(tmp_path, scheduler):
    spec = tiny(
        tmp_path,
        scheduler=scheduler,
        scheduler_step_size=1,
        scheduler_t_max=8,
        gradient_accumulation=2,
    )
    prepare_data(spec)
    whole = train_segment(spec, None, 2, "main", tmp_path / "whole", 120)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    resumed = train_segment(
        spec, first.checkpoint, 1, "main", tmp_path / "resumed", 120
    )
    a = torch.load(whole.checkpoint.path, weights_only=False)
    b = torch.load(resumed.checkpoint.path, weights_only=False)
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    assert a["training_config"] == b["training_config"]
    assert a["optimizer_steps"] == b["optimizer_steps"]


def test_cosine_minimum_learning_rate_checkpoint_can_resume(tmp_path):
    spec = tiny(tmp_path, scheduler="cosine", scheduler_t_max=1)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    result = train_segment(
        spec, first.checkpoint, 1, "main", tmp_path / "continued", 120
    )
    assert result.status == "COMPLETE"


def test_replacement_scheduler_uses_candidate_not_old_initial_lr(tmp_path):
    spec = tiny(tmp_path, learning_rate=0.1, scheduler="step")
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    candidate = first.training_config.model_copy(
        update={"learning_rate": 0.027, "scheduler": "cosine", "scheduler_t_max": 1}
    )
    changed = train_segment(
        spec,
        first.checkpoint,
        3,
        "trial",
        tmp_path / "trial",
        120,
        training_config=candidate,
    )
    assert [event.learning_rate for event in changed.events] == pytest.approx(
        [0.027, 0.0, 0.027]
    )
