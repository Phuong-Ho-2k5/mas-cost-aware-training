"""Former fixed interventions are now typed, atomic configuration patches."""

import pytest
from mas_demo.contracts import TrainingConfig, TaskSpec, ExperimentPolicy
from mas_demo.configuration import compile_proposal, config_hash
from mas_demo.data import prepare_data
from mas_demo.training.worker import train_segment
from test_config_proposals import proposal


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
def test_optimizer_capability_runs_and_survives_resume(tmp_path, optimizer):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        training=TrainingConfig(optimizer=optimizer, batch_size=4),
    )
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    resumed = train_segment(
        spec, first.checkpoint, 1, "main", tmp_path / "resumed", 120
    )
    assert resumed.status == "COMPLETE"
    assert resumed.training_config.optimizer == optimizer
    assert resumed.transition["optimizer_reset"] is False


def test_scheduler_change_records_reset_and_new_config(tmp_path):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
    )
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "first", 120)
    config = first.training_config.model_copy(
        update={"scheduler": "step", "scheduler_step_size": 1}
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
    assert result.transition["scheduler_reset"]
    assert result.training_config.learning_rate == pytest.approx(
        config.learning_rate * config.scheduler_gamma
    )
