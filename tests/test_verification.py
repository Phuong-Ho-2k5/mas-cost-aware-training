from pathlib import Path
import pytest
from mas_demo.contracts import (
    TaskSpec,
    TrainingConfig,
    ExperimentPolicy,
    ConfigPlanResponse,
)
from mas_demo.configuration import compile_plan
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_plan
from mas_demo.control.verification import verify_round
from mas_demo.training.worker import train_segment
from mas_demo.data import prepare_data
from test_config_proposals import proposal


class InProcessBackend:
    def run(
        self,
        spec,
        start,
        epochs,
        branch_id,
        output_dir,
        budget_seconds,
        training_config=None,
    ):
        return train_segment(
            spec, start, epochs, branch_id, output_dir, budget_seconds, training_config
        )


def test_real_multi_trial_round_has_shared_fork_and_comparison_epoch(tmp_path):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
        training=TrainingConfig(batch_size=4),
        policy=ExperimentPolicy(max_epochs=6),
    )
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "initial", 120)
    response = ConfigPlanResponse(
        proposals=[
            proposal(first.training_config, {"learning_rate": 0.027}),
            proposal(first.training_config, {"learning_rate": 0.043}, "p2"),
        ],
        verification_epochs=2,
        reason="Compare",
    )
    compiled = compile_plan(
        first.training_config, response, spec.policy, [1, 2], 5, {"h1"}
    )
    ledger = BudgetLedger(900)
    admission = admit_plan(compiled, ledger, 1, 10, 0, 0, 1)
    result = verify_round(
        spec,
        first.checkpoint,
        admission,
        ledger,
        InProcessBackend(),
        tmp_path / "round",
        first.training_config,
    )
    for part in [result.control, *result.trials.values()]:
        assert part.status == "COMPLETE"
        assert part.checkpoint.parent_sha256 == first.checkpoint.sha256
        assert part.checkpoint.epoch == 3
    assert result.comparison_epoch == 3
    assert ledger.snapshot()["reserved"] == 0


def test_corrupt_checkpoint_fails_before_any_trial(tmp_path):
    spec = TaskSpec(
        dataset="synthetic",
        data_dir=str(tmp_path / "data"),
        train_samples=16,
        val_samples=8,
    )
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path / "initial", 120)
    response = ConfigPlanResponse(
        proposals=[proposal(first.training_config, {"learning_rate": 0.027})],
        verification_epochs=1,
        reason="Compare",
    )
    compiled = compile_plan(
        first.training_config, response, spec.policy, [1, 2], 5, {"h1"}
    )
    ledger = BudgetLedger(100)
    admission = admit_plan(compiled, ledger, 1, 1, 0, 0, 1)
    Path(first.checkpoint.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        verify_round(
            spec,
            first.checkpoint,
            admission,
            ledger,
            InProcessBackend(),
            tmp_path / "round",
            first.training_config,
        )
