from pathlib import Path
import pytest
from mas_demo.contracts import TaskSpec, TrialSpec
from mas_demo.data import prepare_data
from mas_demo.training.worker import train_segment
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_trials
from mas_demo.control.verification import verify_round
from mas_demo.orchestration import run


class InProcessBackend:
    def run(self, spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor=None, intervention=None):
        return train_segment(spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor, intervention=intervention)


def test_admission_counts_control_once_and_preserves_continuation_budget():
    trials = [TrialSpec(trial_id="A", lr_factor=0.5, horizon=2),
              TrialSpec(trial_id="B", lr_factor=0.1, horizon=2)]
    ledger = BudgetLedger(65)
    admission = admit_trials(trials, ledger, epoch_seconds=10, overhead=0, margin=0, continuation=10, round_id=1)
    assert [a.trial_id for a in admission.approved] == ["A"]
    assert admission.skipped == ["B"]
    assert ledger.available == 15  # control 20 + trial 20 + continuation 10


def test_real_multi_trial_round_has_shared_fork_and_same_epoch(tmp_path):
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=32, val_samples=16, batch_size=8, max_epochs=4)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"initial", 120)
    trials = [TrialSpec(trial_id="A", lr_factor=0.5, horizon=1),
              TrialSpec(trial_id="B", lr_factor=0.1, horizon=1)]
    ledger = BudgetLedger(900)
    admission = admit_trials(trials, ledger, 1, 10, 0, 0, 1)
    result = verify_round(spec, first.checkpoint, admission, ledger,
                          InProcessBackend(), tmp_path/"round")
    assert result.comparison_epoch == 2
    assert result.control.status == "COMPLETE"
    assert set(result.trials) == {"A", "B"}
    for branch in [result.control, *result.trials.values()]:
        assert branch.checkpoint.parent_sha256 == first.checkpoint.sha256
        assert branch.events[-1].epoch == 2
    assert ledger.consumed > 0 and ledger.available > 0


def test_invalid_checkpoint_does_not_run_any_trial(tmp_path):
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=16, val_samples=8)
    prepare_data(spec)
    first = train_segment(spec, None, 1, "main", tmp_path/"first", 120)
    Path(first.checkpoint.path).write_bytes(b"corrupt")
    ledger = BudgetLedger(100)
    admission = admit_trials([TrialSpec(trial_id="A", lr_factor=0.5, horizon=1)], ledger, 1, 1, 0, 0, 1)
    with pytest.raises(ValueError, match="hash"):
        verify_round(spec, first.checkpoint, admission, ledger, InProcessBackend(), tmp_path/"round")


def test_orchestrator_limits_epochs_and_resume_is_noop_after_completion(tmp_path, provider):
    gateway, _ = provider
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=32, val_samples=16, batch_size=8, max_epochs=3,
                    observation_interval=1, budget_seconds=120,
                    epoch_seconds_estimate=0.1, worker_overhead_seconds=2)
    directory = tmp_path/"run"
    result = run(spec, directory, backend=InProcessBackend(), gateway=gateway)
    assert result["status"] == "COMPLETED" and result["epoch"] == 3
    assert result["gpu_hours"] is None and result["resource_type"] == "cpu"
    before = (directory/"events.jsonl").read_bytes()
    resumed = run(spec, directory, resume=True, backend=InProcessBackend(), gateway=gateway)
    assert resumed["epoch"] == 3
    assert (directory/"events.jsonl").read_bytes() == before


def test_real_anomalous_loop_evaluates_then_continues_from_selected_checkpoint(tmp_path, provider):
    gateway, _ = provider
    import json
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=128, val_samples=32, batch_size=16, max_epochs=6,
                    observation_interval=2, trial_horizon=1, fault_epoch=3, fault_lr=1,
                    budget_seconds=180, epoch_seconds_estimate=2, worker_overhead_seconds=10)
    directory = tmp_path/"loop"
    result = run(spec, directory, backend=InProcessBackend(), gateway=gateway)
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    evaluation_index = next(i for i, e in enumerate(events) if e["kind"] == "evaluation")
    evaluation = events[evaluation_index]
    assert evaluation["verdict"]["decision"] in ("PROMOTE", "KEEP_MAIN")
    selected = evaluation["result"]["trials"][evaluation["verdict"]["selected_branch"]] if evaluation["verdict"]["decision"] == "PROMOTE" else evaluation["result"]["control"]
    continuation = next(e for e in events[evaluation_index+1:] if e["kind"] == "segment")
    assert continuation["result"]["checkpoint"]["parent_sha256"] == selected["checkpoint"]["sha256"]
    assert result["status"] == "COMPLETED" and result["epoch"] == 6


def test_pause_before_training_resumes_without_resetting_budget(tmp_path, provider):
    gateway, _ = provider
    spec = TaskSpec(dataset="synthetic", device="cpu", data_dir=str(tmp_path/"data"),
                    train_samples=16, val_samples=8, max_epochs=2,
                    epoch_seconds_estimate=1, worker_overhead_seconds=2)
    directory = tmp_path/"paused"
    directory.mkdir()
    (directory/"stop.flag").touch()
    paused = run(spec, directory, backend=InProcessBackend(), gateway=gateway)
    assert paused["status"] == "PAUSED" and paused["epoch"] == 0
    (directory/"stop.flag").unlink()
    finished = run(spec, directory, resume=True, backend=InProcessBackend(), gateway=gateway)
    assert finished["status"] == "COMPLETED" and finished["budget"]["consumed"] > 0
