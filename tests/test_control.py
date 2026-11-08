from mas_demo.control.budget import BudgetLedger
from mas_demo.contracts import EpochEvent, EvidenceWindow, TrialSpec
from mas_demo.observation.collector import build_evidence
from mas_demo.agents.planner import eligible_trials
from mas_demo.memory.store import MemoryStore


def event(epoch, loss=1.0, acc=0.4, branch="main"):
    return EpochEvent(epoch=epoch, branch_id=branch, train_loss=loss,
                      val_loss=loss, val_accuracy=acc, learning_rate=0.1, grad_norm=1)


def test_budget_reservations_cannot_double_spend():
    ledger = BudgetLedger(100)
    assert ledger.reserve("control", 40)
    assert ledger.reserve("A", 30)
    assert not ledger.reserve("B", 40)
    assert not ledger.reserve("A", 30)
    ledger.settle("A", 25)
    assert ledger.available == 35 and ledger.consumed == 25
    ledger.settle("control", 40)
    assert ledger.available == 35


def test_missing_metric_is_not_plateau_and_branches_are_isolated():
    e = build_evidence([event(1), event(2, branch="other")], "main", window=2)
    assert e.epochs == [1]
    assert e.status == "INSUFFICIENT"


def test_loss_spike_detector_uses_real_loss_history():
    e = build_evidence([event(1, 1), event(2, 0.9), event(3, 3)], "main", window=2)
    assert "loss_spike" in e.signals


def test_memory_cooldown_filters_executed_but_not_unexecuted_trials(tmp_path):
    memory = MemoryStore(tmp_path / "memory.sqlite")
    context = "cifar10:smallcnn:loss_spike"
    memory.record(context, 0.5, "REJECT", -0.1, 10, 1)
    memory.record(context, 0.1, "NOT_EXECUTED", 0, 0, 1)
    history = memory.retrieve(context, before_round=2)
    trials = eligible_trials([0.5, 0.1], history, 2, 2)
    assert [t.lr_factor for t in trials] == [0.1]
    assert memory.retrieve(context, before_round=1) == []


def test_executed_failed_trial_is_not_retried_during_cooldown(tmp_path):
    memory = MemoryStore(tmp_path/"memory.sqlite")
    memory.record("context", 0.5, "FAILED", 0, 100, 1)
    memory.record("context", 0.1, "NOT_EXECUTED", 0, 0, 1)
    trials = eligible_trials([0.5, 0.1], memory.retrieve("context", 2), 2, 1)
    assert [t.lr_factor for t in trials] == [0.1]
