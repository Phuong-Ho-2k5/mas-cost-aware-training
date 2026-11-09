import pytest
from mas_demo.contracts import (
    TrainingConfig,
    ExperimentPolicy,
    ConfigPlanResponse,
    EpochEvent,
)
from mas_demo.configuration import compile_plan, ProposalError
from mas_demo.control.budget import BudgetLedger
from mas_demo.control.controller import admit_plan
from mas_demo.observation.collector import build_evidence
from test_config_proposals import proposal


def test_budget_reservations_cannot_double_spend():
    ledger = BudgetLedger(100)
    assert ledger.reserve("control", 40)
    assert ledger.reserve("trial", 30)
    assert not ledger.reserve("other", 40)
    assert not ledger.reserve("trial", 30)
    ledger.settle("trial", 25)
    assert ledger.available == 35


def test_admission_is_all_or_nothing_and_does_not_shorten_horizon():
    base = TrainingConfig()
    response = ConfigPlanResponse(
        proposals=[
            proposal(base, {"learning_rate": 0.027}),
            proposal(base, {"learning_rate": 0.043}, "p2"),
        ],
        verification_epochs=2,
        reason="Compare",
    )
    plan = compile_plan(base, response, ExperimentPolicy(), [1, 2], 10, {"h1"})
    ledger = BudgetLedger(65)
    with pytest.raises(ProposalError, match="INSUFFICIENT_BUDGET"):
        admit_plan(plan, ledger, 10, 0, 0, 10, 1)
    assert ledger.reservations == {}
    assert plan.verification_epochs == 2 and len(plan.candidates) == 2
    ledger = BudgetLedger(100)
    admission = admit_plan(plan, ledger, 10, 0, 0, 10, 1)
    assert admission.horizon == 2 and len(admission.approved) == 2
    assert ledger.available == 30


def event(epoch, loss=1, branch="main"):
    return EpochEvent(
        epoch=epoch,
        branch_id=branch,
        train_loss=loss,
        val_loss=loss,
        val_accuracy=0.5,
        learning_rate=0.1,
        grad_norm=1,
    )


def test_collector_preserves_window_without_classifying_anomalies():
    records = [event(1), event(2, 3), event(3), event(4, 3), event(5), event(6, 9)]
    evidence = build_evidence(records, "main", 5)
    assert evidence.epochs == [2, 3, 4, 5, 6]
    assert evidence.signals == []


def test_collector_isolates_branches_and_does_not_require_full_window():
    evidence = build_evidence([event(1), event(2, branch="trial")], "main", 5)
    assert evidence.status == "INSUFFICIENT" and evidence.epochs == [1]
