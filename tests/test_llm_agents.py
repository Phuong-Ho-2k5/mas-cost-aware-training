import pytest
from mas_demo.contracts import TaskSpec, EpochEvent, ExperimentPolicy
from mas_demo.agents.evaluation import evaluate_with_agent


def metric(epoch=4, accuracy=0.5, loss=1.0):
    return EpochEvent(
        epoch=epoch,
        branch_id="main",
        train_loss=1,
        val_loss=loss,
        val_accuracy=accuracy,
        learning_rate=0.1,
        grad_norm=1,
    )


def test_evaluation_uses_model_choice_and_computes_gain(provider):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE",
        "selected_branch": "A",
        "reason": "Observed improvement",
    }
    result, metadata = evaluate_with_agent(
        TaskSpec(),
        [metric()],
        {"A": [metric(accuracy=0.6, loss=0.9)], "B": [metric(accuracy=0.7, loss=0.8)]},
        4,
        gateway=gateway,
    )
    assert result.selected_branch == "A" and result.gain == pytest.approx(0.1)


@pytest.mark.parametrize(
    "epoch,accuracy,loss", [(3, 0.7, 0.8), (4, 0.7, 1.1), (4, 0.501, 0.8)]
)
def test_evaluation_cannot_promote_ineligible_or_wrong_epoch(
    provider, epoch, accuracy, loss
):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE",
        "selected_branch": "A",
        "reason": "Promote",
    }
    result, metadata = evaluate_with_agent(
        TaskSpec(),
        [metric()],
        {"A": [metric(epoch, accuracy, loss)]},
        4,
        gateway=gateway,
    )
    assert result.decision == "INCONCLUSIVE" and metadata["error"]


def test_evaluation_compares_exact_epoch_not_latest(provider):
    gateway, state = provider
    result, _ = evaluate_with_agent(
        TaskSpec(),
        [metric(), metric(5, 0.99, 0.1)],
        {"A": [metric(4, 0.7, 0.8)]},
        4,
        gateway=gateway,
    )
    assert result.decision == "PROMOTE"


def test_incomplete_control_cannot_be_promoted(provider):
    gateway, state = provider
    state["responses"]["training_evaluation"] = {
        "decision": "PROMOTE",
        "selected_branch": "A",
        "reason": "Promote",
    }
    result, metadata = evaluate_with_agent(
        TaskSpec(), [], {"A": [metric(4, 0.7, 0.8)]}, 4, gateway=gateway
    )
    assert result.decision == "INCONCLUSIVE"
