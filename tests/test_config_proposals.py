import pytest
from pydantic import ValidationError

from mas_demo.contracts import (
    ConfigPatch,
    ConfigPlanResponse,
    ConfigProposal,
    ExperimentPolicy,
    TrainingConfig,
    TrainingPatch,
)
from mas_demo.configuration import (
    config_hash,
    compile_plan,
    compile_proposal,
    ProposalError,
)


def proposal(base, changes, name="p1"):
    return ConfigProposal(
        proposal_id=name,
        hypothesis_id="h1",
        patch=ConfigPatch(
            base_config_hash=config_hash(base),
            changes=changes,
            reason="Test the observed problem",
            evidence_epochs=[1, 2],
        ),
        expected_effect="Improve learning",
    )


def test_novel_absolute_values_and_multi_field_patch():
    base = TrainingConfig(learning_rate=0.1, weight_decay=0.5)
    result = compile_proposal(
        base, proposal(base, {"learning_rate": 0.027, "weight_decay": 0.0037})
    )
    assert result.training_config.learning_rate == 0.027
    assert result.training_config.weight_decay == 0.0037
    assert base.learning_rate == 0.1
    assert result.config_hash == config_hash(result.training_config)


def test_optimizer_switch_preserves_omitted_compatible_options():
    base = TrainingConfig(optimizer="adam", beta1=0.8, beta2=0.95, epsilon=1e-6)
    result = compile_proposal(base, proposal(base, {"optimizer": "adamw"}))
    assert result.diff == {"optimizer": {"before": "adam", "after": "adamw"}}
    assert result.training_config.beta1 == 0.8
    assert result.training_config.beta2 == 0.95
    assert result.training_config.epsilon == 1e-6


def test_null_means_disable_and_missing_preserves_clipping():
    base = TrainingConfig(grad_clip_norm=1.7)
    disabled = compile_proposal(base, proposal(base, {"grad_clip_norm": None}))
    kept = compile_proposal(base, proposal(base, {"learning_rate": 0.027}))
    assert disabled.training_config.grad_clip_norm is None
    assert kept.training_config.grad_clip_norm == 1.7


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"momentum": True},
        {"batch_size": "32"},
        {"weight_decay": -0.1},
        {"budget_seconds": 10000},
        {"unknown": 1},
        {"learning_rate": None},
    ],
)
def test_invalid_patch_is_never_partially_applied(changes):
    base = TrainingConfig()
    before = base.model_dump()
    with pytest.raises((ValidationError, ProposalError)):
        compile_proposal(base, proposal(base, changes))
    assert base.model_dump() == before


def test_stale_noop_and_duplicate_plans_are_rejected():
    base = TrainingConfig()
    with pytest.raises(ProposalError, match="NO_OP"):
        compile_proposal(base, proposal(base, {"learning_rate": base.learning_rate}))
    stale = proposal(base, {"learning_rate": 0.027})
    stale.patch.base_config_hash = "wrong"
    with pytest.raises(ProposalError, match="STALE"):
        compile_proposal(base, stale)
    first = proposal(base, {"learning_rate": 0.027})
    second = proposal(base, {"learning_rate": 0.027}, "p2")
    plan = ConfigPlanResponse(
        proposals=[first, second], verification_epochs=3, reason="Compare"
    )
    with pytest.raises(ProposalError, match="DUPLICATE"):
        compile_plan(base, plan, ExperimentPolicy(), [1, 2], 10, {"h1"})


def test_plan_has_one_llm_selected_horizon():
    base = TrainingConfig()
    plan = ConfigPlanResponse(
        proposals=[proposal(base, {"learning_rate": 0.027})],
        verification_epochs=3,
        reason="Observe three epochs",
    )
    compiled = compile_plan(base, plan, ExperimentPolicy(), [1, 2], 10, {"h1"})
    assert compiled.verification_epochs == 3
    assert compiled.candidates[0].training_config.learning_rate == 0.027
    with pytest.raises(ProposalError, match="HORIZON"):
        compile_plan(base, plan, ExperimentPolicy(), [1, 2], 2, {"h1"})
    for horizon in [0, 1.5, True, None]:
        with pytest.raises(ValidationError):
            ConfigPlanResponse(
                proposals=plan.proposals, verification_epochs=horizon, reason="Invalid"
            )
    assert (
        ConfigPlanResponse(
            proposals=[], verification_epochs=None, reason="Abstain"
        ).proposals
        == []
    )


@pytest.mark.parametrize(
    "base,changes",
    [
        (TrainingConfig(optimizer="adam"), {"momentum": 0.7}),
        (TrainingConfig(), {"beta1": 0.7}),
        (TrainingConfig(), {"scheduler_gamma": 0.7}),
    ],
)
def test_ignored_optimizer_or_scheduler_fields_are_not_exposed_as_effective_changes(
    base, changes
):
    with pytest.raises(ProposalError, match="INACTIVE_FIELD"):
        compile_proposal(base, proposal(base, changes))
