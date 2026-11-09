"""Admission validates the complete model-selected plan without changing it."""

import math
from mas_demo.contracts import AdmissionBatch
from mas_demo.configuration import ProposalError


def estimate_epoch_cost(base, candidate, epoch_estimate, train_samples):
    # Conservative relative work estimate; measurements update the base each segment.
    old_batches = math.ceil(train_samples / base.batch_size)
    new_batches = math.ceil(train_samples / candidate.batch_size)
    factor = max(1.0, new_batches / max(old_batches, 1))
    if candidate.optimizer != base.optimizer:
        factor *= 2
    return epoch_estimate * factor


def admit_plan(
    compiled_plan,
    ledger,
    epoch_estimate,
    overhead,
    margin,
    continuation,
    round_id,
    current_config=None,
    train_samples=1,
):
    if not compiled_plan.candidates or compiled_plan.verification_epochs is None:
        raise ProposalError("EMPTY_PLAN", "There are no candidates to admit")
    horizon = compiled_plan.verification_epochs
    control_key = f"round-{round_id}:control"
    costs = {control_key: (epoch_estimate * horizon + overhead) * (1 + margin)}
    keys = {}
    for candidate in compiled_plan.candidates:
        estimate = (
            estimate_epoch_cost(
                current_config, candidate.training_config, epoch_estimate, train_samples
            )
            if current_config
            else epoch_estimate
        )
        key = f"round-{round_id}:{candidate.trial_id}"
        keys[candidate.trial_id] = key
        costs[key] = (estimate * horizon + overhead) * (1 + margin)
    continuation_key = f"round-{round_id}:continuation" if continuation > 0 else None
    if continuation_key:
        costs[continuation_key] = continuation
    total = sum(costs.values())
    if total > ledger.available:
        raise ProposalError(
            "INSUFFICIENT_BUDGET",
            "Entire plan cannot be reserved",
            estimated_total_seconds=total,
            remaining_budget_seconds=ledger.available,
        )
    reserved = []
    try:
        for key, cost in costs.items():
            if not ledger.reserve(key, cost):
                raise ProposalError("RESERVATION_FAILED", "Plan admission failed")
            reserved.append(key)
    except Exception:
        for key in reserved:
            ledger.release(key)
        raise
    return AdmissionBatch(
        round_id=round_id,
        horizon=horizon,
        control_key=control_key,
        approved=compiled_plan.candidates,
        reservation_keys=keys,
        continuation_key=continuation_key,
    )
