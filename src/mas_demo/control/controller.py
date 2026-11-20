"""Estimate and admit trials in Analysis priority order, within a shared ledger."""

import math
from decimal import Decimal
from mas_demo.contracts import AdmissionBatch, PlanCostQuote, SkippedTrial, TrialCostQuote
from mas_demo.configuration import ProposalError


def _sum_seconds(values):
    return float(sum((Decimal(str(value)) for value in values), Decimal(0)))


def _branch_seconds(estimate, horizon, overhead, margin):
    return float((Decimal(str(estimate)) * horizon + Decimal(str(overhead)))
                 * (Decimal(1) + Decimal(str(margin))))


def estimate_epoch_cost(base, candidate, epoch_estimate, train_samples):
    old_batches = math.ceil(train_samples / base.batch_size)
    new_batches = math.ceil(train_samples / candidate.batch_size)
    factor = max(1.0, new_batches / max(old_batches, 1))
    if candidate.optimizer != base.optimizer:
        factor *= 2
    return epoch_estimate * factor


def quote_plan_cost(compiled_plan, epoch_estimate, overhead, margin, remaining_epochs,
                    *, current_config, train_samples):
    if not compiled_plan.candidates or compiled_plan.verification_epochs is None:
        raise ProposalError("EMPTY_PLAN", "There are no candidates to admit")
    horizon = compiled_plan.verification_epochs
    if (not all(math.isfinite(x) for x in (epoch_estimate, overhead, margin))
        or epoch_estimate <= 0 or overhead < 0 or margin < 0
        or type(train_samples) is not int or train_samples <= 0
        or type(remaining_epochs) is not int or not 1 <= horizon <= remaining_epochs):
        raise ProposalError("INVALID_COST_INPUT", "Invalid cost estimate or horizon")
    ranks = [c.hypothesis_rank for c in compiled_plan.candidates]
    if (any(type(r) is not int or not 1 <= r <= 3 for r in ranks)
        or len(ranks) != len(set(ranks))):
        raise ProposalError("INVALID_HYPOTHESIS_RANK", "Candidates need unique Analysis ranks")
    ids = [c.trial_id for c in compiled_plan.candidates]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ProposalError("DUPLICATE_ID", "Candidates need distinct trial IDs")
    control = _branch_seconds(epoch_estimate, horizon, overhead, margin)
    continuation = (_branch_seconds(epoch_estimate, 1, overhead, margin)
                    if horizon < remaining_epochs else 0)
    costs = {}
    for candidate in compiled_plan.candidates:
        estimate = estimate_epoch_cost(current_config, candidate.training_config,
                                       epoch_estimate, train_samples)
        costs[candidate.trial_id] = TrialCostQuote(
            trial_id=candidate.trial_id, proposal_id=candidate.proposal_id,
            hypothesis_id=candidate.hypothesis_id, hypothesis_rank=candidate.hypothesis_rank,
            epoch_seconds=estimate, reserved_seconds=_branch_seconds(estimate, horizon, overhead, margin))
    return PlanCostQuote(horizon=horizon, control_seconds=control,
                         continuation_seconds=continuation, trial_costs=costs,
                         all_trials_total_seconds=_sum_seconds(
                             [control, continuation, *(c.reserved_seconds for c in costs.values())]))


def admit_plan(compiled_plan, ledger, epoch_estimate, overhead, margin, remaining_epochs,
               round_id, *, current_config, train_samples, max_trials, cooldown_reasons=None):
    if type(max_trials) is not int or max_trials < 0:
        raise ProposalError("INVALID_POLICY", "Trial execution limit must be nonnegative")
    quote = quote_plan_cost(compiled_plan, epoch_estimate, overhead, margin, remaining_epochs,
                            current_config=current_config, train_samples=train_samples)
    available = ledger.available
    approved, skipped = [], []
    total = 0
    blocked = cooldown_reasons or {}
    for candidate in sorted(compiled_plan.candidates, key=lambda c: c.hypothesis_rank):
        cost = quote.trial_costs[candidate.trial_id].reserved_seconds
        next_total = _sum_seconds([total, cost] if approved
                                 else [quote.control_seconds, quote.continuation_seconds, cost])
        if candidate.trial_id in blocked:
            code, reason = "SKIPPED_COOLDOWN", blocked[candidate.trial_id]
        elif len(approved) >= max_trials:
            code, reason = "SKIPPED_POLICY", "Trial execution limit reached"
        elif next_total > available:
            code, reason = "SKIPPED_BUDGET", f"Group requires {next_total:g}s; available {available:g}s"
        else:
            approved.append(candidate)
            total = next_total
            continue
        skipped.append(SkippedTrial(
            trial_id=candidate.trial_id, proposal_id=candidate.proposal_id,
            hypothesis_id=candidate.hypothesis_id, hypothesis_rank=candidate.hypothesis_rank,
            estimated_seconds=cost, code=code, reason=reason))
    control_key = f"round-{round_id}:control" if approved else None
    continuation_key = (f"round-{round_id}:continuation"
                        if approved and quote.continuation_seconds > 0 else None)
    keys = {c.trial_id: f"round-{round_id}:{c.trial_id}" for c in approved}
    reservations = {control_key: quote.control_seconds} if approved else {}
    reservations.update({keys[c.trial_id]: quote.trial_costs[c.trial_id].reserved_seconds for c in approved})
    if continuation_key:
        reservations[continuation_key] = quote.continuation_seconds
    reserved = []
    try:
        for key, cost in reservations.items():
            if not ledger.reserve(key, cost):
                raise ProposalError("RESERVATION_FAILED", "Plan admission failed")
            reserved.append(key)
    except Exception:
        for key in reserved:
            ledger.release(key)
        raise
    return AdmissionBatch(round_id=round_id, horizon=quote.horizon, control_key=control_key,
                          approved=approved, reservation_keys=keys, continuation_key=continuation_key,
                          skipped=skipped, cost_quote=quote, budget_available_before=available,
                          reserved_total_seconds=total)
