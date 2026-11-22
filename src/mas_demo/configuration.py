"""Validate and compile LLM plans, never manufacture actions or horizons."""

import hashlib
import json
from pydantic import ValidationError
from mas_demo.contracts import (
    CompiledPlan,
    CompiledProposal,
    TrainingConfig,
    TrainingPatch,
)


class ProposalError(ValueError):
    def __init__(self, code, message, **details):
        self.code, self.details = code, details
        super().__init__(f"{code}: {message}")

    def as_dict(self):
        return {"code": self.code, "reason": str(self), **self.details}


def config_hash(config):
    value = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def training_capabilities():
    return {
        "supported_fields": list(TrainingPatch.model_fields),
        "constraints": "Use field types and bounds in the response JSON schema; all numbers must be finite.",
        "semantics": "Absolute settings, applied together at an epoch boundary.",
        "null_disables": ["grad_clip_norm", "early_stop_patience"],
        "component_options": {
            "sgd": ["momentum"],
            "adam_or_adamw": ["beta1", "beta2", "epsilon"],
            "step": ["scheduler_step_size", "scheduler_gamma"],
            "cosine": ["scheduler_t_max"],
        },
        "optimizer_change": "Weights retained; incompatible optimizer state reset.",
        "scheduler_change": "Scheduler reset when its type or configuration changes.",
        "comparison": "Complete epochs; all branches use the same model-chosen horizon.",
    }


def compile_proposal(current, proposal, capabilities=None):
    if proposal.patch.base_config_hash != config_hash(current):
        raise ProposalError(
            "STALE_CONFIG", "Proposal base differs from active configuration"
        )
    changes = proposal.patch.changes.model_dump(exclude_unset=True)
    if not changes:
        raise ProposalError("NO_OP", "Patch is empty")
    values = current.model_dump()
    values.update(changes)
    try:
        candidate = TrainingConfig.model_validate(values)
    except ValidationError as error:
        raise ProposalError("INVALID_CONFIG", str(error)) from error
    inactive = (
        {"beta1", "beta2", "epsilon"} if candidate.optimizer == "sgd" else {"momentum"}
    )
    if candidate.scheduler != "step":
        inactive |= {"scheduler_step_size", "scheduler_gamma"}
    if candidate.scheduler != "cosine":
        inactive.add("scheduler_t_max")
    if inactive & changes.keys():
        raise ProposalError(
            "INACTIVE_FIELD",
            "Patch changes options not used by its selected components",
            fields=sorted(inactive & changes.keys()),
        )
    before, after = current.model_dump(), candidate.model_dump()
    diff = {
        key: {"before": before[key], "after": value}
        for key, value in after.items()
        if value != before[key]
    }
    if not diff:
        raise ProposalError(
            "NO_OP", "Candidate is identical to the active configuration"
        )
    return CompiledProposal(
        trial_id="",
        proposal_id=proposal.proposal_id,
        hypothesis_id=proposal.hypothesis_id,
        base_config_hash=config_hash(current),
        training_config=candidate,
        config_hash=config_hash(candidate),
        diff=diff,
        evidence_epochs=proposal.patch.evidence_epochs,
    )


def compile_plan(
    current, response, policy, evidence_epochs, remaining_epochs, hypothesis_ranks
):
    if not response.proposals:
        if hypothesis_ranks:
            raise ProposalError("HYPOTHESIS_COVERAGE", "A diagnosed set requires one trial per ranked hypothesis")
        reason = response.reason.strip()
        if not reason or (reason.startswith("<") and reason.endswith(">")):
            raise ProposalError(
                "INVALID_ABSTENTION_REASON",
                "State a concrete blocker from the inputs, not a placeholder; otherwise provide the complete trial plan",
            )
        return CompiledPlan(candidates=[], verification_epochs=None)
    if (not isinstance(hypothesis_ranks, dict) or not 1 <= len(hypothesis_ranks) <= 3
        or any(type(rank) is not int for rank in hypothesis_ranks.values())
        or set(hypothesis_ranks.values()) != set(range(1, len(hypothesis_ranks) + 1))):
        raise ProposalError("INVALID_HYPOTHESIS_RANK", "Expected one to three hypotheses with distinct Analysis ranks 1..N")
    if hypothesis_ranks is not None:
        required = set(hypothesis_ranks)
        assigned = [proposal.hypothesis_id for proposal in response.proposals]
        unknown = set(assigned) - required
        if unknown:
            raise ProposalError(
                "UNKNOWN_HYPOTHESIS", "Unavailable hypothesis", hypothesis_ids=sorted(unknown)
            )
        if len(assigned) != len(required) or set(assigned) != required:
            raise ProposalError(
                "HYPOTHESIS_COVERAGE",
                "Provide exactly one distinct trial per retained hypothesis",
                required_hypothesis_ids=sorted(required),
                received_hypothesis_ids=assigned,
                missing_hypothesis_ids=sorted(required - set(assigned)),
            )
    if not 1 <= response.verification_epochs <= remaining_epochs:
        raise ProposalError(
            "INVALID_HORIZON",
            "Shared epoch count exceeds remaining epochs",
            remaining_epochs=remaining_epochs,
        )
    candidates, seen_ids, config_groups = [], set(), {}
    for index, proposal in enumerate(response.proposals, 1):
        if proposal.proposal_id in seen_ids:
            raise ProposalError("DUPLICATE_ID", "Repeated proposal identifier")
        if not proposal.patch.evidence_epochs or not set(
            proposal.patch.evidence_epochs
        ) <= set(evidence_epochs):
            raise ProposalError("UNAVAILABLE_EVIDENCE", "Unavailable evidence epochs")
        compiled = compile_proposal(current, proposal)
        compiled.hypothesis_rank = hypothesis_ranks[proposal.hypothesis_id]
        group = config_groups.setdefault(compiled.config_hash, dict(
            config_hash=compiled.config_hash, proposal_ids=[], hypothesis_ids=[], patches=[]))
        group["proposal_ids"].append(proposal.proposal_id)
        group["hypothesis_ids"].append(proposal.hypothesis_id)
        group["patches"].append(proposal.patch.changes.model_dump(exclude_unset=True))
        compiled.trial_id = f"trial-{index}"
        candidates.append(compiled)
        seen_ids.add(proposal.proposal_id)
    duplicates = [group for group in config_groups.values() if len(group["proposal_ids"]) > 1]
    if duplicates:
        raise ProposalError("DUPLICATE_CONFIG", "Proposals produce the same configuration",
            duplicate_groups=duplicates,
            repair_instruction="Rebuild each colliding trial from its own hypothesis mechanism; use a minimal relevant patch, not arbitrary values just to make hashes different.")
    return CompiledPlan(
        candidates=candidates, verification_epochs=response.verification_epochs
    )
