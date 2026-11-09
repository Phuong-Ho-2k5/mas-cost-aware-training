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
    current, response, policy, evidence_epochs, remaining_epochs, hypothesis_ids=None
):
    if len(response.proposals) > policy.max_trials:
        raise ProposalError(
            "MAX_TRIALS", "Plan exceeds trial limit", limit=policy.max_trials
        )
    if not response.proposals:
        return CompiledPlan(candidates=[], verification_epochs=None)
    if not 1 <= response.verification_epochs <= remaining_epochs:
        raise ProposalError(
            "INVALID_HORIZON",
            "Shared epoch count exceeds remaining epochs",
            remaining_epochs=remaining_epochs,
        )
    candidates, seen_ids, seen_hashes = [], set(), set()
    for index, proposal in enumerate(response.proposals, 1):
        if proposal.proposal_id in seen_ids:
            raise ProposalError("DUPLICATE_ID", "Repeated proposal identifier")
        if hypothesis_ids is not None and proposal.hypothesis_id not in hypothesis_ids:
            raise ProposalError("UNKNOWN_HYPOTHESIS", "Unavailable hypothesis")
        if not proposal.patch.evidence_epochs or not set(
            proposal.patch.evidence_epochs
        ) <= set(evidence_epochs):
            raise ProposalError("UNAVAILABLE_EVIDENCE", "Unavailable evidence epochs")
        compiled = compile_proposal(current, proposal)
        if compiled.config_hash in seen_hashes:
            raise ProposalError(
                "DUPLICATE_CONFIG", "Proposals produce the same configuration"
            )
        compiled.trial_id = f"trial-{index}"
        candidates.append(compiled)
        seen_ids.add(proposal.proposal_id)
        seen_hashes.add(compiled.config_hash)
    return CompiledPlan(
        candidates=candidates, verification_epochs=response.verification_epochs
    )
