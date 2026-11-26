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
    value = dict(config.model_dump(mode="json") if hasattr(config, "model_dump") else config)
    # Preserve historical v3 config hashes when Level-1/Level-2 optional fields stay at inert defaults.
    if value.get("grad_clip_value") is None:
        value.pop("grad_clip_value", None)
    if value.get("label_smoothing", 0.0) == 0.0:
        value.pop("label_smoothing", None)
    if value.get("scheduler") != "plateau" and value.get("scheduler_patience", 5) == 5:
        value.pop("scheduler_patience", None)
    if value.get("hidden_activation", "keep") == "keep":
        value.pop("hidden_activation", None)
    if value.get("use_batchnorm", False) is False:
        value.pop("use_batchnorm", None)
    if value.get("weight_init", "keep") == "keep":
        value.pop("weight_init", None)
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def training_capabilities():
    return {
        "supported_fields": list(TrainingPatch.model_fields),
        "constraints": "Use field types and bounds in the response JSON schema; all numbers must be finite.",
        "semantics": "Absolute settings, applied together at an epoch boundary.",
        "null_disables": ["grad_clip_norm", "grad_clip_value", "early_stop_patience"],
        "component_options": {
            "sgd": ["momentum"],
            "rmsprop": ["momentum", "epsilon"],
            "adam_or_adamw": ["beta1", "beta2", "epsilon"],
            "step": ["scheduler_step_size", "scheduler_gamma"],
            "cosine": ["scheduler_t_max"],
            "plateau": ["scheduler_gamma", "scheduler_patience"],
        },
        "optimizer_change": "Weights retained; incompatible optimizer state reset.",
        "scheduler_change": "Scheduler reset when its type or configuration changes.",
        "architecture_change": "hidden_activation/use_batchnorm/weight_init modify hidden layers (output layer preserved); incompatible weights/optimizer state reset cleanly.",
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
    if candidate.optimizer == "sgd":
        inactive = {"beta1", "beta2", "epsilon"}
    elif candidate.optimizer == "rmsprop":
        inactive = {"beta1", "beta2"}
    else:
        inactive = {"momentum"}
    if candidate.scheduler != "step":
        inactive.add("scheduler_step_size")
    if candidate.scheduler not in ("step", "plateau"):
        inactive.add("scheduler_gamma")
    if candidate.scheduler != "plateau":
        inactive.add("scheduler_patience")
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
    # Reject micro-tweaks that only bypass != current_config without meaningful training impact.
    meaningful = False
    for key, pair in diff.items():
        b, a = pair["before"], pair["after"]
        if key in ("optimizer", "scheduler", "batch_size", "gradient_accumulation", "early_stop_patience",
                   "scheduler_step_size", "scheduler_t_max", "scheduler_patience",
                   "hidden_activation", "use_batchnorm", "weight_init"):
            meaningful = True
        elif key == "momentum":
            if abs(a - b) >= 0.02:
                meaningful = True
        elif key == "weight_decay":
            if (b == 0.0 and a >= 1e-4) or (b > 0.0 and (a == 0.0 or abs(a - b) / b >= 0.15)):
                meaningful = True
        elif key == "learning_rate":
            if b == 0.0 or abs(a - b) / max(abs(b), 1e-12) >= 0.15:
                meaningful = True
        elif key in ("grad_clip_norm", "grad_clip_value"):
            if b is None or a is None or abs(a - b) / max(abs(b), 1e-12) >= 0.15:
                meaningful = True
        elif key == "label_smoothing":
            if abs(a - b) >= 0.01:
                meaningful = True
        else:
            meaningful = True
    if not meaningful:
        raise ProposalError(
            "NEGLIGIBLE_CHANGE",
            "Patch changes are too small to meaningfully affect training (e.g. momentum delta < 0.02, lr delta < 15%, weight_decay < 1e-4)",
            diff=diff,
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
    current, response, policy, evidence_epochs, remaining_epochs, hypothesis_ranks,
    rejected_hashes=None,
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
        if rejected_hashes and compiled.config_hash in rejected_hashes:
            raise ProposalError(
                "ALREADY_TRIED",
                "This exact configuration was already tried and REJECTED from the same base config; choose a substantially stronger or different value (e.g. 10x larger learning_rate like 0.01 or 0.1, or switch optimizer to adam).",
                config_hash=compiled.config_hash,
                diff=compiled.diff,
            )
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
