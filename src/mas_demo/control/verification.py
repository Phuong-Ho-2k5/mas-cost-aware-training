"""Execute all branches from one fork with the same admitted horizon."""

from pathlib import Path
from mas_demo.contracts import VerificationRoundResult, SegmentResult
from mas_demo.configuration import config_hash
from mas_demo.training.checkpoint import validate_checkpoint
from mas_demo.io import write_json


def verify_round(
    spec, fork, admission, ledger, backend, directory, current_config=None
):
    validate_checkpoint(fork)
    current_config = current_config or spec.training
    if fork.config_hash != config_hash(current_config):
        raise ValueError("Fork/config hash mismatch")
    if not admission.approved or any(
        c.base_config_hash != fork.config_hash for c in admission.approved
    ):
        raise ValueError("Candidates are not based on the active fork config")
    directory = Path(directory)
    control = backend.run(
        spec,
        fork,
        admission.horizon,
        "main",
        directory / "control",
        ledger.reservations[admission.control_key],
        training_config=current_config,
    )
    ledger.settle(admission.control_key, control.seconds)
    result = VerificationRoundResult(
        comparison_epoch=fork.epoch + admission.horizon, fork_ref=fork, control=control
    )
    for candidate in admission.approved:
        key = admission.reservation_keys[candidate.trial_id]
        if control.status != "COMPLETE" or ledger.consumed >= ledger.cap:
            ledger.release(key)
            part = SegmentResult(
                status="FAILED",
                error="Not executed: control incomplete or cap exhausted",
            )
        else:
            part = backend.run(
                spec,
                fork,
                admission.horizon,
                candidate.trial_id,
                directory / candidate.trial_id,
                ledger.reservations[key],
                training_config=candidate.training_config,
            )
            ledger.settle(key, part.seconds)
        result.trials[candidate.trial_id] = part
        write_json(directory / "result.json", result)
    if admission.continuation_key:
        ledger.release(admission.continuation_key)
    return result
