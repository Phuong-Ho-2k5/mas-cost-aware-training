from pathlib import Path
from mas_demo.contracts import VerificationRoundResult, SegmentResult
from mas_demo.training.checkpoint import validate_checkpoint
from mas_demo.io import write_json


def verify_round(spec, fork, admission, ledger, backend, directory):
    validate_checkpoint(fork)
    if not admission.approved or admission.control_key is None:
        raise ValueError("No admitted trial")
    if any(t.intervention not in spec.intervention_allowlist or t.horizon != admission.horizon for t in admission.approved):
        raise ValueError("Invalid action or incomparable trial")
    directory = Path(directory)
    key = admission.control_key
    control = backend.run(spec, fork, admission.horizon, "main", directory/"control",
                          ledger.reservations[key])
    ledger.settle(key, control.seconds)
    result = VerificationRoundResult(comparison_epoch=fork.epoch + admission.horizon,
                                     fork_ref=fork, control=control)
    for trial in admission.approved:
        key = admission.reservation_keys[trial.trial_id]
        if control.status != "COMPLETE" or ledger.consumed >= ledger.cap:
            ledger.release(key)
            result.trials[trial.trial_id] = SegmentResult(status="FAILED", error="Control incomplete or budget exhausted; not executed")
        else:
            part = backend.run(spec, fork, trial.horizon, trial.trial_id,
                               directory/trial.trial_id, ledger.reservations[key], intervention=trial.intervention)
            ledger.settle(key, part.seconds)
            result.trials[trial.trial_id] = part
        write_json(directory/"round_result.json", result)
    if admission.continuation_key:
        ledger.release(admission.continuation_key)
    write_json(directory/"round_result.json", result)
    return result
