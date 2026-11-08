from mas_demo.contracts import AdmissionBatch


def admit_trials(trials, ledger, epoch_seconds, overhead, margin, continuation, round_id):
    horizon = trials[0].horizon if trials else 1
    batch = AdmissionBatch(round_id=round_id, horizon=horizon)
    if not trials:
        return batch
    if any(t.horizon != horizon for t in trials):
        raise ValueError("Trials must share a horizon")
    cost = (epoch_seconds * horizon + overhead) * (1 + margin)
    continuation_key = f"round-{round_id}:continuation"
    control_key = f"round-{round_id}:control"
    if continuation > 0 and not ledger.reserve(continuation_key, continuation):
        batch.skipped = [t.trial_id for t in trials]
        return batch
    batch.continuation_key = continuation_key if continuation > 0 else None
    if ledger.available < cost * 2 or not ledger.reserve(control_key, cost):
        if batch.continuation_key:
            ledger.release(batch.continuation_key)
            batch.continuation_key = None
        batch.skipped = [t.trial_id for t in trials]
        return batch
    batch.control_key = control_key
    for trial in trials:
        key = f"round-{round_id}:{trial.trial_id}"
        if ledger.reserve(key, cost):
            batch.approved.append(trial)
            batch.reservation_keys[trial.trial_id] = key
        else:
            batch.skipped.append(trial.trial_id)
    return batch
