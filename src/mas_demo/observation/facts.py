"""Auditable observations and effective configuration, without anomaly thresholds."""
import hashlib
import json
from math import isfinite

from mas_demo.contracts import ObservationFact, ConfigFact
from mas_demo.observation.trends import build_metric_trends

METRICS = ("train_loss", "val_loss", "val_accuracy", "grad_norm")


def build_window_summary(evidence):
    """Compact 4-decimal numeric summary + objective trend flags for small-model detection."""
    events = sorted(evidence.recent_events, key=lambda e: e.epoch)

    def _r(val):
        if val is None or not isfinite(val):
            return None
        out = round(float(val), 4)
        return 0.0 if out == 0.0 else out

    v_losses = [e.val_loss for e in events if e.val_loss is not None and isfinite(e.val_loss)]
    t_losses = [e.train_loss for e in events if e.train_loss is not None and isfinite(e.train_loss)]
    v_accs = [e.val_accuracy for e in events if e.val_accuracy is not None and isfinite(e.val_accuracy)]
    grads = [e.grad_norm for e in events if e.grad_norm is not None and isfinite(e.grad_norm)]
    val_loss_decrease = _r(v_losses[0] - v_losses[-1]) if len(v_losses) >= 2 else None
    val_acc_increase = _r(v_accs[-1] - v_accs[0]) if len(v_accs) >= 2 else None
    val_loss_range = _r(max(v_losses) - min(v_losses)) if len(v_losses) >= 2 else None
    val_acc_range = _r(max(v_accs) - min(v_accs)) if len(v_accs) >= 2 else None
    stalled = bool(
        len(events) >= 3
        and val_loss_decrease is not None
        and val_acc_increase is not None
        and val_loss_decrease < 0.01
        and val_acc_increase <= 0.005
    )
    numerical_error = (
        "numerical_error" in (getattr(evidence, "quality_flags", None) or [])
        or any(
            e.train_loss is None or e.val_loss is None or e.val_accuracy is None
            for e in events
        )
    )
    unstable = bool(
        not numerical_error
        and len(events) >= 2
        and (
            (val_loss_decrease is not None and val_loss_decrease < -0.02)
            or (val_acc_increase is not None and val_acc_increase < -0.05)
            or (val_loss_range is not None and val_loss_range > 0.15 and val_loss_decrease is not None and val_loss_decrease < 0.05)
        )
    )
    return {
        "num_epochs": len(events),
        "val_loss_start": _r(v_losses[0]) if v_losses else None,
        "val_loss_end": _r(v_losses[-1]) if v_losses else None,
        "val_loss_decrease": val_loss_decrease,
        "val_loss_range": val_loss_range,
        "val_acc_start": _r(v_accs[0]) if v_accs else None,
        "val_acc_end": _r(v_accs[-1]) if v_accs else None,
        "val_acc_increase": val_acc_increase,
        "val_acc_range": val_acc_range,
        "train_loss_decrease": _r(t_losses[0] - t_losses[-1]) if len(t_losses) >= 2 else None,
        "grad_norm_mean": _r(sum(grads) / len(grads)) if grads else None,
        "stalled_window": stalled,
        "unstable_window": unstable,
        "numerical_error": numerical_error,
    }


_TELEMETRY_EXCLUDE = {"seconds", "peak_vram_mb", "config_hash"}


def clean_evidence_for_llm(evidence):
    """Strip non-deterministic wall-clock and hash telemetry from recent_events before sending to LLMs."""
    data = evidence.model_dump(exclude={"signals"})
    data["recent_events"] = [
        {k: v for k, v in ev.items() if k not in _TELEMETRY_EXCLUDE}
        for ev in data.get("recent_events", [])
    ]
    return data


def build_epoch_log_payload(evidence):
    """Worker epoch records (without wall-clock/hash jitter) plus compact 4-decimal window summary."""
    if len(evidence.epochs) != len(set(evidence.epochs)):
        raise ValueError("Duplicate evidence epochs")
    record_epochs = [event.epoch for event in evidence.recent_events]
    if len(record_epochs) != len(set(record_epochs)) or set(record_epochs) != set(evidence.epochs):
        raise ValueError("Selected epochs must match distinct original worker records")
    return dict(schema_version=3, input_format="epoch_logs",
                window_summary=build_window_summary(evidence),
                evidence=clean_evidence_for_llm(evidence))


def fact_has_measurement(fact):
    data = fact if isinstance(fact, dict) else fact.model_dump()
    return data.get("value") is not None or data.get("pattern") in (
        "mixed", "flat", "nondecreasing", "nonincreasing")


def build_observation_facts(evidence):
    if len(evidence.epochs) != len(set(evidence.epochs)):
        raise ValueError("Duplicate evidence epochs")
    events = sorted((e for e in evidence.recent_events
                     if e.branch_id == evidence.branch_id and e.epoch in evidence.epochs),
                    key=lambda e: e.epoch)
    if len({e.epoch for e in events}) != len(events):
        raise ValueError("Duplicate observed epochs")
    facts = []

    def add(metric, kind, epochs, **values):
        content = dict(branch=evidence.branch_id, metric=metric, kind=kind, epochs=epochs, **values)
        identity = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        facts.append(ObservationFact(fact_id=identity, metric=metric, kind=kind, epochs=epochs, **values))

    for e in events:
        for metric in METRICS:
            value = getattr(e, metric)
            if value is not None and isfinite(value):
                add(metric, "value", [e.epoch], value=value)
    trends = build_metric_trends(evidence)
    for metric, summary in trends["metrics"].items():
        if not any(getattr(event, metric) is not None for event in events):
            continue
        for pair, value, direction in zip(trends["epoch_pairs"], summary["step_deltas"], summary["step_directions"]):
            if value is not None:
                add(metric, "step_delta", pair, value=value, direction=direction)
        value = summary["window_delta"]
        if value is not None:
            add(metric, "window_delta", [events[0].epoch, events[-1].epoch], value=value,
                direction="up" if value > 0 else "down" if value < 0 else "flat")
        add(metric, "pattern", [e.epoch for e in events], pattern=summary["pattern"])
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("Duplicate fact identity")
    return facts


def build_config_facts(config):
    def active(field):
        if field == "grad_clip_norm": return config.grad_clip_norm is not None
        if field == "grad_clip_value": return config.grad_clip_value is not None
        if field == "label_smoothing": return config.label_smoothing > 0
        if field == "weight_decay": return config.weight_decay > 0
        if field == "momentum": return config.optimizer in ("sgd", "rmsprop") and config.momentum > 0
        if field in ("beta1", "beta2"): return config.optimizer in ("adam", "adamw")
        if field == "epsilon": return config.optimizer in ("adam", "adamw", "rmsprop")
        if field == "scheduler_step_size": return config.scheduler == "step"
        if field == "scheduler_gamma": return config.scheduler in ("step", "plateau")
        if field == "scheduler_patience": return config.scheduler == "plateau"
        if field == "scheduler_t_max": return config.scheduler == "cosine"
        if field == "scheduler": return config.scheduler != "none"
        if field == "early_stop_patience": return config.early_stop_patience is not None
        if field == "hidden_activation": return config.hidden_activation != "keep"
        if field == "use_batchnorm": return bool(config.use_batchnorm)
        if field == "weight_init": return config.weight_init != "keep"
        return True
    return [ConfigFact(field=field, value=value, active=active(field))
            for field, value in config.model_dump().items()]


def build_detection_payload(evidence, facts):
    own = sorted((e for e in evidence.recent_events
                  if e.branch_id == evidence.branch_id and e.epoch in evidence.epochs), key=lambda e: e.epoch)
    return dict(
        schema_version=3,
        evidence=dict(epochs=list(evidence.epochs), branch_id=evidence.branch_id,
                      recent_events=[dict(epoch=e.epoch, **{m: getattr(e, m) for m in METRICS}) for e in own]),
        metric_trends=build_metric_trends(evidence),
        availability={m: sum(getattr(e, m) is not None for e in own) for m in METRICS},
        accuracy_delta_unit="percentage_points",
        facts=[f.model_dump(exclude_none=True) for f in facts],
    )
