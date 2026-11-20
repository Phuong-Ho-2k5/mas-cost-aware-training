"""Auditable observations and effective configuration, without anomaly thresholds."""
import hashlib
import json
from math import isfinite

from mas_demo.contracts import ObservationFact, ConfigFact
from mas_demo.observation.trends import build_metric_trends

METRICS = ("train_loss", "val_loss", "val_accuracy", "grad_norm")


def build_epoch_log_payload(evidence):
    """Full worker epoch records; no metric projection or derived fact tables."""
    if len(evidence.epochs) != len(set(evidence.epochs)):
        raise ValueError("Duplicate evidence epochs")
    record_epochs = [event.epoch for event in evidence.recent_events]
    if len(record_epochs) != len(set(record_epochs)) or set(record_epochs) != set(evidence.epochs):
        raise ValueError("Selected epochs must match distinct original worker records")
    return dict(schema_version=3, input_format="epoch_logs",
                evidence=evidence.model_dump(exclude={"signals"}))


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
        if field == "weight_decay": return config.weight_decay > 0
        if field == "momentum": return config.optimizer == "sgd" and config.momentum > 0
        if field in ("beta1", "beta2", "epsilon"): return config.optimizer in ("adam", "adamw")
        if field in ("scheduler_step_size", "scheduler_gamma"): return config.scheduler == "step"
        if field == "scheduler_t_max": return config.scheduler == "cosine"
        if field == "scheduler": return config.scheduler != "none"
        if field == "early_stop_patience": return config.early_stop_patience is not None
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
