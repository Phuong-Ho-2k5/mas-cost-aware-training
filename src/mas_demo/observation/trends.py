"""Arithmetic summaries of observed metrics, without anomaly thresholds."""

from math import isfinite


def _delta(before, after, scale):
    if before is None or after is None or not isfinite(before) or not isfinite(after):
        return None
    change = (after - before) * scale
    # Keep small nonzero changes visible without sending long float expansions.
    return float(f"{change:.6g}") if isfinite(change) else None


def _pattern(directions):
    observed = set(directions)
    if not observed:
        return "insufficient"
    if "unavailable" in observed:
        return "incomplete"
    if "up" in observed and "down" in observed:
        return "mixed"
    if "up" in observed:
        return "nondecreasing"
    if "down" in observed:
        return "nonincreasing"
    return "flat"


def build_metric_trends(evidence):
    """Compare neighboring observations in the evidence window's own branch."""
    epochs = set(evidence.epochs)
    events = sorted(
        (event for event in evidence.recent_events
         if event.branch_id == evidence.branch_id and event.epoch in epochs),
        key=lambda event: event.epoch,
    )
    pairs = list(zip(events, events[1:]))
    summary = {
        "epoch_pairs": [[before.epoch, after.epoch] for before, after in pairs],
        "accuracy_delta_unit": "percentage_points",
        "metrics": {},
    }
    if not pairs:
        return summary
    for name in ("train_loss", "val_loss", "val_accuracy", "grad_norm"):
        scale = 100 if name == "val_accuracy" else 1
        deltas = [
            _delta(getattr(before, name), getattr(after, name), scale)
            for before, after in pairs
        ]
        directions = [
            "unavailable" if change is None else
            "up" if change > 0 else "down" if change < 0 else "flat"
            for change in deltas
        ]
        summary["metrics"][name] = {
            "step_deltas": deltas,
            "step_directions": directions,
            "window_delta": _delta(getattr(events[0], name), getattr(events[-1], name), scale),
            "pattern": _pattern(directions),
        }
    return summary
