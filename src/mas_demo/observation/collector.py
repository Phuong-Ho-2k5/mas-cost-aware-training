import math
import statistics
from mas_demo.contracts import EvidenceWindow


def build_evidence(events, branch_id, window=2):
    own = sorted((e for e in events if e.branch_id == branch_id), key=lambda e: e.epoch)
    recent = own[-window:]
    quality = []
    if any(e.peak_vram_mb is None for e in recent):
        quality.append("gpu_telemetry_unavailable")
    valid = len(recent) == window and all(e.val_loss is not None for e in recent)
    signals = []
    if any(e.val_loss is None or e.train_loss is None for e in recent):
        signals.append("numerical_error")
    if valid:
        losses = [e.val_loss for e in recent]
        previous = [e.val_loss for e in own[:-1] if e.val_loss is not None]
        if previous and losses[-1] > min(previous) * 1.3:
            signals.append("loss_spike")
        progress = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-8)
        if progress < 0.01:
            signals.append("plateau")
        if len(losses) >= 3:
            changes = [b - a for a, b in zip(losses, losses[1:])]
            if any(a * b < 0 for a, b in zip(changes, changes[1:])) and statistics.pstdev(losses) > 0.05 * max(abs(statistics.mean(losses)), 1e-8):
                signals.append("oscillation")
    return EvidenceWindow(epochs=[e.epoch for e in recent], branch_id=branch_id,
                          status="READY" if valid else "INSUFFICIENT", signals=signals,
                          recent_events=recent, quality_flags=quality)
