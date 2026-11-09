from mas_demo.contracts import EvidenceWindow


def build_evidence(events, branch_id, window=2):
    """Collect observed metrics; anomaly detection belongs to the Analysis LLM."""
    own = sorted((e for e in events if e.branch_id == branch_id), key=lambda e: e.epoch)
    recent = own[-window:]
    quality = []
    if any(e.peak_vram_mb is None for e in recent):
        quality.append("gpu_telemetry_unavailable")
    valid = len(recent) == window and all(e.val_loss is not None for e in recent)
    return EvidenceWindow(
        epochs=[e.epoch for e in recent],
        branch_id=branch_id,
        status="READY" if valid else "INSUFFICIENT",
        recent_events=recent,
        quality_flags=quality,
    )
