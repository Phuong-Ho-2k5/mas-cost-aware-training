"""One finite training segment; subprocess entry point releases CUDA on exit."""
import argparse
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from mas_demo.contracts import TaskSpec, CheckpointRef, EpochEvent, SegmentResult, Intervention
from mas_demo.data import get_manifest, load_datasets
from mas_demo.io import append_event, read_json, write_json
from mas_demo.training.checkpoint import save_checkpoint, validate_checkpoint
from mas_demo.training.model import SmallCNN


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly for a CPU smoke run")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def train_segment(spec, start, epochs, branch_id, output_dir, budget_seconds, lr_factor=None, intervention=None):
    if lr_factor is not None:
        if intervention is not None:
            raise ValueError("Provide only one intervention")
        intervention = Intervention(parameter="learning_rate", operation="multiply", value=lr_factor)
    if intervention is not None:
        intervention = Intervention.model_validate(intervention)
        if intervention not in spec.intervention_allowlist:
            raise ValueError("Intervention outside allowlist")
    began = time.monotonic()
    deadline = began + budget_seconds
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = SegmentResult(status="BUDGET_EXHAUSTED", checkpoint=start)
    if budget_seconds <= 0:
        return result
    torch.set_num_threads(2)
    device = resolve_device(spec.device)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
    manifest = get_manifest(spec)
    train, val = load_datasets(spec, manifest)
    generator = torch.Generator().manual_seed(spec.seed)
    model = SmallCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec.learning_rate, momentum=spec.momentum,
                                weight_decay=spec.weight_decay)
    clip_norm = spec.grad_clip_norm
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    epoch = 0
    fault_applied = False
    if start:
        validate_checkpoint(start, manifest.version)
        # Only local trusted artifacts: pickle state includes Python/NumPy RNG.
        state = torch.load(start.path, map_location=device, weights_only=False)
        if state["batch_size"] != spec.batch_size or state["seed"] != spec.seed:
            raise ValueError("Checkpoint batch/seed mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        generator.set_state(state["generator_state"].cpu())
        if device == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        epoch, fault_applied = state["epoch"], state["fault_applied"]
        clip_norm = state.get("effective_settings", {}).get("grad_clip_norm")
    if intervention is not None:
        if intervention.parameter == "grad_clip_norm":
            clip_norm = intervention.value
        elif intervention.parameter == "learning_rate":
            for group in optimizer.param_groups:
                group["lr"] *= intervention.value
            scheduler.base_lrs = [lr * intervention.value for lr in scheduler.base_lrs]
        else:
            for group in optimizer.param_groups:
                group[intervention.parameter] = intervention.value

    def effective_settings():
        group = optimizer.param_groups[0]
        return {"learning_rate": group["lr"], "momentum": group["momentum"],
                "weight_decay": group["weight_decay"], "grad_clip_norm": clip_norm}
    loader = DataLoader(train, batch_size=spec.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation = DataLoader(val, batch_size=spec.batch_size, shuffle=False, num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_quality = (-1, -float("inf"))
    guard = spec.unit_guard_seconds

    def enough():
        return time.monotonic() + guard < deadline

    for current in range(epoch + 1, min(epoch + epochs, spec.max_epochs) + 1):
        tick = time.monotonic()
        if not enough():
            break
        if spec.fault_epoch == current and not fault_applied:
            for group in optimizer.param_groups:
                group["lr"] = spec.fault_lr
            fault_applied = True
        model.train()
        train_loss, count, grad_total, batches = 0, 0, 0, 0
        numerical = False
        for x, y in loader:
            if not enough():
                break
            unit = time.monotonic()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                numerical = True
                break
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm if clip_norm is not None else float("inf"))
            if not torch.isfinite(norm):
                numerical = True
                break
            optimizer.step()
            train_loss += loss.item() * len(y)
            count += len(y)
            batches += 1
            grad_total += norm.item()
            if device == "cuda":
                torch.cuda.synchronize()
            guard = max(guard, (time.monotonic() - unit) * 1.5)
        if numerical:
            result.status = "NUMERICAL_ERROR"
            result.events.append(EpochEvent(epoch=current, branch_id=branch_id, train_loss=None,
                                           val_loss=None, val_accuracy=None,
                                           **effective_settings(), grad_norm=None))
            break
        if count != len(train):
            break  # partial epoch never becomes a valid comparison checkpoint
        model.eval()
        val_loss, correct, seen = 0, 0, 0
        with torch.no_grad():
            for x, y in validation:
                if not enough():
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    numerical = True
                    break
                val_loss += loss.item() * len(y)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(y)
        if numerical:
            result.status = "NUMERICAL_ERROR"
            break
        if seen != len(val) or not enough():
            break
        event = EpochEvent(epoch=current, branch_id=branch_id, train_loss=train_loss/count,
                           val_loss=val_loss/seen, val_accuracy=correct/seen,
                           **effective_settings(), grad_norm=grad_total/max(batches, 1),
                           seconds=time.monotonic()-tick,
                           peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else None)
        scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": current,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "generator_state": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                 "fault_applied": fault_applied, "batch_size": spec.batch_size, "seed": spec.seed,
                 "effective_settings": effective_settings()}
        result.checkpoint = save_checkpoint(output_dir/f"epoch-{current:04}.pt", state,
                                            branch_id, manifest.version, start)
        result.events.append(event)
        append_event(output_dir/"epochs.jsonl", "epoch", event=event)
        quality = (event.val_accuracy, -event.val_loss)
        if quality > best_quality:
            best_quality = quality
            result.best_checkpoint = result.checkpoint
    if result.status != "NUMERICAL_ERROR" and result.events and result.events[-1].epoch == min(epoch + epochs, spec.max_epochs):
        result.status = "COMPLETE"
    result.seconds = time.monotonic() - began
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        spec = TaskSpec.model_validate(request["spec"])
        start = CheckpointRef.model_validate(request["start"]) if request["start"] else None
        result = train_segment(spec, start, request["epochs"], request["branch_id"],
                               request["output_dir"], request["budget_seconds"], request.get("lr_factor"),
                               request.get("intervention"))
    except Exception as error:
        result = SegmentResult(status="FAILED", error=f"{type(error).__name__}: {error}")
    write_json(Path(request["output_dir"])/"result.json", result)


if __name__ == "__main__":
    main()
