"""Finite training segments; proposals are applied only after checkpoint restore."""

import argparse
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from mas_demo.contracts import (
    TaskSpec,
    TrainingConfig,
    CheckpointRef,
    EpochEvent,
    SegmentResult,
)
from mas_demo.configuration import config_hash
from mas_demo.data import get_manifest, load_datasets
from mas_demo.io import append_event, read_json, write_json
from mas_demo.training.checkpoint import save_checkpoint, validate_checkpoint
from mas_demo.training.components import (
    make_optimizer,
    make_scheduler,
    scheduler_identity,
    apply_optimizer_settings,
)
from mas_demo.training.model import create_model, model_signature


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select cpu explicitly")
    return (
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else ("cpu" if requested == "auto" else requested)
    )


def train_segment(
    spec, start, epochs, branch_id, output_dir, budget_seconds, training_config=None
):
    began = time.monotonic()
    deadline = began + budget_seconds
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = SegmentResult(status="BUDGET_EXHAUSTED")
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
    train, validation_data = load_datasets(spec, manifest)
    generator = torch.Generator().manual_seed(spec.seed)
    model = create_model(spec).to(device)
    previous = None
    if start:
        validate_checkpoint(start, manifest.version)
        previous = torch.load(start.path, map_location=device, weights_only=False)
        if previous.get("schema_version") != 3:
            raise ValueError(
                "Legacy checkpoint requires explicit migration; cannot resume as v3"
            )
        if previous["seed"] != spec.seed:
            raise ValueError("Checkpoint seed mismatch")
        if (
            config_hash(previous["training_config"]) != previous["config_hash"]
            or start.config_hash != previous["config_hash"]
        ):
            raise ValueError("Checkpoint configuration hash mismatch")
        if previous.get("model_signature", "SmallCNN") != model_signature(spec):
            raise ValueError("Checkpoint model identity mismatch")
        model.load_state_dict(previous["model"])
    old_config = (
        TrainingConfig.model_validate(previous["training_config"])
        if previous
        else spec.training
    )
    config = TrainingConfig.model_validate(training_config or old_config).model_copy(
        deep=True
    )
    optimizer = make_optimizer(model, config)
    optimizer_reset = bool(previous and old_config.optimizer != config.optimizer)
    scheduler_reset = bool(
        previous
        and (
            optimizer_reset
            or scheduler_identity(old_config) != scheduler_identity(config)
            or (old_config.learning_rate == 0 and config.learning_rate > 0)
        )
    )
    if previous and not optimizer_reset:
        optimizer.load_state_dict(previous["optimizer"])
    apply_optimizer_settings(optimizer, config)
    if scheduler_reset:
        for group in optimizer.param_groups:
            group.pop("initial_lr", None)
    scheduler = make_scheduler(optimizer, config)
    if previous and scheduler is not None and not scheduler_reset:
        scheduler.load_state_dict(previous["scheduler"])
        ratio = (
            config.learning_rate / old_config.learning_rate
            if old_config.learning_rate
            else 1.0
        )
        scheduler.base_lrs = [value * ratio for value in scheduler.base_lrs]
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [config.learning_rate] * len(optimizer.param_groups)
    apply_optimizer_settings(optimizer, config)
    result.transition = {
        "optimizer_reset": optimizer_reset,
        "scheduler_reset": scheduler_reset,
    }
    epoch = 0
    samples_seen = optimizer_steps = bad_epochs = 0
    best_val_loss = float("inf")
    if previous:
        epoch = previous["epoch"]
        samples_seen, optimizer_steps = (
            previous["samples_seen"],
            previous["optimizer_steps"],
        )
        best_val_loss, bad_epochs = previous["best_val_loss"], previous["bad_epochs"]
        if config_hash(config) != config_hash(old_config):
            best_val_loss, bad_epochs = float("inf"), 0
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        torch.set_rng_state(previous["torch_rng"].cpu())
        generator.set_state(previous["generator_state"].cpu())
        if device == "cuda" and previous["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in previous["cuda_rng"]]
            )
    loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    target_epoch = min(epoch + epochs, spec.max_epochs)
    best_quality = (-1, -float("inf"))
    guard = spec.unit_guard_seconds

    def enough():
        return time.monotonic() + guard < deadline

    for current in range(epoch + 1, target_epoch + 1):
        if not enough():
            break
        tick = time.monotonic()
        model.train()
        train_loss = grad_total = 0.0
        count = step_count = 0
        numerical = False
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (x, y) in enumerate(loader):
            if not enough():
                break
            batch_started = time.monotonic()
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                numerical = True
                break
            group_start = (
                batch_index // config.gradient_accumulation
            ) * config.gradient_accumulation
            group_samples = min(
                len(train) - group_start * config.batch_size,
                config.gradient_accumulation * config.batch_size,
            )
            (loss * len(y) / group_samples).backward()
            train_loss += loss.item() * len(y)
            count += len(y)
            if (
                batch_index + 1
            ) % config.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm or float("inf")
                )
                if not torch.isfinite(norm):
                    numerical = True
                    break
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                step_count += 1
                grad_total += norm.item()
            if device == "cuda":
                torch.cuda.synchronize()
            guard = max(guard, (time.monotonic() - batch_started) * 1.5)
        if numerical:
            result.status = "NUMERICAL_ERROR"
            break
        if count != len(train):
            break
        model.eval()
        val_loss, correct, seen = 0.0, 0, 0
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
        if seen != len(validation_data) or not enough():
            break
        samples_seen += count
        event = EpochEvent(
            epoch=current,
            branch_id=branch_id,
            train_loss=train_loss / count,
            val_loss=val_loss / seen,
            val_accuracy=correct / seen,
            learning_rate=config.learning_rate,
            momentum=config.momentum if config.optimizer == "sgd" else None,
            weight_decay=config.weight_decay,
            grad_clip_norm=config.grad_clip_norm,
            grad_norm=grad_total / max(step_count, 1),
            seconds=time.monotonic() - tick,
            peak_vram_mb=(
                torch.cuda.max_memory_allocated() / 1024**2
                if device == "cuda"
                else None
            ),
            config_hash=config_hash(config),
            samples_seen=samples_seen,
            optimizer_steps=optimizer_steps,
        )
        if event.val_loss < best_val_loss:
            best_val_loss, bad_epochs = event.val_loss, 0
        else:
            bad_epochs += 1
        if scheduler is not None:
            scheduler.step()
            config.learning_rate = optimizer.param_groups[0]["lr"]
        state = {
            "schema_version": 3,
            "model_signature": model_signature(spec),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": current,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "generator_state": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "seed": spec.seed,
            "training_config": config.model_dump(),
            "config_hash": config_hash(config),
            "samples_seen": samples_seen,
            "optimizer_steps": optimizer_steps,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "transition": result.transition,
        }
        result.checkpoint = save_checkpoint(
            directory / f"epoch-{current:04}.pt",
            state,
            branch_id,
            manifest.version,
            start,
        )
        result.events.append(event)
        append_event(directory / "epochs.jsonl", "epoch", event=event)
        result.training_config = config.model_copy(deep=True)
        quality = (event.val_accuracy, -event.val_loss)
        if quality > best_quality:
            best_quality = quality
            result.best_checkpoint = result.checkpoint
        if (
            config.early_stop_patience is not None
            and bad_epochs >= config.early_stop_patience
        ):
            result.status = "EARLY_STOPPED"
            break
    if (
        result.status not in ("NUMERICAL_ERROR", "EARLY_STOPPED")
        and result.events
        and result.events[-1].epoch == target_epoch
    ):
        result.status = "COMPLETE"
    result.seconds = time.monotonic() - began
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    request = read_json(parser.parse_args().request)
    try:
        spec = TaskSpec.model_validate(request["spec"])
        start = (
            CheckpointRef.model_validate(request["start"]) if request["start"] else None
        )
        result = train_segment(
            spec,
            start,
            request["epochs"],
            request["branch_id"],
            request["output_dir"],
            request["budget_seconds"],
            request.get("training_config"),
        )
    except Exception as error:
        result = SegmentResult(
            status="FAILED", error=f"{type(error).__name__}: {error}"
        )
    write_json(Path(request["output_dir"]) / "result.json", result)


if __name__ == "__main__":
    main()
