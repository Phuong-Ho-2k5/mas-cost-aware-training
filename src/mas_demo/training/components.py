"""Typed adapters for the optimizer and scheduler capabilities exposed to Planner."""

import torch


def make_optimizer(model, config):
    common = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=config.momentum, **common)
    cls = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
    return cls(
        model.parameters(),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        **common
    )


def make_scheduler(optimizer, config):
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_t_max
    )


def scheduler_identity(config):
    return (
        config.scheduler,
        config.scheduler_step_size,
        config.scheduler_gamma,
        config.scheduler_t_max,
    )


def apply_optimizer_settings(optimizer, config):
    for group in optimizer.param_groups:
        group.update(lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.optimizer == "sgd":
            group["momentum"] = config.momentum
        else:
            group.update(betas=(config.beta1, config.beta2), eps=config.epsilon)
