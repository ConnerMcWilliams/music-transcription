import torch
import torch.optim as optim
from typing import Optional, List


def make_optimizer(
    model,
    optimizer_type: str = "adam",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    fused: Optional[bool] = None,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model to optimize
        optimizer_type: Type of optimizer ('adam' or 'adamw')
        lr: Learning rate
        weight_decay: L2 regularization coefficient
        betas: Momentum coefficients for Adam (beta1, beta2)
        fused: Use fused optimizer implementation if available. If None, auto-detect for CUDA.
    
    Returns:
        Configured optimizer instance
    
    Example:
        optimizer = make_optimizer(model, lr=1e-4)
        optimizer = make_optimizer(model, optimizer_type='adamw', lr=1e-4, weight_decay=1e-5)
    """
    # Auto-detect fused capability if not specified
    if fused is None:
        fused = torch.cuda.is_available()
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            fused=fused,
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            fused=fused,
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=betas[0],  # Use first beta as momentum
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def make_scheduler(
    optimizer,
    scheduler_type: str = "onecycle",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('onecycle', 'cosine', 'linear', 'constant')
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        **kwargs: Additional arguments for scheduler
    
    Returns:
        Scheduler instance or None if scheduler_type == 'constant'
    
    Example:
        scheduler = make_scheduler(optimizer, epochs=100, steps_per_epoch=1000)
        for epoch in range(100):
            for step in range(1000):
                optimizer.step()
                scheduler.step()
    """
    total_steps = epochs * steps_per_epoch
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "onecycle":
        max_lr = kwargs.pop("max_lr", 1e-3)
        pct_start = kwargs.pop("pct_start", 0.1)
        anneal_strategy = kwargs.pop("anneal_strategy", "cos")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            **kwargs
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            **kwargs
        )
    elif scheduler_type == "cosine_warmup":
        # Linear warmup followed by cosine decay
        warmup_steps = kwargs.pop("warmup_steps", int(0.1 * total_steps))
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )
    elif scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps,
            **kwargs
        )
    elif scheduler_type == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    elif scheduler_type == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
