"""standard, non-adaptive sam with exact restoration before the optimizer step."""

from contextlib import contextmanager

import torch


@contextmanager
def ascent_step(parameters, radius):
    parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not parameters or radius < 0:
        raise ValueError("ascent requires gradients and a nonnegative radius")
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite ascent gradient")
    originals = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            scale = radius / norm.clamp_min(1e-12)
            for p in parameters:
                p.add_(p.grad * scale)
        yield
    finally:
        # copying restores exact original values even when the second pass fails
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)


def sam_step(closure, parameters, optimizer, radius):
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss = closure()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite first sam loss")
    loss.backward()
    with ascent_step(parameters, radius):
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = closure()
        if not torch.isfinite(perturbed_loss):
            raise FloatingPointError("non-finite second sam loss")
        perturbed_loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("non-finite second sam gradient")
    # apply adamw and weight decay to original parameters using perturbed gradients
    optimizer.step()
    return loss.item(), perturbed_loss.item()
