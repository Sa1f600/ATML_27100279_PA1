"""biased, nonnegative multi-kernel mmd on penultimate features."""

import torch


def mmd_loss(source, target):
    features = torch.cat([source, target])
    distances = torch.cdist(features, features).square()
    # exclude self-distances and stop gradients through the bandwidth heuristic
    mask = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    median = distances.detach()[mask].median().clamp_min(1e-8)
    # bandwidth denotes the squared-distance scale: k = exp(-d_squared / (2 * bandwidth))
    kernels = sum(torch.exp(-distances / (2 * factor * median)) for factor in (0.5, 1, 2))
    n = len(source)
    return kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()
