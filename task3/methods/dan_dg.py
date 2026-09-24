"""average the same task 2 mmd over all three source-domain pairs."""

from itertools import combinations

import torch

from task2.methods.dan import mmd_loss


def pairwise_mmd(features, batch_size=8):
    if len(features) != 3 * batch_size:
        raise ValueError("expected three equally sized source batches")
    domains = features.split(batch_size)
    return torch.stack([mmd_loss(a, b) for a, b in combinations(domains, 2)]).mean()
