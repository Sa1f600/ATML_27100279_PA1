"""negative log-sum-exp, with temperature fixed at one."""

import torch


def score(logits):
    return -torch.logsumexp(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()
