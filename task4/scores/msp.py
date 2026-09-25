"""larger values indicate lower maximum softmax confidence."""

import torch


def score(logits):
    return (1 - torch.as_tensor(logits, dtype=torch.float64).softmax(1).max(1).values).numpy()
