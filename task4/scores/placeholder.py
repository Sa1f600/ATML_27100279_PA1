"""reference CONF_DeltaP score, with fixed T=1024 and zero logit bias."""

import torch

from task4.methods.proser import collapse_dummy


def score(logits):
    probabilities = (collapse_dummy(torch.as_tensor(logits, dtype=torch.float64)) / 1024.).softmax(1)
    return (probabilities[:, -1] - probabilities[:, :10].max(1).values).numpy()
