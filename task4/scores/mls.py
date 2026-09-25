"""negative maximum known-class logit."""

import numpy as np


def score(logits):
    return -np.asarray(logits, dtype=np.float64)[:, :10].max(axis=1)
