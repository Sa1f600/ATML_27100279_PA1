"""the shared seven-class linear head."""

from torch import nn


def make_classifier():
    return nn.Linear(512, 7)
