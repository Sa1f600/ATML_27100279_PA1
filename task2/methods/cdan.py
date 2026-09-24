"""conditional alignment without entropy weighting or detached predictions."""

import torch

from task2.methods.dann import domain_loss


def conditional_features(features, logits):
    probabilities = logits.softmax(dim=1)
    # gradients flow through both the features and class probabilities
    return torch.einsum("bi,bj->bij", features, probabilities).flatten(1)


def conditional_loss(features, logits, source_count, discriminator, strength):
    return domain_loss(conditional_features(features, logits), source_count,
                       discriminator, strength)
