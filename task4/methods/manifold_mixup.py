"""mix only different-class pairs from the second half of a known batch."""

import torch


def mix_hidden(hidden, labels, alpha=2.0):
    permutation = torch.randperm(len(labels), device=labels.device)
    keep = labels != labels[permutation]
    weight = torch.distributions.Beta(alpha, alpha).sample().to(hidden.device)
    # omit same-class pairs, including the all-one-class edge case
    mixed = weight * hidden[keep] + (1 - weight) * hidden[permutation[keep]]
    return mixed, {'permutation': permutation, 'keep': keep, 'weight': weight}
