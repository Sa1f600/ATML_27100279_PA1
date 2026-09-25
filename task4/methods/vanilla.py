"""ordinary ten-class cross-entropy."""

import torch.nn.functional as F


def loss(model, images, labels):
    value = F.cross_entropy(model(images), labels)
    return value, {'classification_loss': value.detach(), 'placeholder_loss': 0., 'mixup_loss': 0., 'mixed_count': 0}
