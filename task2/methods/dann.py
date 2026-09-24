"""marginal adversarial domain alignment."""

import torch
import torch.nn.functional as F

from task2.models.domain_discriminator import reverse


def domain_loss(features, source_count, discriminator, strength):
    labels = torch.arange(len(features), device=features.device).ge(source_count).long()
    logits = discriminator(reverse(features, strength))
    return F.cross_entropy(logits, labels)
