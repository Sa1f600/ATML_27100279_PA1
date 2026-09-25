"""paper equations 5-7: nearest dummy, classifier placeholders, and mixup."""

import torch
import torch.nn.functional as F

from task4.methods.manifold_mixup import mix_hidden


def collapse_dummy(logits):
    if logits.shape[1] <= 10:
        raise ValueError('PROSER requires dummy outputs')
    return torch.cat([logits[:, :10], logits[:, 10:].max(dim=1, keepdim=True).values], dim=1)


def classifier_placeholder_loss(logits, labels, beta=1.0):
    combined = collapse_dummy(logits)
    classification = F.cross_entropy(combined, labels)
    # remove the true known class so the strongest dummy competes with wrong classes
    masked = combined.scatter(1, labels[:, None], float('-inf'))
    placeholder = F.cross_entropy(masked, torch.full_like(labels, 10))
    return classification + beta * placeholder, classification, placeholder


def loss(model, images, labels, beta=1.0, gamma=0.1, alpha=2.0):
    if len(images) % 2 or len(images) < 2:
        raise ValueError('PROSER needs two equal nonempty batch halves')
    half = len(images) // 2
    first = model(images[:half])
    classifier_loss, classification, placeholder = classifier_placeholder_loss(first, labels[:half], beta)
    hidden = model.before_mix(images[half:])
    mixed, _ = mix_hidden(hidden, labels[half:], alpha)
    data_loss = classifier_loss.new_zeros(())
    if len(mixed):
        logits = collapse_dummy(model.classify(model.after_mix(mixed)))
        data_loss = F.cross_entropy(logits, torch.full((len(mixed),), 10, device=labels.device, dtype=torch.long))
    return classifier_loss + gamma * data_loss, {
        'classification_loss': classification.detach(), 'placeholder_loss': placeholder.detach(),
        'mixup_loss': data_loss.detach(), 'mixed_count': len(mixed)}
