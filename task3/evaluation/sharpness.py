"""one fixed source-validation batch and one radius-0.05 ascent diagnostic."""

import numpy as np
import torch
import torch.nn.functional as F

from shared.pacs_protocol import SOURCES
from task3.methods.sam import ascent_step


def sharpness_paths(sources, seed=6304, per_domain=32):
    rng = np.random.default_rng(seed)
    selected = {}
    for domain in SOURCES:
        paths = sources[domain]['val']
        if len(paths) < per_domain:
            raise ValueError(f'{domain} needs {per_domain} validation images')
        ids = rng.choice(len(paths), per_domain, replace=False)
        selected[domain] = [paths[i] for i in ids]
    return selected


def sharpness(backbone, classifier, images, labels, radius=0.05):
    modules = list(backbone.modules()) + list(classifier.modules())
    modes = [module.training for module in modules]
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    previous_gradients = [p.grad for p in parameters]
    backbone.eval()
    classifier.eval()
    try:
        for p in parameters:
            p.grad = None
        with torch.enable_grad():
            loss = F.cross_entropy(classifier(backbone(images)), labels)
            loss.backward()
        with ascent_step(parameters, radius), torch.no_grad():
            perturbed = F.cross_entropy(classifier(backbone(images)), labels)
        if not torch.isfinite(perturbed):
            raise FloatingPointError('non-finite sharpness diagnostic')
        return {'loss': loss.item(), 'perturbed_loss': perturbed.item(),
                'delta': perturbed.item() - loss.item(), 'radius': radius}
    finally:
        for p, gradient in zip(parameters, previous_gradients):
            p.grad = gradient
        for module, mode in zip(modules, modes):
            module.training = mode
