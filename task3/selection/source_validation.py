"""source-only data access and checkpoint scores."""

from pathlib import PurePosixPath

import numpy as np

from shared.pacs import make_loader
from shared.pacs_protocol import CLASSES, SOURCES
from task2.evaluation.metrics import classification_metrics, collect


def source_splits(manifest):
    if manifest['seed'] != 6304 or manifest['classes'] != list(CLASSES):
        raise ValueError('unexpected shared PACS protocol')
    sources = manifest['sources']
    if set(sources) != set(SOURCES):
        raise ValueError('expected exactly three source domains')
    for domain in SOURCES:
        partitions = sources[domain]
        for paths in partitions.values():
            for path in paths:
                parts = PurePosixPath(path).parts
                if not parts or parts[0] != domain or '..' in parts or PurePosixPath(path).is_absolute():
                    raise ValueError(f'non-source or unsafe path: {path}')
        train, val = partitions['train'], partitions['val']
        if set(train) & set(val) or len(set(train + val)) != len(train + val):
            raise ValueError(f'overlapping or duplicate source paths in {domain}')
    # downstream training and source diagnostics receive no target entry
    return {domain: sources[domain] for domain in SOURCES}


def validation_loaders(root, sources, config):
    return {domain: make_loader(root, sources[domain]['val'], config['eval_batch_size'],
                                config['seed'], config['workers']) for domain in SOURCES}


def validate(backbone, classifier, loaders, device):
    scores = {}
    for domain, loader in loaders.items():
        values = collect(backbone, classifier, loader, device)
        scores[domain] = classification_metrics(values['labels'], values['logits'].argmax(1))
    return scores, float(np.mean([score['macro_f1'] for score in scores.values()]))
