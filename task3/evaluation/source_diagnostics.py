"""run all diagnostics using source validation images only."""

import json
from pathlib import Path

import numpy as np
import torch

from shared.pacs import make_loader
from shared.pacs_protocol import SOURCES
from task3.evaluation.domain_metrics import classification_metrics, collect, source_summary
from task3.evaluation.sharpness import sharpness, sharpness_paths
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.selection.source_validation import source_splits, validation_loaders
from task3.utils import digest, load_model, save_csv, save_json, verify_lock


def diagnose(root, output, device):
    output = Path(output)
    plan, _ = verify_lock(output)
    sources = source_splits(json.loads((output / 'splits.json').read_text()))
    selected = sharpness_paths(sources)
    folder = output / 'source_diagnostics'
    folder.mkdir(exist_ok=True)
    save_json(folder / 'sharpness_batch.json', selected)
    paths = [path for domain in SOURCES for path in selected[domain]]
    images, labels = next(iter(make_loader(root, paths, 96, 6304, workers=0)))
    images, labels = images.to(device), labels.to(device)
    rows = []
    for name, config in plan['runs'].items():
        backbone, classifier, checkpoint = load_model(output / name / 'best.pt', device)
        features, scores = {}, {}
        for domain, loader in validation_loaders(root, sources, config).items():
            values = collect(backbone, classifier, loader, device)
            features[domain] = values['features']
            scores[domain] = classification_metrics(values['labels'], values['logits'].argmax(1))
            np.savez_compressed(folder / f'{name}_{domain}.npz', **values,
                                paths=np.asarray(sources[domain]['val']))
        probe = source_domain_separability(features)
        proxy = sharpness(backbone, classifier, images, labels)
        save_json(folder / f'{name}_probe.json', probe)
        save_json(folder / f'{name}_sharpness.json', proxy)
        rows.append({'method': name, 'selected_epoch': checkpoint['epoch'], **source_summary(scores),
                     'source_domain_separability': probe['accuracy'], 'sharpness_delta': proxy['delta']})
        print(f"{name}: source probe={probe['accuracy']:.4f}, sharpness={proxy['delta']:.4f}", flush=True)
    save_csv(folder / 'source_results.csv', rows)
    save_json(folder / 'source_results.json', rows)
    save_json(folder / 'complete.json', {
        'checkpoint_lock_sha256': digest(output / 'checkpoints_locked.json'),
        'results_sha256': digest(folder / 'source_results.json')})
