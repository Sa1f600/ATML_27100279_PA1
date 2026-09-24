"""task 3 experiment definitions and reproducibility helpers."""

import hashlib
import json
from pathlib import Path

import torch
import yaml

from task2.utils import REPO, digest, save_csv, save_json, seed_everything


def experiment_configs():
    folder = REPO / 'task3/configs'
    base = yaml.safe_load((folder / 'erm.yaml').read_text())
    runs = {'erm': base}
    for method in ('dan_dg', 'sam'):
        runs[method] = base | yaml.safe_load((folder / f'{method}.yaml').read_text())
    for radius in (0.01, 0.1):
        runs[f'sam_rho_{radius:g}'] = runs['sam'] | {'radius': radius}
    return runs


def code_hash():
    files = sorted(p for name in ('shared', 'task2', 'task3')
                   for p in (REPO / name).rglob('*') if p.suffix in {'.py', '.yaml'})
    return hashlib.sha256(''.join(str(p.relative_to(REPO)) + digest(p) for p in files).encode()).hexdigest()


def verify_lock(output):
    output = Path(output)
    plan = json.loads((output / 'experiment_plan.json').read_text())
    lock = json.loads((output / 'checkpoints_locked.json').read_text())
    if lock['plan_sha256'] != digest(output / 'experiment_plan.json'):
        raise ValueError('experiment plan changed after locking')
    if plan['code_sha256'] != code_hash():
        raise ValueError('code changed after locking; restore the original version')
    if plan['split_sha256'] != digest(output / 'splits.json'):
        raise ValueError('shared split changed after locking')
    if set(lock['checkpoints']) != set(plan['runs']):
        raise ValueError('every planned checkpoint must be fixed first')
    for name, checksum in lock['checkpoints'].items():
        if digest(output / name / 'best.pt') != checksum:
            raise ValueError(f'checkpoint changed: {name}')
    return plan, lock


def load_model(checkpoint_path, device):
    from task3.models.backbone import make_backbone
    from task3.models.classifier_head import make_classifier
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    backbone, classifier = make_backbone(pretrained=False).to(device), make_classifier().to(device)
    backbone.load_state_dict(checkpoint['backbone'])
    classifier.load_state_dict(checkpoint['classifier'])
    return backbone, classifier, checkpoint
