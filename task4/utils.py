"""reproducibility, configuration, and immutable experiment records."""

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

TASK = Path(__file__).resolve().parent


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, values):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(values, temporary)
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_hash():
    paths = sorted(p for p in TASK.rglob('*') if p.suffix in {'.py', '.yaml'})
    return hashlib.sha256(''.join(str(p.relative_to(TASK)) + digest(p) for p in paths).encode()).hexdigest()


def configurations():
    return {name: yaml.safe_load((TASK / 'configs' / f'{name}.yaml').read_text())
            for name in ('vanilla', 'gcsc', 'proser')}


def verify_lock(output):
    output = Path(output)
    plan = json.loads((output / 'experiment_plan.json').read_text())
    lock = json.loads((output / 'checkpoints_locked.json').read_text())
    if lock['plan_sha256'] != digest(output / 'experiment_plan.json') or plan['code_sha256'] != code_hash():
        raise ValueError('experiment code or plan changed; restore the original version')
    if plan['split_sha256'] != digest(output / 'splits.json'):
        raise ValueError('split changed')
    if set(lock['checkpoints']) != set(plan['runs']):
        raise ValueError('all three models must be fixed before extraction')
    for name, checksum in lock['checkpoints'].items():
        if checksum != digest(output / name / 'best.pt'):
            raise ValueError(f'checkpoint changed: {name}')
    return plan, lock
