"""run the fixed protocol, including calibration before final evaluation."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

from task4.data.cifar10 import load_known
from task4.data.cifar100_unknowns import NEAR, FAR
from task4.data.make_splits import make_splits
from task4.evaluation.calibration import calibrate_all, verify_calibration
from task4.evaluate_osr import evaluate
from task4.extract_outputs import extract, verify_cache
from task4.train import train
from task4.utils import code_hash, configurations, digest, save_json, verify_lock


def run(root, output, cache, device):
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    base = load_known(root, train=True)
    splits = make_splits(base.targets, output / 'splits.json')
    plan = {'runs': configs, 'code_sha256': code_hash(), 'split_sha256': digest(output / 'splits.json'),
            'selection': 'maximum known validation accuracy; earliest tied epoch',
            'near_classes': NEAR, 'far_classes': FAR, 'unknown_partition': 'CIFAR100 test only',
            'mahalanobis': 'shared within-class diagonal variance / N + 1e-6; unaugmented train',
            'threshold': 'validation 95th percentile, linear; accept score <= threshold',
            'placeholder': 'max dummy; softmax T=1024; bias=0; dummy minus max known probability'}
    # normalize tuples before comparing a saved JSON plan on resume
    plan = json.loads(json.dumps(plan))
    path = output / 'experiment_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('existing experiment differs; restore its code/configuration or use a fresh output folder')
    if not path.exists():
        save_json(path, plan)
        save_json(output / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__, 'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
        (output / 'packages.txt').write_text(packages.stdout)
    for name, config in configs.items():
        parent = output / 'vanilla' / 'best.pt' if name == 'proser' else None
        train(config, base, splits, output / name, device, parent)
    origins = [json.loads((output / name / 'initialization.json').read_text())['initial_state_sha256']
               for name in ('vanilla', 'gcsc')]
    if origins[0] != origins[1]:
        raise ValueError('Vanilla and GCSC initializations differ')
    lock = {'plan_sha256': digest(path),
            'checkpoints': {name: digest(output / name / 'best.pt') for name in configs}}
    lock_path = output / 'checkpoints_locked.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError('fixed checkpoints changed')
    if not lock_path.exists():
        save_json(lock_path, lock)
    verify_lock(output)
    if (output / 'known_outputs.json').exists():
        verify_cache(output, cache, 'known')
    else:
        extract(root, output, cache, device, 'known')
    if (output / 'calibration_locked.json').exists():
        verify_calibration(output)
    else:
        calibrate_all(output, cache)
    if (output / 'final_outputs.json').exists():
        verify_cache(output, cache, 'final')
    else:
        extract(root, output, cache, device, 'final')
    evaluate(root, output, cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output', default='task4/results')
    parser.add_argument('--cache', default='task4/cache')
    args = parser.parse_args()
    run(args.data_root, args.output, args.cache, 'cuda' if torch.cuda.is_available() else 'cpu')
