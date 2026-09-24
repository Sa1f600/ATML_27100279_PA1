"""lock the experiment, train six runs, then release target labels for analysis."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

from shared.pacs_protocol import SPLIT_FILE, make_splits
from task2.evaluate_final import evaluate
from task2.train import train
from task2.utils import code_hash, digest, experiment_configs, save_json

HYPOTHESIS = ('Increasing MMD weight should make domains harder to distinguish, but excessive '
              'alignment may reduce source class discrimination and target recognition. '
              'Target accuracy need not improve monotonically. All three strengths are fixed '
              'before target evaluation; source-validation macro-F1 alone could select a strength.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='folder containing photo/art_painting/cartoon/sketch')
    parser.add_argument('--output', default='task2/results')
    parser.add_argument('--split-file', default=str(SPLIT_FILE))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = make_splits(args.data_root, args.split_file)
    if (output / 'splits.json').exists():
        if json.loads((output / 'splits.json').read_text()) != splits:
            raise ValueError('results directory belongs to a different split')
    else:
        save_json(output / 'splits.json', splits)
    plan = {'runs': experiment_configs(), 'hypothesis': HYPOTHESIS,
            'code_sha256': code_hash(), 'split_sha256': digest(output / 'splits.json'),
            'selection': 'mean source-validation macro-F1; strict improvement; earliest tie',
            'source_epoch': 'ceil(total source training images / 24) balanced updates',
            'mmd_estimator': 'biased V-statistic, sum of exp(-squared_distance/(2*bandwidth))',
            'probe': 'balanced source-val/target, stratified 70/30, train-only scaling, C=1'}
    plan_file = output / 'experiment_plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError('locked experiment differs; restore the original code/configuration')
    else:
        save_json(plan_file, plan)
    environment = output / 'environment.json'
    if not environment.exists():
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        save_json(environment, {'python': platform.python_version(), 'torch': torch.__version__,
                  'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                  'git_commit': commit.stdout.strip()})
        with (output / 'pip_freeze.txt').open('w') as file:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=file, check=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('warning: full PACS training is intended for a Colab GPU', flush=True)
    lock_file = output / 'checkpoints_locked.json'
    if not lock_file.exists():
        for name, config in plan['runs'].items():
            train(config, args.data_root, splits, output / name, device)
        # all checkpoints are fingerprinted before any target labels are evaluated
        save_json(lock_file, {'plan_sha256': digest(plan_file), 'checkpoints': {
            name: digest(output / name / 'best.pt') for name in plan['runs']}})
    evaluate(args.data_root, output, device)
    print(f'finished: {output / "final/main_comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
