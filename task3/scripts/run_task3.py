"""reuse erm, train fixed source-only methods, diagnose, then evaluate sketch."""

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torchvision

from shared.pacs_protocol import SPLIT_FILE
from task3.evaluation.source_diagnostics import diagnose
from task3.selection.source_validation import source_splits
from task3.train import train
from task3.utils import code_hash, digest, experiment_configs, save_json, verify_lock

HYPOTHESIS = ('Increasing SAM radius is expected to reduce the common local sharpness proxy, '
              'but a large radius may reduce source accuracy; Sketch recognition may be '
              'non-monotonic. The three prescribed radii are fixed before training. '
              'Source validation alone selects checkpoints, and the main SAM radius remains 0.05.')


def prepare(output, task2_results, split_file):
    output, previous = Path(output), Path(task2_results)
    manifest = json.loads(Path(split_file).read_text())
    source_splits(manifest)
    if digest(split_file) != digest(previous / 'splits.json'):
        raise ValueError('shared split differs from the actual task 2 run')
    baseline = previous / 'source_only'
    completion = json.loads((baseline / 'complete.json').read_text())
    checksum = digest(baseline / 'best.pt')
    if checksum != completion['checkpoint_sha256']:
        raise ValueError('task 2 erm checkpoint differs from its recorded hash')
    checkpoint = torch.load(baseline / 'best.pt', map_location='cpu', weights_only=True)
    original_config = json.loads((baseline / 'config.json').read_text())
    if checkpoint['config'] != original_config or original_config['method'] != 'source_only':
        raise ValueError('expected the original source-only checkpoint')
    runs = experiment_configs()
    for key in ('seed', 'max_epochs', 'patience', 'learning_rate', 'weight_decay', 'source_batch_size'):
        if runs['erm'][key] != original_config[key]:
            raise ValueError(f'task 3 must reuse task 2 setting: {key}')
    with (baseline / 'history.csv').open() as file:
        history = list(csv.DictReader(file))
    best = max(history, key=lambda row: float(row['source_macro_f1']))
    if int(best['epoch']) != checkpoint['epoch']:
        raise ValueError('erm is not the source-selected checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    plan = {'runs': runs, 'hypothesis': HYPOTHESIS, 'code_sha256': code_hash(),
            'split_sha256': digest(split_file), 'erm_checkpoint_sha256': checksum,
            'selection': 'mean source-validation macro-F1; strict improvement; earliest tie',
            'source_epoch': 'ceil(total source training images / 24) balanced updates',
            'study': 'SAM radius 0.01, 0.05, 0.1; fixed independently of task 2 target results',
            'sharpness': {'radius': 0.05, 'per_domain': 32, 'seed': 6304},
            'probe': {'C': 1, 'train_fraction': 0.7, 'standardized': True, 'seed': 6304}}
    plan_file = output / 'experiment_plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise ValueError('locked experiment differs; restore its original code and settings')
    else:
        save_json(plan_file, plan)
    # retain identical bytes so the shared split fingerprint remains valid
    pairs = [(Path(split_file), output / 'splits.json')]
    pairs += [(baseline / name, output / 'erm' / name)
              for name in ('best.pt', 'history.csv', 'config.json', 'complete.json')]
    for source, destination in pairs:
        if destination.exists():
            if digest(source) != digest(destination):
                raise ValueError(f'existing file differs: {destination}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    save_json(output / 'erm/reuse.json', {'source': str(baseline), 'checkpoint_sha256': checksum,
              'retrained': False, 'task3_name': 'erm', 'original_method': 'source_only'})
    return manifest, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--task2-results', required=True)
    parser.add_argument('--split-file', default=str(SPLIT_FILE))
    parser.add_argument('--output', default='task3/results')
    args = parser.parse_args()
    output = Path(args.output)
    manifest, plan = prepare(output, args.task2_results, args.split_file)
    if not (output / 'environment.json').exists():
        save_json(output / 'environment.json', {'python': platform.python_version(),
                  'torch': torch.__version__, 'torchvision': torchvision.__version__,
                  'cuda': torch.version.cuda,
                  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})
        with (output / 'pip_freeze.txt').open('w') as file:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=file, check=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not (output / 'checkpoints_locked.json').exists():
        for name, config in plan['runs'].items():
            if name != 'erm':
                train(config, args.data_root, manifest, output / name, device)
        save_json(output / 'checkpoints_locked.json', {
            'plan_sha256': digest(output / 'experiment_plan.json'),
            'checkpoints': {name: digest(output / name / 'best.pt') for name in plan['runs']}})
    verify_lock(output)
    diagnose(args.data_root, output, device)
    # target evaluation is a separate module reached only after source work is finished
    from task3.evaluate_sketch import evaluate
    evaluate(args.data_root, output, args.task2_results, device)
    print(f'finished: {output / "final/main_comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
