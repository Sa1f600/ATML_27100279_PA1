"""the only task 3 stage that loads sketch, after all checkpoints are fixed."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.pacs import make_loader
from shared.pacs_protocol import CLASSES
from task2.evaluation.class_analysis import class_analysis
from task2.evaluate_final import plot_class_results, plot_examples
from task3.evaluation.domain_metrics import classification_metrics, collect
from task3.utils import digest, load_model, save_csv, save_json, verify_lock


def make_plots(output, plan, study):
    folder = output / 'final'
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in plan['runs']:
        with (output / name / 'history.csv').open() as file:
            history = list(csv.DictReader(file))
        for axis, key in zip(axes, ('classification_loss', 'alignment_loss', 'source_macro_f1')):
            axis.plot([int(row['epoch']) for row in history], [float(row[key]) for row in history], label=name)
            axis.set(xlabel='source epoch', ylabel=key.replace('_', ' '))
    axes[0].set_yscale('symlog', linthresh=1)
    axes[1].set_yscale('symlog', linthresh=0.1)
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / 'training_curves.png', dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ('source_mean_macro_f1', 'source_worst_macro_f1', 'target_accuracy'):
        axes[0].plot([row['radius'] for row in study], [row[key] for row in study], 'o-', label=key)
    axes[0].set(xlabel='SAM training radius', ylabel='score', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].plot([row['radius'] for row in study], [row['sharpness_delta'] for row in study], 'o-')
    axes[1].set(xlabel='SAM training radius', ylabel='sharpness at fixed diagnostic radius 0.05')
    figure.tight_layout()
    figure.savefig(folder / 'sam_study.png', dpi=180)
    plt.close(figure)


def task2_comparison(task2_results, output, rows, class_rows):
    # task 2 target metrics are opened only after task 3 checkpoints are fixed
    folder = Path(task2_results) / 'final'
    comparisons, per_class = [], []
    with (folder / 'all_results.csv').open() as file:
        old = {row['method']: row for row in csv.DictReader(file)}
    with (folder / 'per_class.csv').open() as file:
        old_classes = {(row['method'], row['class']): row for row in csv.DictReader(file)}
    for current, previous in [('erm', 'source_only'), ('dan_dg', 'dan')]:
        row = next(row for row in rows if row['method'] == current)
        comparisons.append({'task3_method': current, 'task2_method': previous,
                            'task3_target_accuracy': row['target_accuracy'],
                            'task2_target_accuracy': float(old[previous]['target_accuracy']),
                            'task3_minus_task2_pp': 100 * (row['target_accuracy'] - float(old[previous]['target_accuracy']))})
        for item in class_rows:
            if item['method'] == current:
                before = float(old_classes[(previous, item['class'])]['accuracy'])
                per_class.append({'task3_method': current, 'task2_method': previous,
                                  'class': item['class'], 'task3_accuracy': item['accuracy'],
                                  'task2_accuracy': before, 'difference_pp': 100 * (item['accuracy'] - before)})
    save_csv(output / 'final/task2_comparison.csv', comparisons)
    save_csv(output / 'final/task2_per_class_comparison.csv', per_class)
    save_json(output / 'final/task2_comparison_provenance.json', {
        name: digest(folder / name) for name in ('all_results.csv', 'per_class.csv')})


def evaluate(root, output, task2_results, device):
    output = Path(output)
    plan, _ = verify_lock(output)
    diagnostic = output / 'source_diagnostics'
    complete = json.loads((diagnostic / 'complete.json').read_text())
    if complete['checkpoint_lock_sha256'] != digest(output / 'checkpoints_locked.json'):
        raise ValueError('source diagnostics belong to different checkpoints')
    if complete['results_sha256'] != digest(diagnostic / 'source_results.json'):
        raise ValueError('source diagnostics changed')
    sources = {row['method']: row for row in json.loads((diagnostic / 'source_results.json').read_text())}
    paths = json.loads((output / 'splits.json').read_text())['target']
    if not paths or any(not p.startswith('sketch/') or '..' in Path(p).parts for p in paths):
        raise ValueError('unexpected target paths')
    folder = output / 'final'
    folder.mkdir(exist_ok=True)
    rows, class_rows, baseline = [], [], None
    for name, config in plan['runs'].items():
        backbone, classifier, _ = load_model(output / name / 'best.pt', device)
        values = collect(backbone, classifier, make_loader(
            root, paths, config['eval_batch_size'], config['seed'], config['workers']), device)
        predictions = values['logits'].argmax(1)
        if name == 'erm':
            baseline = predictions.copy()
        if baseline is None:
            raise ValueError('evaluate reused erm first')
        metrics = classification_metrics(values['labels'], predictions)
        row = sources[name] | {f'target_{key}': value for key, value in metrics.items()}
        row['target_delta_pp'] = 100 * (metrics['accuracy'] - np.mean(baseline == values['labels']))
        rows.append(row)
        items, matrix, examples = class_analysis(values['labels'], predictions, baseline, paths)
        class_rows.extend({'method': name, **item} for item in items)
        save_json(folder / f'{name}_confusion_matrix.json', {'classes': CLASSES, 'matrix': matrix.tolist()})
        save_json(folder / f'{name}_failure_cases.json', examples)
        np.savez_compressed(folder / f'{name}_target_outputs.npz', **values, paths=np.asarray(paths))
        plot_class_results(folder, name, items, matrix)
        plot_examples(folder, root, name, examples, items)
        print(f"{name}: final Sketch accuracy={metrics['accuracy']:.4f}", flush=True)
    save_csv(folder / 'all_results.csv', rows)
    save_csv(folder / 'main_comparison.csv', rows[:3])
    save_csv(folder / 'per_class.csv', class_rows)
    study = sorted([row | {'radius': plan['runs'][row['method']]['radius']}
                    for row in rows if plan['runs'][row['method']]['method'] == 'sam'], key=lambda r: r['radius'])
    save_csv(folder / 'sam_study.csv', study)
    save_json(folder / 'summary.json', {'results': rows, 'study': study, 'per_class': class_rows})
    task2_comparison(task2_results, output, rows, class_rows)
    make_plots(output, plan, study)
    save_json(folder / 'complete.json', {'checkpoint_lock_sha256': digest(output / 'checkpoints_locked.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='task3/results')
    parser.add_argument('--task2-results', required=True)
    args = parser.parse_args()
    evaluate(args.data_root, args.output, args.task2_results, 'cuda' if torch.cuda.is_available() else 'cpu')
