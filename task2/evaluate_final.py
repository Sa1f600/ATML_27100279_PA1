"""final-only target evaluation after every checkpoint has been locked."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from shared.pacs import make_loader
from shared.pacs_protocol import CLASSES, SOURCES
from task2.evaluation.class_analysis import class_analysis
from task2.evaluation.domain_separability import domain_separability
from task2.evaluation.metrics import classification_metrics, collect
from task2.models.backbone import make_backbone
from task2.models.classifier_head import make_classifier
from task2.utils import code_hash, digest, save_csv, save_json


def plot_losses(output, names):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        with (output / name / 'history.csv').open() as file:
            history = list(csv.DictReader(file))
        for axis, key in zip(axes, ['classification_loss', 'alignment_loss']):
            axis.plot([int(row['epoch']) for row in history],
                      [float(row[key]) for row in history], label=name)
            axis.set(xlabel='source epoch', ylabel=key.replace('_', ' '))
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'final/loss_curves.png', dpi=180)
    plt.close(figure)


def plot_class_results(folder, name, rows, matrix):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(CLASSES, [row['delta_pp'] for row in rows])
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].set(ylabel='accuracy change vs source-only (pp)', title=name)
    axes[0].tick_params(axis='x', rotation=45)
    axes[1].imshow(matrix, cmap='Blues')
    axes[1].set(xticks=range(7), yticks=range(7), xticklabels=CLASSES,
                yticklabels=CLASSES, xlabel='predicted class', ylabel='true class')
    axes[1].tick_params(axis='x', rotation=45)
    for i in range(7):
        for j in range(7):
            axes[1].text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7,
                         color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    figure.tight_layout()
    figure.savefig(folder / f'{name}_classes.png', dpi=180)
    plt.close(figure)


def plot_examples(folder, root, name, examples, rows):
    # focus the contact sheet on the largest class gain and degradation
    selected_classes = {max(rows, key=lambda row: row['delta_pp'])['class'],
                        min(rows, key=lambda row: row['delta_pp'])['class']}
    chosen = [e for e in examples if e['true_class'] in selected_classes
              and e['kind'] in {'gained', 'lost'}][:6]
    if not chosen:
        chosen = [e for e in examples if e['true_class'] in selected_classes][:6]
    if not chosen:
        return
    figure, axes = plt.subplots(1, len(chosen), figsize=(3 * len(chosen), 3), squeeze=False)
    for axis, example in zip(axes[0], chosen):
        with Image.open(Path(root) / example['path']) as image:
            axis.imshow(image.convert('RGB'))
        axis.set_title(f"{example['kind']}: {example['true_class']}\n"
                       f"prediction: {example['predicted_class']}\n"
                       f"baseline: {example['source_only_prediction']}", fontsize=9)
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(folder / f'{name}_examples.png', dpi=160)
    plt.close(figure)


def evaluate(root, output, device):
    output = Path(output)
    frozen = json.loads((output / 'checkpoints_locked.json').read_text())
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if frozen['plan_sha256'] != digest(output / 'experiment_plan.json'):
        raise ValueError('experiment plan changed after checkpoint locking')
    if plan['code_sha256'] != code_hash():
        raise ValueError('code changed after experiment locking')
    if plan['split_sha256'] != digest(output / 'splits.json'):
        raise ValueError('split changed after experiment locking')
    if set(frozen['checkpoints']) != set(plan['runs']):
        raise ValueError('not all planned models have been fixed')
    for name, checksum in frozen['checkpoints'].items():
        if digest(output / name / 'best.pt') != checksum:
            raise ValueError(f'checkpoint changed: {name}')
    splits = json.loads((output / 'splits.json').read_text())
    folder = output / 'final'
    folder.mkdir(exist_ok=True)
    source_paths = [p for domain in SOURCES for p in splits['sources'][domain]['val']]
    baseline, summary, class_rows = None, [], []
    for name, config in plan['runs'].items():
        checkpoint = torch.load(output / name / 'best.pt', map_location='cpu', weights_only=True)
        backbone, classifier = make_backbone(pretrained=False).to(device), make_classifier().to(device)
        backbone.load_state_dict(checkpoint['backbone'])
        classifier.load_state_dict(checkpoint['classifier'])
        row = {'method': name, 'selected_epoch': checkpoint['epoch']}
        source_features, source_scores = [], []
        for domain in SOURCES:
            values = collect(backbone, classifier, make_loader(
                root, splits['sources'][domain]['val'], config['eval_batch_size'],
                config['seed'], config['workers']), device)
            score = classification_metrics(values['labels'], values['logits'].argmax(1))
            source_scores.append(score)
            source_features.append(values['features'])
            row.update({f'{domain}_{key}': value for key, value in score.items()})
            np.savez_compressed(folder / f'{name}_{domain}_outputs.npz', **values,
                                paths=np.asarray(splits['sources'][domain]['val']))
        row.update({f'source_mean_{key}': float(np.mean([s[key] for s in source_scores]))
                    for key in ('accuracy', 'macro_f1')})
        # this is the first stage allowed to decode target class labels
        target = collect(backbone, classifier, make_loader(
            root, splits['target'], config['eval_batch_size'], config['seed'],
            config['workers'], labeled=True), device)
        predictions = target['logits'].argmax(1)
        if name == 'source_only':
            baseline = predictions.copy()
        if baseline is None:
            raise ValueError('source_only must be evaluated first')
        score = classification_metrics(target['labels'], predictions)
        row.update({f'target_{key}': value for key, value in score.items()})
        row['target_delta_pp'] = 100 * (score['accuracy'] - np.mean(baseline == target['labels']))
        row['source_target_gap_pp'] = 100 * (row['source_mean_accuracy'] - score['accuracy'])
        probe = domain_separability(np.concatenate(source_features), target['features'], config['seed'])
        row['domain_separability'] = probe['accuracy']
        save_json(folder / f'{name}_domain_probe.json', probe | {'source_paths': source_paths})
        np.savez_compressed(folder / f'{name}_target_outputs.npz', **target,
                            paths=np.asarray(splits['target']))
        rows, matrix, examples = class_analysis(target['labels'], predictions, baseline, splits['target'])
        class_rows.extend({'method': name, **item} for item in rows)
        save_json(folder / f'{name}_failure_cases.json', examples)
        save_json(folder / f'{name}_confusion_matrix.json', {'classes': CLASSES, 'matrix': matrix.tolist()})
        plot_class_results(folder, name, rows, matrix)
        plot_examples(folder, root, name, examples, rows)
        summary.append(row)
        print(f"{name}: target accuracy={score['accuracy']:.4f}, separability={probe['accuracy']:.4f}")
    save_csv(folder / 'all_results.csv', summary)
    save_csv(folder / 'main_comparison.csv', summary[:4])
    save_csv(folder / 'per_class.csv', class_rows)
    study = [dict(mmd_weight=plan['runs'][row['method']]['mmd_weight'], **row)
             for row in summary if plan['runs'][row['method']]['method'] == 'dan']
    study.sort(key=lambda row: row['mmd_weight'])
    save_csv(folder / 'alignment_study.csv', study)
    save_json(folder / 'summary.json', {'results': summary, 'per_class': class_rows, 'study': study})
    figure, axis = plt.subplots(figsize=(6, 4))
    for key in ['source_mean_macro_f1', 'target_accuracy', 'domain_separability']:
        axis.plot([row['mmd_weight'] for row in study], [row[key] for row in study], 'o-', label=key)
    axis.set(xscale='log', xlabel='MMD weight', ylabel='score', ylim=(0, 1))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / 'alignment_study.png', dpi=180)
    plt.close(figure)
    plot_losses(output, plan['runs'])
    save_json(folder / 'complete.json', {'checkpoint_lock_sha256': digest(output / 'checkpoints_locked.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='task2/results')
    args = parser.parse_args()
    evaluate(args.data_root, args.output, 'cuda' if torch.cuda.is_available() else 'cpu')
