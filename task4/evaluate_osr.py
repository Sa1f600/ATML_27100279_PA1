"""final metrics and report evidence from immutable cached outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from task4.evaluation.calibration import SCORES, scores_for, verify_calibration
from task4.evaluation.failure_analysis import failures
from task4.evaluation.metrics import osr_metrics
from task4.extract_outputs import verify_cache
from task4.utils import digest, save_csv, save_json


def read_outputs(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def evaluate(root, output, cache):
    output, cache = Path(output), Path(cache)
    calibration = verify_calibration(output)
    for stage in ('known', 'final'):
        verify_cache(output, cache, stage)
    parameters = read_outputs(output / 'mahalanobis_parameters.npz')
    rows, all_scores, per_class = [], {}, []
    vanilla_outputs = None
    reference_ids = {}
    for model, names in SCORES.items():
        datasets = {split: read_outputs(cache / f'{model}_{split}.npz')
                    for split in ('val', 'test', 'near', 'far')}
        # every model and score must receive the identical ordered evaluation examples
        for split, data in datasets.items():
            if split in reference_ids:
                if not np.array_equal(data['indices'], reference_ids[split]):
                    raise ValueError('evaluation sample mismatch')
            else:
                reference_ids[split] = data['indices']
        test = datasets['test']
        accuracy = float((test['logits'][:, :10].argmax(1) == test['labels']).mean())
        for name in names:
            values = {split: scores_for(name, data, parameters) for split, data in datasets.items()}
            record = calibration['scores'][f'{model}/{name}']
            threshold = record['threshold']
            rows.append({'model': model, 'score': name, 'csa': accuracy,
                         'validation_acceptance': record['validation_acceptance'],
                         **osr_metrics(values['test'], values['near'], values['far'], threshold)})
            for split, scores in values.items():
                all_scores[f'{model}_{name}_{split}'] = scores
            for group in ('near', 'far'):
                for label in np.unique(datasets[group]['labels']):
                    mask = datasets[group]['labels'] == label
                    per_class.append({'model': model, 'score': name, 'group': group,
                        'cifar100_class_id': int(label), 'count': int(mask.sum()),
                        'rejection': float((values[group][mask] > threshold).mean())})
        if model == 'vanilla':
            vanilla_outputs = datasets
    save_csv(output / 'posthoc_comparison.csv', [r for r in rows if r['model'] == 'vanilla'])
    save_csv(output / 'trained_model_comparison.csv', [r for r in rows if r['score'] in ('mls', 'placeholder')])
    save_csv(output / 'all_results.csv', rows)
    save_json(output / 'all_results.json', rows)
    np.savez_compressed(output / 'scores.npz', **all_scores)
    # roc plots use unknowns as positives, with no outcome-dependent score flipping
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name in zip(axes, ('msp', 'mls', 'mahalanobis')):
        known = all_scores[f'vanilla_{name}_test']
        for group in ('near', 'far'):
            unknown = all_scores[f'vanilla_{name}_{group}']
            fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(known)), np.ones(len(unknown))], np.r_[known, unknown])
            axis.plot(fpr, tpr, label=group)
        axis.plot([0, 1], [0, 1], '--', color='gray')
        axis.set(title=name.upper(), xlabel='known incorrectly rejected', ylabel='unknown correctly rejected')
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'vanilla_roc.png', dpi=180)
    plt.close(figure)
    from task4.data.cifar100_unknowns import load_unknowns
    unknown_base, _ = load_unknowns(root)
    for row in per_class:
        row['unknown_class'] = unknown_base.classes[row['cifar100_class_id']]
    save_csv(output / 'unknown_per_class.csv', per_class)
    failures(unknown_base, vanilla_outputs,
             {g: all_scores[f'vanilla_mls_{g}'] for g in ('near', 'far')},
             calibration['scores']['vanilla/mls']['threshold'], output)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model in SCORES:
        with (output / model / 'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row['epoch']) for row in history]
        for axis, key in zip(axes, ('loss', 'validation_accuracy')):
            axis.plot(epochs, [float(row[key]) for row in history], label=model)
            axis.set(xlabel='epoch', ylabel=key.replace('_', ' '))
            axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'training_curves.png', dpi=180)
    plt.close(figure)
    save_json(output / 'evaluation_complete.json', {
        'calibration_lock_sha256': digest(output / 'calibration_locked.json'),
        'results_sha256': digest(output / 'all_results.json'), 'rows': len(rows)})
    print('Task 4 complete. Results:', output, flush=True)
    for row in rows:
        print(f"{row['model']}/{row['score']}: CSA={row['csa']:.4f}, "
              f"near AUROC={row['near_auroc']:.4f}, far AUROC={row['far_auroc']:.4f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='task4/results')
    parser.add_argument('--cache', default='task4/cache')
    args = parser.parse_args()
    evaluate(args.data_root, args.output, args.cache)
