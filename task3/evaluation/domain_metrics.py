"""per-source, mean-source, and worst-source metrics."""

import numpy as np

from task2.evaluation.metrics import classification_metrics, collect


def source_summary(scores):
    row = {f'{domain}_{metric}': value for domain, values in scores.items()
           for metric, value in values.items()}
    for metric in ('accuracy', 'macro_f1'):
        values = [score[metric] for score in scores.values()]
        row[f'source_mean_{metric}'] = float(np.mean(values))
        row[f'source_worst_{metric}'] = float(min(values))
    return row
