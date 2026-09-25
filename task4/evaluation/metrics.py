"""unknowns are positive for AUROC; FPR is the unknown acceptance rate."""

import numpy as np
from sklearn.metrics import roc_auc_score

from task4.evaluation.thresholds import accepted


def osr_metrics(known, near, far, threshold):
    result = {'threshold': threshold, 'known_acceptance': float(accepted(known, threshold).mean())}
    for name, unknown in [('near', near), ('far', far), ('all', np.concatenate([near, far]))]:
        labels = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
        result[f'{name}_auroc'] = float(roc_auc_score(labels, np.concatenate([known, unknown])))
        result[f'{name}_rejection'] = float((~accepted(unknown, threshold)).mean())
        result[f'{name}_fpr_at_val95tpr'] = float(accepted(unknown, threshold).mean())
    return result
