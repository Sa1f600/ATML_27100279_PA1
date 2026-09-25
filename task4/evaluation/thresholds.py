"""calibrate on known validation scores only; ties are accepted."""

import numpy as np


def calibrate(validation_scores):
    values = np.asarray(validation_scores)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('expected finite known-validation scores')
    return float(np.quantile(values, 0.95, method='linear'))


def accepted(scores, threshold):
    return np.asarray(scores) <= threshold
