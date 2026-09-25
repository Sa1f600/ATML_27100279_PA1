"""class means and pooled within-class diagonal covariance from training only."""

import numpy as np


def fit(features, labels):
    features, labels = np.asarray(features, dtype=np.float64), np.asarray(labels)
    if set(labels.tolist()) != set(range(10)):
        raise ValueError('Mahalanobis fitting requires all ten training classes')
    means = np.stack([features[labels == label].mean(0) for label in range(10)])
    residual = features - means[labels]
    # use maximum-likelihood pooled within-class variance with the prescribed floor
    variance = np.mean(residual ** 2, axis=0) + 1e-6
    return {'means': means, 'variance': variance}


def score(features, parameters):
    result = []
    for start in range(0, len(features), 1024):
        x = np.asarray(features[start:start + 1024], dtype=np.float64)
        distances = ((x[:, None, :] - parameters['means'][None, :, :]) ** 2 / parameters['variance']).sum(2)
        result.append(distances.min(1))
    return np.concatenate(result)
