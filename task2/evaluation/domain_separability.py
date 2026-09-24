"""a held-out, balanced linear probe for residual domain information."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def domain_separability(source, target, seed=6304):
    rng = np.random.default_rng(seed)
    n = min(len(source), len(target))
    source_ids = rng.choice(len(source), n, replace=False)
    target_ids = rng.choice(len(target), n, replace=False)
    features = np.concatenate([source[source_ids], target[target_ids]])
    labels = np.repeat([0, 1], n)
    train, test = train_test_split(np.arange(2 * n), test_size=0.3,
                                  stratify=labels, random_state=seed)
    # scaling is fitted only on the probe training partition
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, class_weight="balanced", max_iter=3000, random_state=seed))
    probe.fit(features[train], labels[train])
    return {"accuracy": float(probe.score(features[test], labels[test])),
            "source_indices": source_ids.tolist(), "target_indices": target_ids.tolist(),
            "train_indices": train.tolist(), "test_indices": test.tolist(),
            "C": 1, "standardized": True, "max_iter": 3000,
            "iterations": probe[-1].n_iter_.tolist()}
