"""balanced three-way source-domain probe; chance accuracy is one third."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.pacs_protocol import SOURCES


def source_domain_separability(features, seed=6304):
    rng = np.random.default_rng(seed)
    count = min(len(features[domain]) for domain in SOURCES)
    indices = {domain: rng.choice(len(features[domain]), count, replace=False)
               for domain in SOURCES}
    x = np.concatenate([features[domain][indices[domain]] for domain in SOURCES])
    y = np.repeat(np.arange(3), count)
    train, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                  stratify=y, random_state=seed)
    # lbfgs uses multinomial logistic regression for the three-class problem
    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, solver='lbfgs', class_weight='balanced', max_iter=3000, random_state=seed))
    probe.fit(x[train], y[train])
    return {'accuracy': float(probe.score(x[test], y[test])), 'C': 1,
            'standardized': True, 'seed': seed, 'solver': 'lbfgs', 'max_iter': 3000,
            'domain_order': list(SOURCES), 'chance': 1 / 3,
            'sample_indices': {name: values.tolist() for name, values in indices.items()},
            'train_indices': train.tolist(), 'test_indices': test.tolist(),
            'iterations': probe[-1].n_iter_.tolist()}
