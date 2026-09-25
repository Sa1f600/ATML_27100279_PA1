"""fixed 90/10 stratified split of official CIFAR-10 training indices."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from task4.utils import save_json


def make_splits(labels, path, seed=6304):
    train, val = train_test_split(np.arange(len(labels)), test_size=0.1,
                                  stratify=labels, random_state=seed)
    result = {'seed': seed, 'train': sorted(train.tolist()), 'val': sorted(val.tolist())}
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('existing CIFAR-10 split differs')
    if not path.exists():
        save_json(path, result)
    return result
