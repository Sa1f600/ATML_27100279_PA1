"""fix every score threshold using only known training and validation outputs."""

import json
from pathlib import Path

import numpy as np

from task4.scores import energy, mahalanobis, mls, msp, placeholder
from task4.evaluation.thresholds import calibrate
from task4.utils import digest, save_json, verify_lock

SCORES = {'vanilla': ('msp', 'mls', 'energy', 'mahalanobis'),
          'gcsc': ('mls',), 'proser': ('mls', 'placeholder')}


def scores_for(name, outputs, parameters):
    if name == 'mahalanobis':
        return mahalanobis.score(outputs['features'], parameters)
    functions = {'msp': msp.score, 'mls': mls.score, 'energy': energy.score, 'placeholder': placeholder.score}
    return functions[name](outputs['logits'])


def calibrate_all(output, cache):
    from task4.extract_outputs import verify_cache
    output, cache = Path(output), Path(cache)
    verify_lock(output)
    verify_cache(output, cache, 'known')
    with np.load(cache / 'vanilla_train.npz') as train:
        parameters = mahalanobis.fit(train['features'], train['labels'])
    np.savez_compressed(output / 'mahalanobis_parameters.npz', **parameters)
    records = {}
    for model, names in SCORES.items():
        with np.load(cache / f'{model}_val.npz') as validation:
            for name in names:
                values = scores_for(name, validation, parameters)
                threshold = calibrate(values)
                records[f'{model}/{name}'] = {'threshold': threshold,
                    'validation_acceptance': float((values <= threshold).mean()), 'validation_count': len(values)}
    save_json(output / 'thresholds.json', {'scores': records, 'quantile': 0.95, 'quantile_method': 'linear',
              'acceptance_rule': 'score <= threshold', 'placeholder_temperature': 1024., 'placeholder_bias': 0.})
    save_json(output / 'calibration_locked.json', {
        'checkpoint_lock_sha256': digest(output / 'checkpoints_locked.json'),
        'thresholds_sha256': digest(output / 'thresholds.json'),
        'mahalanobis_sha256': digest(output / 'mahalanobis_parameters.npz')})


def verify_calibration(output):
    output = Path(output)
    verify_lock(output)
    lock = json.loads((output / 'calibration_locked.json').read_text())
    for key, path in [('checkpoint_lock_sha256', 'checkpoints_locked.json'),
                      ('thresholds_sha256', 'thresholds.json'),
                      ('mahalanobis_sha256', 'mahalanobis_parameters.npz')]:
        if lock[key] != digest(output / path):
            raise ValueError(f'calibration input changed: {path}')
    return json.loads((output / 'thresholds.json').read_text())
