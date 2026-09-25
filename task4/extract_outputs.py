"""cache model outputs after checkpoint locking, with known and final stages."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from task4.data.cifar10 import KnownImages, load_known, loader
from task4.models.resnet_cifar import ResNetCIFAR
from task4.utils import digest, save_json, verify_lock


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ResNetCIFAR()
    if checkpoint['config']['method'] == 'proser':
        model.add_dummy(checkpoint['config']['dummy_count'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()


@torch.inference_mode()
def collect(model, batches, device):
    features, logits, labels, indices = [], [], [], []
    for images, y, ids in tqdm(batches, desc='extracting outputs', mininterval=15, leave=False):
        f, z = model(images.to(device), return_features=True)
        features.append(f.cpu().numpy())
        logits.append(z.cpu().numpy())
        labels.append(y.numpy())
        indices.append(ids.numpy())
    return {name: np.concatenate(values) for name, values in
            [('features', features), ('logits', logits), ('labels', labels), ('indices', indices)]}


def extract(root, output, cache, device, stage='known'):
    output, cache = Path(output), Path(cache)
    plan, _ = verify_lock(output)
    cache.mkdir(parents=True, exist_ok=True)
    splits = json.loads((output / 'splits.json').read_text())
    if stage == 'known':
        base = load_known(root, train=True)
        datasets = {name: KnownImages(base, splits[name]) for name in ('train', 'val')}
    elif stage == 'final':
        # enforce calibration before even constructing the unknown dataset
        from task4.evaluation.calibration import verify_calibration
        verify_calibration(output)
        test = load_known(root, train=False)
        from task4.data.cifar100_unknowns import load_unknowns
        unknown, groups = load_unknowns(root)
        datasets = {'test': KnownImages(test, range(len(test))), **groups}
        save_json(output / 'unknown_manifest.json', {'classes': unknown.classes,
                  'near_indices': groups['near'].indices, 'far_indices': groups['far'].indices})
    else:
        raise ValueError('stage must be known or final')
    records = {}
    for name, config in plan['runs'].items():
        model = load_model(output / name / 'best.pt', device)
        for split, dataset in datasets.items():
            if split == 'train' and name != 'vanilla':
                continue
            values = collect(model, loader(dataset, config['eval_batch_size'], config['workers']), device)
            if not np.isfinite(values['features']).all() or not np.isfinite(values['logits']).all():
                raise FloatingPointError('non-finite cached model outputs')
            filename = f'{name}_{split}.npz'
            temporary = cache / (filename + '.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **values)
            temporary.replace(cache / filename)
            records[filename] = digest(cache / filename)
            print(f'cached {name}/{split}: {len(dataset)} images', flush=True)
    save_json(output / f'{stage}_outputs.json', {
        'checkpoint_lock_sha256': digest(output / 'checkpoints_locked.json'), 'files': records})


def verify_cache(output, cache, stage):
    output, cache = Path(output), Path(cache)
    manifest = json.loads((output / f'{stage}_outputs.json').read_text())
    if manifest['checkpoint_lock_sha256'] != digest(output / 'checkpoints_locked.json'):
        raise ValueError('cached outputs belong to different checkpoints')
    for filename, checksum in manifest['files'].items():
        if digest(cache / filename) != checksum:
            raise ValueError(f'cached outputs changed: {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', default='task4/results')
    parser.add_argument('--cache', default='task4/cache')
    parser.add_argument('--stage', choices=['known', 'final'], default='known')
    args = parser.parse_args()
    extract(args.data_root, args.output, args.cache, 'cuda' if torch.cuda.is_available() else 'cpu', args.stage)
