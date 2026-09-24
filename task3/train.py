"""source-only training for dan-dg and sam; erm is reused, never retrained."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from shared.pacs import cycle, make_loader
from shared.pacs_protocol import SOURCES
from task3.methods.dan_dg import pairwise_mmd
from task3.methods.erm import classification_loss
from task3.methods.sam import sam_step
from task3.models.backbone import freeze_bn_statistics, make_backbone
from task3.models.classifier_head import make_classifier
from task3.selection.source_validation import source_splits, validate, validation_loaders
from task3.utils import digest, save_csv, save_json, seed_everything


def train(config, root, manifest, output, device):
    sources = source_splits(manifest)
    if config['method'] not in {'dan_dg', 'sam'}:
        raise ValueError('only dan_dg and sam are trained; reuse the task 2 erm checkpoint')
    output = Path(output)
    if (output / 'complete.json').exists():
        complete = json.loads((output / 'complete.json').read_text())
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('completed run configuration differs')
        if digest(output / 'best.pt') != complete['checkpoint_sha256']:
            raise ValueError('completed checkpoint changed')
        print(f'already complete: {output.name}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'config.json', config)
    seed_everything(config['seed'])
    backbone, classifier = make_backbone().to(device), make_classifier().to(device)
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    loaders = {domain: make_loader(root, sources[domain]['train'], config['source_batch_size'],
                                   config['seed'] + i, config['workers'], training=True)
               for i, domain in enumerate(SOURCES)}
    streams = {domain: cycle(loader) for domain, loader in loaders.items()}
    validation = validation_loaders(root, sources, config)
    steps = math.ceil(sum(len(loader.dataset) for loader in loaders.values()) /
                      (3 * config['source_batch_size']))
    history, best, stale = [], -1.0, 0
    started = time.time()
    for epoch in range(config['max_epochs']):
        backbone.train()
        classifier.train()
        freeze_bn_statistics(backbone)
        totals = np.zeros(4)
        for step in tqdm(range(steps), desc=f'{output.name}: epoch {epoch + 1}',
                         mininterval=10, leave=False):
            # seed only cpu augmentation; do not reset accelerator randomness
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['seed'] + epoch * steps + step)
                batches = [next(streams[domain]) for domain in SOURCES]
            images = torch.cat([batch[0] for batch in batches]).to(device)
            labels = torch.cat([batch[1] for batch in batches]).to(device)
            if config['method'] == 'sam':
                def closure():
                    # both passes reuse the same augmented images and frozen bn statistics
                    return classification_loss(classifier(backbone(images)), labels)
                cls, perturbed = sam_step(closure, parameters, optimizer, config['radius'])
                totals += [cls, 0, cls, perturbed]
            else:
                features = backbone(images)
                cls = classification_loss(classifier(features), labels)
                alignment = pairwise_mmd(features, config['source_batch_size'])
                loss = cls + config['mmd_weight'] * alignment
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite dan-dg loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('non-finite dan-dg gradient')
                optimizer.step()
                totals += [cls.item(), alignment.item(), loss.item(), 0]
        scores, mean_f1 = validate(backbone, classifier, validation, device)
        row = dict(epoch=epoch + 1, classification_loss=totals[0] / steps,
                   alignment_loss=totals[1] / steps, total_loss=totals[2] / steps,
                   perturbed_loss=totals[3] / steps, source_macro_f1=mean_f1)
        row.update({f'{domain}_{metric}': value for domain, values in scores.items()
                    for metric, value in values.items()})
        history.append(row)
        save_csv(output / 'history.csv', history)
        if mean_f1 > best:
            best, stale = mean_f1, 0
            torch.save({'backbone': backbone.state_dict(), 'classifier': classifier.state_dict(),
                        'epoch': epoch + 1, 'source_scores': scores, 'config': config}, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
        else:
            stale += 1
        print(f'{output.name}: epoch {epoch + 1}, source macro-f1={mean_f1:.4f}', flush=True)
        if stale >= config['patience']:
            break
    save_json(output / 'complete.json', {'best_source_macro_f1': best, 'epochs_run': len(history),
              'steps_per_epoch': steps, 'seconds': time.time() - started,
              'checkpoint_sha256': digest(output / 'best.pt')})
