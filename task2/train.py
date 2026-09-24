"""one source-selected training loop for all task 2 methods."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from shared.pacs import cycle, make_loader
from shared.pacs_protocol import SOURCES
from task2.evaluation.metrics import classification_metrics, collect
from task2.methods.cdan import conditional_loss
from task2.methods.dan import mmd_loss
from task2.methods.dann import domain_loss
from task2.methods.source_only import classification_loss
from task2.models.backbone import freeze_bn_statistics, make_backbone
from task2.models.classifier_head import make_classifier
from task2.models.domain_discriminator import make_discriminator, schedule
from task2.utils import digest, save_csv, save_json, seed_everything


def train(config, root, splits, output, device):
    output = Path(output)
    if (output / 'complete.json').exists():
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError(f"configuration mismatch in {output}")
        completion = json.loads((output / 'complete.json').read_text())
        if digest(output / 'best.pt') != completion['checkpoint_sha256']:
            raise ValueError(f"saved checkpoint changed in {output}")
        print(f"already complete: {output.name}")
        return
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'config.json', config)
    seed_everything(config['seed'])
    backbone = make_backbone().to(device)
    classifier = make_classifier().to(device)
    method = config['method']
    discriminator = None
    if method in {'dann', 'cdan'}:
        discriminator = make_discriminator(512 if method == 'dann' else 512 * 7).to(device)
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    if discriminator is not None:
        parameters += list(discriminator.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    loaders, validation = {}, {}
    for index, domain in enumerate(SOURCES):
        loaders[domain] = make_loader(root, splits['sources'][domain]['train'],
                                     config['source_batch_size'], config['seed'] + index,
                                     config['workers'], training=True)
        validation[domain] = make_loader(root, splits['sources'][domain]['val'],
                                        config['eval_batch_size'], config['seed'],
                                        config['workers'])
    streams = {name: cycle(loader) for name, loader in loaders.items()}
    target = None
    if method != 'source_only':
        target = cycle(make_loader(root, splits['target'], config['target_batch_size'],
                                   config['seed'] + 10, config['workers'],
                                   training=True, labeled=False))
    # one source epoch equals ceil(total source training images / 24) balanced updates
    source_size = sum(len(loader.dataset) for loader in loaders.values())
    steps = math.ceil(source_size / (3 * config['source_batch_size']))
    budget = config['max_epochs'] * steps
    best, stale, history = -1.0, 0, []
    started = time.time()
    for epoch in range(config['max_epochs']):
        backbone.train()
        classifier.train()
        freeze_bn_statistics(backbone)
        if discriminator is not None:
            discriminator.train()
        totals = np.zeros(3)
        for step in tqdm(range(steps), desc=f'{output.name}: epoch {epoch + 1}', leave=False):
            update = epoch * steps + step
            # isolate crop/flip randomness even when workers=0 and dropout consumes rng
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config['seed'] + update)
                batches = [next(streams[domain]) for domain in SOURCES]
            images = torch.cat([batch[0] for batch in batches]).to(device)
            labels = torch.cat([batch[1] for batch in batches]).to(device)
            count = len(labels)
            if target is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(config['seed'] + 100000 + update)
                    target_images, _ = next(target)
                images = torch.cat([images, target_images.to(device)])
            features = backbone(images)
            logits = classifier(features)
            cls = classification_loss(logits[:count], labels)
            strength = config['max_grl'] * schedule(update / max(budget - 1, 1))
            alignment = cls.new_zeros(())
            weight = 1.0
            if method == 'dan':
                alignment = mmd_loss(features[:count], features[count:])
                weight = config['mmd_weight']
            elif method == 'dann':
                alignment = domain_loss(features, count, discriminator, strength)
            elif method == 'cdan':
                alignment = conditional_loss(features, logits, count, discriminator, strength)
            loss = cls + weight * alignment
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss in {output.name}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += [cls.item(), alignment.item(), loss.item()]
        scores = {}
        for domain, loader in validation.items():
            values = collect(backbone, classifier, loader, device)
            scores[domain] = classification_metrics(values['labels'], values['logits'].argmax(1))
        mean_f1 = float(np.mean([score['macro_f1'] for score in scores.values()]))
        row = dict(epoch=epoch + 1, classification_loss=totals[0] / steps,
                   alignment_loss=totals[1] / steps, total_loss=totals[2] / steps,
                   grl_strength=strength, source_macro_f1=mean_f1)
        for domain, score in scores.items():
            row.update({f'{domain}_{key}': value for key, value in score.items()})
        history.append(row)
        save_csv(output / 'history.csv', history)
        if mean_f1 > best:
            best, stale = mean_f1, 0
            # only source-validation macro-f1 chooses the checkpoint; ties keep the first
            checkpoint = {'backbone': backbone.state_dict(), 'classifier': classifier.state_dict(),
                          'discriminator': discriminator.state_dict() if discriminator is not None else None,
                          'epoch': epoch + 1, 'source_scores': scores, 'config': config}
            torch.save(checkpoint, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
        else:
            stale += 1
        print(f'{output.name}: epoch {epoch + 1}, source macro-f1={mean_f1:.4f}', flush=True)
        if stale >= config['patience']:
            break
    save_json(output / 'complete.json', {'best_source_macro_f1': best,
              'epochs_run': len(history), 'steps_per_epoch': steps,
              'seconds': time.time() - started, 'checkpoint_sha256': digest(output / 'best.pt')})
