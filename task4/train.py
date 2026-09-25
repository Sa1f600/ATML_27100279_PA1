"""CIFAR-10-only training with validation selection and epoch-level resume."""

import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from task4.data.cifar10 import KnownImages, loader
from task4.methods import gcsc, proser, vanilla
from task4.models.resnet_cifar import ResNetCIFAR
from task4.utils import digest, save_checkpoint, save_csv, save_json, seed_everything


@torch.inference_mode()
def validate(model, batches, device):
    model.eval()
    correct, count = 0, 0
    for images, labels, _ in batches:
        predictions = model(images.to(device))[:, :10].argmax(1).cpu()
        correct += int((predictions == labels).sum())
        count += len(labels)
    return correct / count


def initialize(config, vanilla_checkpoint=None):
    seed_everything(config['seed'])
    model = ResNetCIFAR()
    parent = None
    if config['method'] == 'proser':
        if vanilla_checkpoint is None:
            raise ValueError('PROSER requires the selected Vanilla checkpoint')
        checkpoint = torch.load(vanilla_checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        parent = digest(vanilla_checkpoint)
        # preserve every learned vanilla parameter, then create five new classifiers
        model.add_dummy(config['dummy_count'])
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.cpu().numpy().tobytes())
    return model, {'initial_state_sha256': checksum.hexdigest(), 'vanilla_checkpoint_sha256': parent}


def train(config, base, splits, output, device, vanilla_checkpoint=None):
    output = Path(output)
    if (output / 'complete.json').exists():
        completion = json.loads((output / 'complete.json').read_text())
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('completed configuration differs')
        if completion['checkpoint_sha256'] != digest(output / 'best.pt'):
            raise ValueError('completed checkpoint changed')
        print(f'already complete: {output.name}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('existing run configuration differs')
    save_json(output / 'config.json', config)
    model, origin = initialize(config, vanilla_checkpoint)
    if (output / 'initialization.json').exists() and json.loads((output / 'initialization.json').read_text()) != origin:
        raise ValueError('initialization or vanilla checkpoint changed')
    save_json(output / 'initialization.json', origin)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    start, best, best_epoch, elapsed, history = 0, -1., 0, 0., []
    if (output / 'last.pt').exists():
        state = torch.load(output / 'last.pt', map_location=device, weights_only=True)
        if state['config'] != config:
            raise ValueError('resume checkpoint configuration differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, best_epoch = state['epoch'], state['best_accuracy'], state['best_epoch']
        history, elapsed = state['history'], state['seconds']
        print(f'{output.name}: resuming after epoch {start}', flush=True)
    validation = loader(KnownImages(base, splits['val']), config['eval_batch_size'], config['workers'])
    started = time.time()
    for epoch in range(start, config['epochs']):
        # per-epoch seeds make a resumed run reproduce its uninterrupted continuation
        seed_everything(config['seed'] + epoch)
        dataset = KnownImages(base, splits['train'], training=True,
                              strong=config['method'] == 'gcsc', epoch=epoch, seed=config['seed'])
        batches = loader(dataset, config['batch_size'], config['workers'], shuffle=True,
                         seed=config['seed'] + epoch)
        model.train()
        totals = dict(loss=0., classification_loss=0., placeholder_loss=0., mixup_loss=0., mixed_count=0.)
        count = 0
        for images, labels, _ in tqdm(batches, desc=f'{output.name}: epoch {epoch + 1}',
                                      mininterval=15, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if config['method'] == 'proser':
                loss, details = proser.loss(model, images, labels, config['beta'],
                                           config['gamma'], config['mixup_alpha'])
            else:
                method = gcsc if config['method'] == 'gcsc' else vanilla
                loss, details = method.loss(model, images, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            loss.backward()
            optimizer.step()
            count += len(labels)
            totals['loss'] += loss.item() * len(labels)
            for key, value in details.items():
                totals[key] += float(value) * (1 if key == 'mixed_count' else len(labels))
        accuracy = validate(model, validation, device)
        row = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'],
               'validation_accuracy': accuracy,
               **{key: value / count for key, value in totals.items() if key != 'mixed_count'},
               'mixed_count': int(totals['mixed_count'])}
        history.append(row)
        if accuracy > best:
            best, best_epoch = accuracy, epoch + 1
            save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'config': config,
                            'epoch': best_epoch, 'validation_accuracy': best})
        scheduler.step()
        save_csv(output / 'history.csv', history)
        save_checkpoint(output / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'config': config, 'epoch': epoch + 1,
                        'best_accuracy': best, 'best_epoch': best_epoch, 'history': history,
                        'seconds': elapsed + time.time() - started})
        print(f'{output.name}: epoch {epoch + 1}/{config["epochs"]}, '
              f'validation accuracy={accuracy:.4f}, loss={row["loss"]:.4f}', flush=True)
    save_json(output / 'complete.json', {'epochs': config['epochs'], 'selected_epoch': best_epoch,
              'validation_accuracy': best, 'checkpoint_sha256': digest(output / 'best.pt'),
              'seconds': elapsed + time.time() - started})
