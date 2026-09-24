"""small correctness checks without downloads or real PACS training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.pacs import PACS
from shared.pacs_protocol import CLASSES, SOURCES, class_label, make_splits
from task2.evaluate_final import evaluate
from task2.methods.cdan import conditional_loss
from task2.methods.dan import mmd_loss
from task2.models.backbone import freeze_bn_statistics, make_backbone
from task2.models.domain_discriminator import make_discriminator, reverse
from task2.train import train
from task2.utils import code_hash, digest, experiment_configs, save_json


class Task2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(6304)

    def test_gradient_reversal(self):
        values = torch.ones(4, requires_grad=True)
        reverse(values, 0.25).sum().backward()
        torch.testing.assert_close(values.grad, torch.full((4,), -0.25))

    def test_mmd(self):
        source = torch.randn(12, 8, requires_grad=True)
        target = torch.randn(12, 8) + 4
        self.assertAlmostEqual(mmd_loss(source, source).item(), 0, places=5)
        self.assertGreater(mmd_loss(source, target).item(), 0)
        mmd_loss(source, target).backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(source.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(mmd_loss(torch.zeros(4, 8), torch.zeros(4, 8))))

    def test_cdan_gradients_reach_predictions(self):
        features = torch.randn(8, 512, requires_grad=True)
        logits = torch.randn(8, 7, requires_grad=True)
        discriminator = make_discriminator(512 * 7)
        conditional_loss(features, logits, 4, discriminator, 1).backward()
        for gradient in (features.grad, logits.grad, discriminator[0].weight.grad):
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_batchnorm_statistics(self):
        model = make_backbone(pretrained=False)
        model.train()
        freeze_bn_statistics(model)
        before = {name: value.clone() for name, value in model.named_buffers()}
        model(torch.randn(2, 3, 64, 64)).square().mean().backward()
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name])
        self.assertIsNotNone(model.bn1.weight.grad)
        self.assertTrue(model.training)
        self.assertFalse(model.bn1.training)

    def test_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'PACS'
            output = Path(directory) / 'results'
            rng = np.random.default_rng(6304)
            for domain in (*SOURCES, 'sketch'):
                for name in CLASSES:
                    folder = root / domain / name
                    folder.mkdir(parents=True)
                    for index in range(5):
                        image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                        Image.fromarray(image).save(folder / f'{index}.png')
            split_file = Path(directory) / 'split.json'
            splits = make_splits(root, split_file)
            self.assertEqual(splits, make_splits(root, split_file))
            for domain in SOURCES:
                train_paths, val_paths = (splits['sources'][domain][key] for key in ('train', 'val'))
                self.assertFalse(set(train_paths) & set(val_paths))
                self.assertEqual((len(train_paths), len(val_paths)), (28, 7))
            # raising inside class_label verifies that unlabeled loading cannot touch it
            with patch('shared.pacs.class_label', side_effect=AssertionError('target label access')):
                _, label = PACS(root, splits['target'], labeled=False)[0]
                self.assertEqual(label, -1)
            configs = {name: config | {'max_epochs': 1, 'workers': 0}
                       for name, config in experiment_configs().items()}
            save_json(output / 'splits.json', splits)
            save_json(output / 'experiment_plan.json', {'runs': configs, 'code_sha256': code_hash(),
                      'split_sha256': digest(output / 'splits.json')})
            source_inputs = {}
            initial_weights = {}

            class TinyBackbone(nn.Sequential):
                def forward(self, inputs):
                    if self.training:
                        source_inputs[current].append(inputs[:24].detach().clone())
                    return super().forward(inputs)

            def tiny_backbone(pretrained=True):
                model = TinyBackbone(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512))
                initial_weights[current] = model[-1].weight.detach().clone()
                return model

            def source_label_only(path):
                if Path(path).parts[0] == 'sketch':
                    raise AssertionError('target class labels were read during training')
                return class_label(path)

            with patch('task2.train.make_backbone', tiny_backbone), patch(
                    'shared.pacs.class_label', side_effect=source_label_only):
                for current, config in configs.items():
                    source_inputs[current] = []
                    train(config, root, splits, output / current, 'cpu')
                    self.assertTrue((output / current / 'best.pt').exists())
                count = len(source_inputs['source_only'])
                train(configs['source_only'], root, splits, output / 'source_only', 'cpu')
                self.assertEqual(len(source_inputs['source_only']), count)
            for name in configs:
                torch.testing.assert_close(initial_weights[name], initial_weights['source_only'])
                for baseline, other in zip(source_inputs['source_only'], source_inputs[name]):
                    torch.testing.assert_close(baseline, other)
            save_json(output / 'checkpoints_locked.json', {
                'plan_sha256': digest(output / 'experiment_plan.json'),
                'checkpoints': {name: digest(output / name / 'best.pt') for name in configs}})
            with patch('task2.evaluate_final.make_backbone', tiny_backbone):
                evaluate(root, output, 'cpu')
            summary = json.loads((output / 'final/summary.json').read_text())
            self.assertEqual(len(summary['results']), 6)
            self.assertEqual(len(summary['per_class']), 42)
            self.assertEqual(len(summary['study']), 3)
            self.assertEqual(summary['results'][0]['target_delta_pp'], 0)
            self.assertTrue((output / 'final/loss_curves.png').exists())
            # a changed checkpoint must fail before target evaluation is entered
            with (output / 'source_only/best.pt').open('ab') as file:
                file.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                evaluate(root, output, 'cpu')


if __name__ == '__main__':
    unittest.main()
