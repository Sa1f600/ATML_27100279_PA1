"""correctness checks without pretrained downloads or real PACS training."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.pacs_protocol import CLASSES, SOURCES, make_splits
from task2.methods.dan import mmd_loss
from task2.train import train as train_task2
from task3.evaluation.sharpness import sharpness, sharpness_paths
from task3.evaluation.source_diagnostics import diagnose
from task3.evaluate_sketch import evaluate
from task3.methods.dan_dg import pairwise_mmd
from task3.methods.sam import ascent_step, sam_step
from task3.models.backbone import freeze_bn_statistics, make_backbone
from task3.scripts.run_task3 import prepare
from task3.selection.source_validation import source_splits
from task3.train import train
from task3.utils import digest, experiment_configs, save_csv, save_json, verify_lock


class Task3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(6304)

    def test_pairwise_mmd(self):
        features = torch.randn(24, 512, requires_grad=True)
        a, b, c = features.split(8)
        expected = (mmd_loss(a, b) + mmd_loss(a, c) + mmd_loss(b, c)) / 3
        torch.testing.assert_close(pairwise_mmd(features), expected)
        pairwise_mmd(features).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        with self.assertRaises(ValueError):
            pairwise_mmd(features[:20])

    def test_ascent_restores_after_exception(self):
        p = nn.Parameter(torch.tensor([1., 2.]))
        p.grad = torch.tensor([3., 4.])
        original = p.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with ascent_step([p], 0.05):
                torch.testing.assert_close((p - original).norm(), torch.tensor(0.05))
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(p, original))
        p.grad.zero_()
        with ascent_step([p], 0.05):
            self.assertTrue(torch.equal(p, original))

    def test_sam_matches_independent_adamw_update(self):
        model = nn.Linear(4, 3).double()
        reference = copy.deepcopy(model)
        x, y = torch.randn(8, 4, dtype=torch.double), torch.arange(8) % 3
        loss = lambda: nn.functional.cross_entropy(model(x), y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=1e-4)
        parameters = list(reference.parameters())
        original = [p.detach().clone() for p in parameters]
        first = nn.functional.cross_entropy(reference(x), y)
        gradients = torch.autograd.grad(first, parameters)
        norm = torch.cat([g.flatten() for g in gradients]).norm()
        with torch.no_grad():
            for p, gradient in zip(parameters, gradients):
                p.add_(0.05 * gradient / norm)
        second = nn.functional.cross_entropy(reference(x), y)
        second.backward()
        with torch.no_grad():
            for p, old in zip(parameters, original):
                p.copy_(old)
        reference_optimizer.step()
        values = sam_step(loss, model.parameters(), optimizer, 0.05)
        self.assertAlmostEqual(values[0], first.item())
        self.assertAlmostEqual(values[1], second.item())
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_bn_frozen_in_both_sam_passes(self):
        model = make_backbone(pretrained=False)
        model.train()
        freeze_bn_statistics(model)
        before = {name: value.clone() for name, value in model.named_buffers()}
        inputs = torch.randn(2, 3, 32, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sam_step(lambda: model(inputs).square().mean(), model.parameters(), optimizer, 0.05)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name])
        self.assertIsNotNone(model.bn1.weight.grad)

    def test_sharpness_restores_state_and_matches_definition(self):
        backbone, head = nn.Linear(4, 8), nn.Linear(8, 3)
        x, y = torch.randn(12, 4), torch.arange(12) % 3
        parameters = list(backbone.parameters()) + list(head.parameters())
        originals = [p.detach().clone() for p in parameters]
        for p in parameters:
            p.grad = torch.ones_like(p)
        previous_gradients = [p.grad for p in parameters]
        result = sharpness(backbone, head, x, y)
        for p, original, gradient in zip(parameters, originals, previous_gradients):
            self.assertTrue(torch.equal(p, original))
            self.assertIs(p.grad, gradient)
        self.assertTrue(backbone.training and head.training)
        base = nn.functional.cross_entropy(head(backbone(x)), y)
        gradients = torch.autograd.grad(base, parameters)
        norm = torch.cat([g.flatten() for g in gradients]).norm()
        with torch.no_grad():
            for p, gradient in zip(parameters, gradients):
                p.add_(0.05 * gradient / norm)
            expected = nn.functional.cross_entropy(head(backbone(x)), y).item() - base.item()
        self.assertAlmostEqual(result['delta'], expected, places=6)

    def test_complete_workflow_without_target_access_until_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root, previous, output = folder / 'PACS', folder / 'task2', folder / 'task3'
            rng = np.random.default_rng(6304)
            for domain in (*SOURCES, 'sketch'):
                for name in CLASSES:
                    directory = root / domain / name
                    directory.mkdir(parents=True)
                    for index in range(25):
                        Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(directory / f'{index}.png')
            split_file = previous / 'splits.json'
            manifest = make_splits(root, split_file)
            sources = source_splits(manifest)
            self.assertNotIn('target', sources)
            self.assertEqual(sharpness_paths(sources), sharpness_paths(sources))
            self.assertEqual(sum(map(len, sharpness_paths(sources).values())), 96)
            wrong = copy.deepcopy(manifest)
            wrong['sources']['photo']['train'][0] = 'sketch/dog/0.png'
            with self.assertRaises(ValueError):
                source_splits(wrong)
            configs = {name: config | {'max_epochs': 1, 'workers': 0}
                       for name, config in experiment_configs().items()}
            baseline_config = configs['erm'] | {'method': 'source_only', 'max_grl': 1.,
                                               'mmd_weight': 1., 'target_batch_size': 24}
            def tiny(pretrained=True):
                return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512))
            original_open = Image.open
            def source_open(path, *args, **kwargs):
                if 'sketch' in Path(path).parts:
                    raise AssertionError('sketch image opened before final evaluation')
                return original_open(path, *args, **kwargs)
            with patch('PIL.Image.open', side_effect=source_open), patch('task2.train.make_backbone', tiny), \
                 patch('task3.train.make_backbone', tiny), patch('task3.models.backbone.make_backbone', tiny), \
                 patch('task3.scripts.run_task3.experiment_configs', return_value=configs):
                train_task2(baseline_config, root, manifest, previous / 'source_only', 'cpu')
                manifest, plan = prepare(output, previous, split_file)
                self.assertEqual(digest(previous / 'source_only/best.pt'), digest(output / 'erm/best.pt'))
                for name, config in configs.items():
                    if name != 'erm':
                        train(config, root, manifest, output / name, 'cpu')
                # completed runs must be skipped on rerun
                train(configs['sam'], root, manifest, output / 'sam', 'cpu')
                save_json(output / 'checkpoints_locked.json', {
                    'plan_sha256': digest(output / 'experiment_plan.json'),
                    'checkpoints': {name: digest(output / name / 'best.pt') for name in configs}})
                diagnose(root, output, 'cpu')
            final = previous / 'final'
            final.mkdir()
            save_csv(final / 'all_results.csv', [{'method': name, 'target_accuracy': 0.1}
                                               for name in ('source_only', 'dan')])
            save_csv(final / 'per_class.csv', [{'method': method, 'class': name, 'accuracy': 0.1}
                     for method in ('source_only', 'dan') for name in CLASSES])
            with patch('task3.models.backbone.make_backbone', tiny):
                evaluate(root, output, previous, 'cpu')
            results = json.loads((output / 'final/summary.json').read_text())
            self.assertEqual(len(results['results']), 5)
            self.assertEqual(len(results['study']), 3)
            self.assertEqual(len(results['per_class']), 35)
            self.assertEqual(results['results'][0]['target_delta_pp'], 0)
            self.assertTrue((output / 'final/task2_per_class_comparison.csv').exists())
            self.assertTrue((output / 'final/training_curves.png').exists())
            with (output / 'sam/best.pt').open('ab') as file:
                file.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                verify_lock(output)


if __name__ == '__main__':
    unittest.main()
