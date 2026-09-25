"""numerical, leakage, resume, and end-to-end checks without dataset downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from task4.data.cifar10 import KnownImages
from task4.data.cifar100_unknowns import NEAR, FAR, UnknownImages
from task4.data.make_splits import make_splits
from task4.evaluation.metrics import osr_metrics
from task4.evaluation.thresholds import calibrate
from task4.methods import proser
from task4.methods.manifold_mixup import mix_hidden
from task4.models.resnet_cifar import ResNetCIFAR
from task4.scores import energy, mahalanobis, mls, msp, placeholder
from task4.train import train, initialize
from task4.utils import configurations, seed_everything, verify_lock


class Images:
    def __init__(self, classes=10, per_class=20):
        self.classes = list(NEAR + FAR) if classes == 16 else [str(i) for i in range(classes)]
        self.targets = np.repeat(np.arange(classes), per_class).tolist()
        self.data = np.random.default_rng(7).integers(0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.data[index]), self.targets[index]


class TinyModel(nn.Module):
    """exercise the full orchestration cheaply; test the actual ResNet separately."""
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 12)
        self.head = nn.Linear(12, 10)
        self.dummy = None

    def add_dummy(self, count=5):
        self.dummy = nn.Linear(12, count)

    def before_mix(self, images):
        return images.mean((2, 3))

    def after_mix(self, hidden):
        return self.body(hidden).tanh()

    def classify(self, features):
        known = self.head(features)
        return torch.cat([known, self.dummy(features)], 1) if self.dummy is not None else known

    def forward(self, images, return_features=False):
        features = self.after_mix(self.before_mix(images))
        logits = self.classify(features)
        return (features, logits) if return_features else logits


class Task4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_persistent_storage_and_code_protection(self):
        import task4.scripts.colab_task4 as launcher
        from task4.utils import code_hash, save_json
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(launcher, 'OUTPUT'), patch.object(launcher, 'CACHE'):
            folder = launcher.prepare_storage(directory)
            checkpoint = launcher.OUTPUT / 'last.pt'
            checkpoint.write_bytes(b'saved epoch')
            save_json(launcher.OUTPUT / 'experiment_plan.json', {'code_sha256': code_hash()})
            launcher.prepare_storage(directory)
            self.assertEqual(checkpoint.read_bytes(), b'saved epoch')
            self.assertTrue((folder / 'task4_source.zip').exists())
            self.assertTrue(launcher.CACHE.is_dir())
            source = (folder / 'task4_source.zip').read_bytes()
            with patch('task4.utils.code_hash', return_value='different'):
                with self.assertRaisesRegex(ValueError, 'different code'):
                    launcher.prepare_storage(directory)
            self.assertEqual(source, (folder / 'task4_source.zip').read_bytes())

    def test_score_formulas_and_orientation(self):
        logits = np.zeros((2, 10))
        self.assertTrue(np.allclose(msp.score(logits), .9))
        self.assertTrue(np.allclose(mls.score(logits), 0))
        self.assertTrue(np.allclose(energy.score(logits), -np.log(10)))
        combined = np.zeros((2, 15))
        combined[1, 14] = 1024
        self.assertAlmostEqual(placeholder.score(combined)[0], 0)
        self.assertGreater(placeholder.score(combined)[1], 0)
        self.assertAlmostEqual(calibrate(np.arange(100)), 94.05)
        result = osr_metrics(np.array([0., 1.]), np.array([2., 3.]), np.array([4., 5.]), 1.)
        self.assertEqual(result['near_auroc'], 1)
        self.assertEqual(result['far_fpr_at_val95tpr'], 0)
        self.assertEqual(result['known_acceptance'], 1)
        reversed_result = osr_metrics(np.array([4., 5.]), np.array([0., 1.]), np.array([2., 3.]), 4.)
        self.assertEqual(reversed_result['near_auroc'], 0)
        with self.assertRaises(ValueError):
            calibrate([np.nan])

    def test_mahalanobis_within_class_covariance(self):
        labels = np.repeat(np.arange(10), 2)
        features = np.repeat(np.arange(10) * 10., 2)[:, None] + np.tile([-1., 1.], 10)[:, None]
        fitted = mahalanobis.fit(features, labels)
        self.assertTrue(np.allclose(fitted['variance'], 1 + 1e-6))
        self.assertTrue(np.allclose(mahalanobis.score(np.array([[0.], [5.]]), fitted), [0, 25 / (1 + 1e-6)]))

    def test_proser_loss_and_dummy_gradients(self):
        logits = torch.zeros(2, 15, requires_grad=True)
        with torch.no_grad():
            logits[:, 14] = 1
        labels = torch.tensor([0, 1])
        loss, classification, reserve = proser.classifier_placeholder_loss(logits, labels)
        denominator = 10 + np.e
        self.assertAlmostEqual(classification.item(), np.log(denominator), places=6)
        self.assertAlmostEqual(reserve.item(), np.log(9 + np.e) - 1, places=6)
        self.assertAlmostEqual(loss.item(), classification.item() + reserve.item(), places=6)
        gradient = torch.autograd.grad(reserve, logits, retain_graph=True)[0]
        self.assertEqual(gradient[0, 0].item(), 0)
        self.assertEqual(gradient[:, 10:14].abs().sum().item(), 0)
        self.assertGreater(gradient[:, 14].abs().sum().item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mixup_pairs_and_empty_set(self):
        seed_everything(6304)
        hidden = torch.arange(80.).reshape(20, 4)
        labels = torch.arange(20) % 10
        mixed, info = mix_hidden(hidden, labels)
        self.assertGreater(len(mixed), 0)
        self.assertTrue((labels[info['keep']] != labels[info['permutation']][info['keep']]).all())
        expected = info['weight'] * hidden[info['keep']] + (1 - info['weight']) * hidden[info['permutation']][info['keep']]
        self.assertTrue(torch.equal(mixed, expected))
        empty, _ = mix_hidden(hidden, torch.zeros(20, dtype=torch.long))
        self.assertEqual(len(empty), 0)
        model = TinyModel()
        model.add_dummy()
        loss, details = proser.loss(model, torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))
        self.assertEqual(details['mixed_count'], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_actual_resnet_proser_backward(self):
        seed_everything(6304)
        model = ResNetCIFAR()
        self.assertEqual(model.backbone.conv1.kernel_size, (3, 3))
        images = torch.randn(8, 3, 32, 32)
        self.assertEqual(tuple(model.before_mix(images).shape), (8, 128, 16, 16))
        model.add_dummy(5)
        features, logits = model(images, return_features=True)
        self.assertEqual(tuple(features.shape), (8, 512))
        self.assertEqual(tuple(logits.shape), (8, 15))
        loss, details = proser.loss(model, images, torch.arange(8))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(details['mixed_count'], 0)
        for part in (model.backbone.conv1, model.backbone.layer3[0].conv1, model.dummy):
            self.assertTrue(torch.isfinite(part.weight.grad).all())
            self.assertGreater(part.weight.grad.abs().sum().item(), 0)

    def test_split_and_rng_preservation(self):
        base = Images()
        with tempfile.TemporaryDirectory() as directory:
            splits = make_splits(base.targets, Path(directory) / 'split.json')
            self.assertEqual(len(splits['train']), 180)
            self.assertFalse(set(splits['train']) & set(splits['val']))
            self.assertEqual(np.bincount(np.array(base.targets)[splits['val']]).tolist(), [2] * 10)
        dataset = KnownImages(base, range(len(base)), training=True, strong=True)
        before = torch.get_rng_state().clone()
        image = dataset[3][0]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(image, dataset[3][0]))

    def test_resume_matches_uninterrupted(self):
        import task4.train as training
        config = dict(configurations()['vanilla'], epochs=2, batch_size=20, workers=0)
        base = Images()
        with tempfile.TemporaryDirectory() as directory, patch.object(training, 'ResNetCIFAR', TinyModel):
            path = Path(directory)
            splits = make_splits(base.targets, path / 'split.json')
            train(config, base, splits, path / 'full', 'cpu')
            original = training.validate
            calls = 0
            def interrupted(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interruption')
                return original(*args)
            with patch.object(training, 'validate', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    train(config, base, splits, path / 'resumed', 'cpu')
            train(config, base, splits, path / 'resumed', 'cpu')
            full = torch.load(path / 'full/last.pt', weights_only=True)
            resumed = torch.load(path / 'resumed/last.pt', weights_only=True)
            self.assertEqual(full['history'], resumed['history'])
            for key, value in full['model'].items():
                self.assertTrue(torch.equal(value, resumed['model'][key]))

    def test_complete_pipeline_and_unknown_gate(self):
        import task4.train as training
        import task4.extract_outputs as extraction
        import task4.scripts.run_task4 as runner
        configs = {name: dict(config, epochs=1, batch_size=20, eval_batch_size=256, workers=0)
                   for name, config in configurations().items()}
        base = Images()
        unknown = Images(16, 100)
        with tempfile.TemporaryDirectory() as directory:
            output, cache = Path(directory) / 'results', Path(directory) / 'cache'
            def unknown_loader(root):
                self.assertTrue((output / 'calibration_locked.json').exists())
                self.assertTrue(all((output / name / 'complete.json').exists() for name in configs))
                return unknown, {'near': UnknownImages(unknown, NEAR), 'far': UnknownImages(unknown, FAR)}
            with patch.object(training, 'ResNetCIFAR', TinyModel), \
                 patch.object(extraction, 'ResNetCIFAR', TinyModel), \
                 patch.object(runner, 'configurations', return_value=configs), \
                 patch.object(runner, 'load_known', return_value=base), \
                 patch.object(extraction, 'load_known', return_value=base), \
                 patch('task4.data.cifar100_unknowns.load_unknowns', side_effect=unknown_loader) as unknown_call:
                runner.run(directory, output, cache, 'cpu')
                self.assertEqual(unknown_call.call_count, 2)
                rows = json.loads((output / 'all_results.json').read_text())
                self.assertEqual(len(rows), 7)
                self.assertTrue((output / 'vanilla_roc.png').exists())
                self.assertTrue((output / 'failure_examples.png').exists())
                vanilla = torch.load(output / 'vanilla/best.pt', weights_only=True)
                # check PROSER inherits every selected vanilla parameter before adding dummies
                model, _ = initialize(configs['proser'], output / 'vanilla/best.pt')
                for key, value in vanilla['model'].items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]))
                runner.run(directory, output, cache, 'cpu')
                with (output / 'vanilla/best.pt').open('ab') as stream:
                    stream.write(b'tampered')
                with self.assertRaises(ValueError):
                    verify_lock(output)


if __name__ == '__main__':
    unittest.main()
