"""pacs datasets with an explicit unlabeled target mode."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from shared.pacs_protocol import class_label


def transform(training=False):
    spatial = [T.RandomCrop(224), T.RandomHorizontalFlip()] if training else [T.CenterCrop(224)]
    return T.Compose([T.Resize((256, 256)), *spatial, T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class PACS(Dataset):
    def __init__(self, root, paths, training=False, labeled=True):
        self.root, self.paths = Path(root), list(paths)
        self.transform, self.labeled = transform(training), labeled

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(self.root / path) as image:
            image = self.transform(image.convert("RGB"))
        # unlabeled target access never calls class_label
        label = class_label(path) if self.labeled else -1
        return image, label


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(root, paths, batch_size, seed, workers=2, training=False, labeled=True):
    return DataLoader(PACS(root, paths, training, labeled), batch_size=batch_size,
                      shuffle=training, drop_last=training, num_workers=workers,
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def cycle(loader):
    if not len(loader):
        raise ValueError("dataset is smaller than a training batch")
    while True:
        yield from loader
