"""CIFAR-10-only training and validation with reproducible augmentation."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def evaluation_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class KnownImages(Dataset):
    def __init__(self, base, indices, training=False, strong=False, epoch=0, seed=6304):
        self.base, self.indices = base, list(indices)
        self.training, self.strong, self.epoch, self.seed = training, strong, epoch, seed
        self.spatial = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
        self.randaugment = T.RandAugment(num_ops=2, magnitude=9)
        self.normalize = evaluation_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.base[original_index]
        if self.training:
            # vanilla and gcsc share crop/flip randomness; randaugment uses its own seed
            seed = self.seed + self.epoch * len(self.base) + original_index
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                image = self.spatial(image)
            if self.strong:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed + 100000000)
                    image = self.randaugment(image)
        return self.normalize(image), label, original_index


def load_known(root, train=True, download=True):
    return CIFAR10(root=root, train=train, download=download)


def loader(dataset, batch_size, workers=2, shuffle=False, seed=6304):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available(), drop_last=False)
