"""fixed evaluation-only CIFAR-100 test groups; never construct training data."""

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from task4.data.cifar10 import evaluation_transform

NEAR = ('bus', 'pickup_truck', 'motorcycle', 'tractor', 'wolf', 'fox', 'leopard', 'camel')
FAR = ('bottle', 'bowl', 'chair', 'clock', 'keyboard', 'mushroom', 'sunflower', 'wardrobe')


class UnknownImages(Dataset):
    def __init__(self, base, names):
        self.base = base
        self.indices = [i for i, label in enumerate(base.targets) if base.classes[label] in names]
        self.normalize = evaluation_transform()
        for name in names:
            count = sum(base.classes[base.targets[i]] == name for i in self.indices)
            if count != 100:
                raise ValueError(f'expected 100 test images for {name}, found {count}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.base[original_index]
        return self.normalize(image), label, original_index


def load_unknowns(root):
    # this function is called only after score calibration has been locked
    base = CIFAR100(root=root, train=False, download=True)
    return base, {'near': UnknownImages(base, NEAR), 'far': UnknownImages(base, FAR)}
