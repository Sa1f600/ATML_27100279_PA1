"""CIFAR ResNet-18 with a mixup boundary between layer2 and layer3."""

import torch
from torch import nn
from torchvision.models import resnet18


class ResNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=10)
        self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.backbone.maxpool = nn.Identity()
        self.dummy = None

    def add_dummy(self, count=5):
        if self.dummy is not None:
            raise ValueError('dummy classifiers already exist')
        self.dummy = nn.Linear(512, count)

    def before_mix(self, images):
        model = self.backbone
        x = model.relu(model.bn1(model.conv1(images)))
        return model.layer2(model.layer1(x))

    def after_mix(self, hidden):
        model = self.backbone
        x = model.layer4(model.layer3(hidden))
        return model.avgpool(x).flatten(1)

    def classify(self, features):
        known = self.backbone.fc(features)
        return torch.cat([known, self.dummy(features)], dim=1) if self.dummy is not None else known

    def forward(self, images, return_features=False):
        features = self.after_mix(self.before_mix(images))
        logits = self.classify(features)
        return (features, logits) if return_features else logits
