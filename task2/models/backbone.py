"""resnet-18 features and frozen batch-normalization statistics."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def make_backbone(pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def freeze_bn_statistics(model):
    # scale and bias remain trainable while running statistics stay fixed
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
