"""frozen feature extractors used in task one"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)
from torchvision.transforms import InterpolationMode

import open_clip


FEATURE_DIMS = {
    "resnet50": 2048,
    "vit_b_16": 768,
    "clip_vit_b_32": 512,
}

CLIP_MODEL_NAME = "ViT-B-32"


def convert_to_rgb(image):
    return image.convert("RGB")


# every model receives the same image before model specific normalization
COMMON_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(convert_to_rgb),
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
    ]
)

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

CLIP_NORMALIZE = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)


def freeze(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.eval()


class ResNet50Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # replacing the classifier exposes the pooled representation
        model.fc = nn.Identity()
        self.model = freeze(model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(IMAGENET_NORMALIZE(images))


class ViTB16Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # replacing the heads exposes the final class token
        model.heads = nn.Identity()
        self.model = freeze(model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(IMAGENET_NORMALIZE(images))


class CLIPViTB32Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = open_clip.create_model(
            CLIP_MODEL_NAME,
            pretrained="openai",
            force_quick_gelu=True,
        )
        self.model = freeze(model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(CLIP_NORMALIZE(images))

        # the assignment requires normalized clip image embeddings
        return F.normalize(features, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(tokens)
        return F.normalize(features, dim=-1)

    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()


def load_backbone(name: str, device: torch.device) -> nn.Module:
    backbones = {
        "resnet50": ResNet50Backbone,
        "vit_b_16": ViTB16Backbone,
        "clip_vit_b_32": CLIPViTB32Backbone,
    }

    if name not in backbones:
        raise ValueError(f"unknown backbone {name}")

    return backbones[name]().to(device)


def get_clip_tokenizer():
    return open_clip.get_tokenizer(CLIP_MODEL_NAME)
