"""controlled image interventions used in task one"""

import torch

from torchvision.transforms import functional as F

from task1.models.backbones import COMMON_TRANSFORM


HUE_FACTOR = 0.25
TRANSLATION_PIXELS = [0, 8, 16, 32]
TRANSLATION_DIRECTIONS = ["up", "down", "left", "right"]
PATCH_GRID_SIZE = 4
PATCH_SEED = 6304


def clean(image):
    return COMMON_TRANSFORM(image)


def grayscale(image):
    image = COMMON_TRANSFORM(image)

    # three equal channels keep the input compatible with every backbone
    return F.rgb_to_grayscale(image, num_output_channels=3)


def hue_rotation(image):
    image = COMMON_TRANSFORM(image)

    # a fixed hue change preserves geometry brightness and saturation
    return F.adjust_hue(image, HUE_FACTOR)


def translation(image, pixels: int, direction: str):
    image = COMMON_TRANSFORM(image)

    if pixels not in TRANSLATION_PIXELS:
        raise ValueError(f"unknown displacement {pixels}")

    if direction not in TRANSLATION_DIRECTIONS:
        raise ValueError(f"unknown direction {direction}")

    if pixels == 0:
        return image

    height, width = image.shape[-2:]

    # reflection padding supplies pixels outside the original boundary
    padded = F.pad(image, [pixels] * 4, padding_mode="reflect")
    crop_positions = {
        "up": (pixels * 2, pixels),
        "down": (0, pixels),
        "left": (pixels, pixels * 2),
        "right": (pixels, 0),
    }
    top, left = crop_positions[direction]

    return F.crop(padded, top, left, height, width)


def get_translation_transform(pixels: int, direction: str):
    def transform(image):
        return translation(image, pixels, direction)

    return transform


def patch_shuffle(image, image_id: int):
    """apply one deterministic non-identity patch permutation per image"""
    image = COMMON_TRANSFORM(image)
    channels, height, width = image.shape

    if height % PATCH_GRID_SIZE or width % PATCH_GRID_SIZE:
        raise ValueError("image dimensions must be divisible by the patch grid")

    patch_height = height // PATCH_GRID_SIZE
    patch_width = width // PATCH_GRID_SIZE

    # split the image into a row-major sequence of sixteen patches
    patches = (
        image.reshape(
            channels,
            PATCH_GRID_SIZE,
            patch_height,
            PATCH_GRID_SIZE,
            patch_width,
        )
        .permute(1, 3, 0, 2, 4)
        .reshape(PATCH_GRID_SIZE**2, channels, patch_height, patch_width)
    )

    # derive the permutation from the saved dataset index for exact reuse
    generator = torch.Generator().manual_seed(PATCH_SEED + int(image_id))
    permutation = torch.randperm(PATCH_GRID_SIZE**2, generator=generator)
    identity = torch.arange(PATCH_GRID_SIZE**2)

    # replace the extremely unlikely identity draw with a fixed cyclic shift
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, shifts=1)

    shuffled_patches = patches[permutation]

    # assemble the shuffled patches back into a 224 by 224 tensor
    return (
        shuffled_patches.reshape(
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
            channels,
            patch_height,
            patch_width,
        )
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, height, width)
    )


def get_transform(name: str):
    interventions = {
        "clean": clean,
        "grayscale": grayscale,
        "hue_rotation": hue_rotation,
    }

    if name not in interventions:
        raise ValueError(f"unknown intervention {name}")

    return interventions[name]
