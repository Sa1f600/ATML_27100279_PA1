"""generate balanced shape and texture cue conflict candidates"""

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import STL10
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from task1.models.backbones import COMMON_TRANSFORM


SEED = 6304
ALPHA = 0.8
CANDIDATES_PER_DIRECTION = 25
CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
CLASS_PAIRS = [
    ("airplane", "ship"),
    ("car", "truck"),
    ("cat", "dog"),
    ("deer", "horse"),
    ("bird", "monkey"),
]

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK_DIR.parent / "data"
SPLITS_FILE = TASK_DIR / "results" / "splits.json"
ADAIN_DIR = TASK_DIR / "external" / "pytorch-AdaIN"
OUTPUT_DIR = TASK_DIR / "results" / "cue_conflicts"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"
REVIEW_DIR = OUTPUT_DIR / "review"
MANIFEST_FILE = OUTPUT_DIR / "manifest.csv"
CONFIG_FILE = OUTPUT_DIR / "generation_config.json"


def load_adain(device: torch.device):
    model_dir = ADAIN_DIR / "models"
    vgg_file = model_dir / "vgg_normalised.pth"
    decoder_file = model_dir / "decoder.pth"

    if not (ADAIN_DIR / "net.py").exists():
        raise FileNotFoundError(f"missing adain repository at {ADAIN_DIR}")

    if not vgg_file.exists() or not decoder_file.exists():
        raise FileNotFoundError(f"missing adain weights in {model_dir}")

    sys.path.insert(0, str(ADAIN_DIR))
    net = importlib.import_module("net")
    functions = importlib.import_module("function")

    decoder = net.decoder
    vgg = net.vgg
    decoder.load_state_dict(torch.load(decoder_file, map_location="cpu"))
    vgg.load_state_dict(torch.load(vgg_file, map_location="cpu"))

    # the released model uses vgg layers through relu four one
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    decoder.eval().to(device)
    vgg.eval().to(device)

    for model in (vgg, decoder):
        for parameter in model.parameters():
            parameter.requires_grad = False

    return vgg, decoder, functions.adaptive_instance_normalization


def stylize(content, style, vgg, decoder, adain) -> torch.Tensor:
    with torch.inference_mode():
        content_features = vgg(content)
        style_features = vgg(style)
        transferred = adain(content_features, style_features)
        blended = ALPHA * transferred + (1 - ALPHA) * content_features
        output = decoder(blended)

    return output.clamp(0, 1)


def make_pairs(labels: np.ndarray, test_indices: list[int]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    indices_by_class = {
        class_id: [index for index in test_indices if labels[index] == class_id]
        for class_id in range(len(CLASS_NAMES))
    }
    pairs = []
    candidate_number = 1

    # both directions are generated for every unordered class pair
    for first_name, second_name in CLASS_PAIRS:
        for content_name, style_name in (
            (first_name, second_name),
            (second_name, first_name),
        ):
            content_class = CLASS_NAMES.index(content_name)
            style_class = CLASS_NAMES.index(style_name)
            content_indices = rng.choice(
                indices_by_class[content_class],
                size=CANDIDATES_PER_DIRECTION,
                replace=False,
            )
            style_indices = rng.choice(
                indices_by_class[style_class],
                size=CANDIDATES_PER_DIRECTION,
                replace=False,
            )

            for content_index, style_index in zip(content_indices, style_indices):
                pairs.append(
                    {
                        "candidate_id": f"conflict_{candidate_number:03d}",
                        "content_index": int(content_index),
                        "content_class_id": content_class,
                        "content_class": content_name,
                        "style_index": int(style_index),
                        "style_class_id": style_class,
                        "style_class": style_name,
                    }
                )
                candidate_number += 1

    return pairs


def save_review_image(
    candidate_id: str,
    content_name: str,
    style_name: str,
    content_image: Image.Image,
    style_image: Image.Image,
    output_image: Image.Image,
) -> None:
    canvas = Image.new("RGB", (672, 250), "white")
    canvas.paste(content_image, (0, 26))
    canvas.paste(style_image, (224, 26))
    canvas.paste(output_image, (448, 26))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), f"{candidate_id} content {content_name}", fill="black")
    draw.text((228, 6), f"style {style_name}", fill="black")
    draw.text((452, 6), "output", fill="black")
    canvas.save(REVIEW_DIR / f"{candidate_id}.jpg", quality=90)


def write_manifest(rows: list[dict]) -> None:
    fieldnames = [
        "candidate_id",
        "content_index",
        "content_class_id",
        "content_class",
        "style_index",
        "style_class_id",
        "style_class",
        "alpha",
        "output_path",
        "review_path",
        "accepted",
        "rejection_reason",
    ]

    with MANIFEST_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if MANIFEST_FILE.exists():
        print(f"manifest already exists at {MANIFEST_FILE}")
        return

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    with SPLITS_FILE.open(encoding="utf-8") as file:
        test_indices = json.load(file)["test_indices"]

    dataset = STL10(root=DATA_DIR, split="test", download=False)
    pairs = make_pairs(np.asarray(dataset.labels), test_indices)
    vgg, decoder, adain = load_adain(device)
    manifest_rows = []

    for pair in tqdm(pairs):
        candidate_id = pair["candidate_id"]
        output_file = CANDIDATE_DIR / f"{candidate_id}.png"
        review_file = REVIEW_DIR / f"{candidate_id}.jpg"

        content_image = dataset[pair["content_index"]][0]
        style_image = dataset[pair["style_index"]][0]
        content_tensor = COMMON_TRANSFORM(content_image).unsqueeze(0).to(device)
        style_tensor = COMMON_TRANSFORM(style_image).unsqueeze(0).to(device)

        if output_file.exists():
            output_image = Image.open(output_file).convert("RGB")
        else:
            output = stylize(content_tensor, style_tensor, vgg, decoder, adain)
            output_image = to_pil_image(output.squeeze(0).cpu())
            output_image.save(output_file)

        if not review_file.exists():
            save_review_image(
                candidate_id,
                pair["content_class"],
                pair["style_class"],
                to_pil_image(content_tensor.squeeze(0).cpu()),
                to_pil_image(style_tensor.squeeze(0).cpu()),
                output_image,
            )

        manifest_rows.append(
            {
                **pair,
                "alpha": ALPHA,
                "output_path": str(output_file.relative_to(TASK_DIR)),
                "review_path": str(review_file.relative_to(TASK_DIR)),
                "accepted": "",
                "rejection_reason": "",
            }
        )

    write_manifest(manifest_rows)
    config = {
        "seed": SEED,
        "alpha": ALPHA,
        "candidates_per_direction": CANDIDATES_PER_DIRECTION,
        "class_pairs": CLASS_PAIRS,
        "total_candidates": len(manifest_rows),
        "adain_source": "https://github.com/naoto0804/pytorch-AdaIN",
    }

    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"generated {len(manifest_rows)} candidates")
    print(f"review images in {REVIEW_DIR}")
    print(f"manifest saved to {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
