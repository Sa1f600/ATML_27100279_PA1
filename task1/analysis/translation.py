"""evaluate prediction stability under image translation"""

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from tqdm.auto import tqdm

from task1.data.transforms import (
    TRANSLATION_DIRECTIONS,
    TRANSLATION_PIXELS,
    get_translation_transform,
)
from task1.models.backbones import FEATURE_DIMS, get_clip_tokenizer, load_backbone


BATCH_SIZE = 32
NUM_CLASSES = 10
BACKBONES = ["resnet50", "vit_b_16", "clip_vit_b_32"]
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

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK_DIR.parent / "data"
RESULTS_DIR = TASK_DIR / "results"
SPLITS_FILE = RESULTS_DIR / "splits.json"
TRANSLATION_DIR = RESULTS_DIR / "translation"
FEATURE_DIR = TRANSLATION_DIR / "features"
PREDICTION_DIR = TRANSLATION_DIR / "predictions"
METRICS_FILE = TRANSLATION_DIR / "translation_metrics.json"
PLOT_FILE = TRANSLATION_DIR / "translation_plot.png"


def condition_name(pixels: int, direction: str) -> str:
    return f"shift_{pixels:02d}_{direction}"


def make_loader(pixels: int, direction: str) -> DataLoader:
    with SPLITS_FILE.open(encoding="utf-8") as file:
        test_indices = json.load(file)["test_indices"]

    dataset = STL10(
        root=DATA_DIR,
        split="test",
        transform=get_translation_transform(pixels, direction),
        download=False,
    )

    return DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def extract_features(model, loader: DataLoader, device: torch.device):
    feature_batches = []
    label_batches = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, leave=False):
            features = model(images.to(device, non_blocking=True))
            feature_batches.append(features.cpu())
            label_batches.append(labels.long())

    return torch.cat(feature_batches), torch.cat(label_batches)


def extract_translation_features(device: torch.device) -> None:
    conditions = [
        (pixels, direction)
        for pixels in TRANSLATION_PIXELS[1:]
        for direction in TRANSLATION_DIRECTIONS
    ]

    # each backbone stays loaded while every shifted condition is processed
    for name in BACKBONES:
        missing = [
            (pixels, direction)
            for pixels, direction in conditions
            if not (
                FEATURE_DIR
                / condition_name(pixels, direction)
                / f"{name}.pt"
            ).exists()
        ]

        if not missing:
            print(f"skipping {name} because its translation features already exist")
            continue

        print(f"loading {name}")
        model = load_backbone(name, device)

        for pixels, direction in missing:
            condition = condition_name(pixels, direction)
            print(f"extracting {name} {condition} features")
            loader = make_loader(pixels, direction)
            features, labels = extract_features(model, loader, device)

            if features.shape != (500, FEATURE_DIMS[name]):
                raise ValueError(f"unexpected feature shape {tuple(features.shape)}")

            output_file = FEATURE_DIR / condition / f"{name}.pt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "labels": labels}, output_file)

        del model
        gc.collect()
        torch.cuda.empty_cache()


def load_head(name: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(
        RESULTS_DIR / "checkpoints" / f"{name}_head.pt",
        map_location="cpu",
    )
    head = nn.Linear(FEATURE_DIMS[name], NUM_CLASSES)
    head.load_state_dict(checkpoint["model_state"])
    return head.eval().to(device)


def save_prediction(output_file: Path, logits: torch.Tensor, labels: torch.Tensor) -> None:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    torch.save(
        {
            "labels": labels,
            "predictions": predictions,
            "confidence": confidence,
        },
        output_file,
    )


def save_head_predictions(device: torch.device) -> None:
    for name in BACKBONES:
        head = load_head(name, device)

        for pixels in TRANSLATION_PIXELS[1:]:
            for direction in TRANSLATION_DIRECTIONS:
                condition = condition_name(pixels, direction)
                output_file = PREDICTION_DIR / f"{condition}_{name}_head.pt"

                if output_file.exists():
                    continue

                data = torch.load(
                    FEATURE_DIR / condition / f"{name}.pt",
                    map_location="cpu",
                )

                with torch.inference_mode():
                    logits = head(data["features"].to(device)).cpu()

                save_prediction(output_file, logits, data["labels"])

        del head
        torch.cuda.empty_cache()


def save_zero_shot_predictions(device: torch.device) -> None:
    output_files = [
        PREDICTION_DIR
        / f"{condition_name(pixels, direction)}_clip_zero_shot.pt"
        for pixels in TRANSLATION_PIXELS[1:]
        for direction in TRANSLATION_DIRECTIONS
    ]

    if all(path.exists() for path in output_files):
        return

    clip_model = load_backbone("clip_vit_b_32", device)
    tokenizer = get_clip_tokenizer()
    prompts = [f"a photo of a {class_name}." for class_name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokens)

        for pixels in TRANSLATION_PIXELS[1:]:
            for direction in TRANSLATION_DIRECTIONS:
                condition = condition_name(pixels, direction)
                output_file = PREDICTION_DIR / f"{condition}_clip_zero_shot.pt"

                if output_file.exists():
                    continue

                data = torch.load(
                    FEATURE_DIR / condition / "clip_vit_b_32.pt",
                    map_location="cpu",
                )
                image_features = data["features"].to(device)
                logits = (
                    clip_model.logit_scale()
                    * image_features
                    @ text_features.T
                ).cpu()
                save_prediction(output_file, logits, data["labels"])

    del clip_model
    torch.cuda.empty_cache()


def condition_metrics(clean_file: Path, shifted_file: Path) -> dict:
    clean = torch.load(clean_file, map_location="cpu")
    shifted = torch.load(shifted_file, map_location="cpu")

    if not torch.equal(clean["labels"], shifted["labels"]):
        raise ValueError("clean and shifted labels do not match")

    return {
        "accuracy": (
            shifted["predictions"] == shifted["labels"]
        ).float().mean().item(),
        "consistency": (
            shifted["predictions"] == clean["predictions"]
        ).float().mean().item(),
    }


def calculate_metrics() -> dict:
    model_names = [f"{name}_head" for name in BACKBONES]
    model_names.append("clip_zero_shot")
    results = {}

    for model_name in model_names:
        clean_file = RESULTS_DIR / "predictions" / f"{model_name}.pt"
        clean = torch.load(clean_file, map_location="cpu")
        clean_accuracy = (
            clean["predictions"] == clean["labels"]
        ).float().mean().item()
        model_results = {
            "0": {
                "accuracy": clean_accuracy,
                "consistency": 1.0,
                "directions": {},
            }
        }

        for pixels in TRANSLATION_PIXELS[1:]:
            direction_results = {}

            for direction in TRANSLATION_DIRECTIONS:
                condition = condition_name(pixels, direction)
                shifted_file = PREDICTION_DIR / f"{condition}_{model_name}.pt"
                direction_results[direction] = condition_metrics(
                    clean_file,
                    shifted_file,
                )

            model_results[str(pixels)] = {
                "accuracy": sum(
                    result["accuracy"] for result in direction_results.values()
                ) / len(direction_results),
                "consistency": sum(
                    result["consistency"] for result in direction_results.values()
                ) / len(direction_results),
                "directions": direction_results,
            }

        results[model_name] = model_results

    return {
        "settings": {
            "displacements": TRANSLATION_PIXELS,
            "directions": TRANSLATION_DIRECTIONS,
            "padding": "reflection",
            "test_images": 500,
        },
        "results": results,
    }


def make_plot(metrics: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for model_name, model_results in metrics["results"].items():
        accuracy = [
            model_results[str(pixels)]["accuracy"]
            for pixels in TRANSLATION_PIXELS
        ]
        consistency = [
            model_results[str(pixels)]["consistency"]
            for pixels in TRANSLATION_PIXELS
        ]
        label = model_name.replace("_", " ")
        axes[0].plot(TRANSLATION_PIXELS, accuracy, marker="o", label=label)
        axes[1].plot(TRANSLATION_PIXELS, consistency, marker="o", label=label)

    axes[0].set_title("translation accuracy")
    axes[1].set_title("prediction consistency")

    for axis in axes:
        axis.set_xlabel("displacement in pixels")
        axis.set_ylim(0, 1.02)
        axis.set_xticks(TRANSLATION_PIXELS)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("consistency")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    extract_translation_features(device)
    save_head_predictions(device)
    save_zero_shot_predictions(device)
    metrics = calculate_metrics()

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    make_plot(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_FILE}")
    print(f"saved {PLOT_FILE}")


if __name__ == "__main__":
    main()
