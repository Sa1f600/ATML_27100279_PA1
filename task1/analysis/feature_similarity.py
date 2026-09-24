"""compute cosine stability between clean and transformed representations"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from task1.data.transforms import TRANSLATION_DIRECTIONS, TRANSLATION_PIXELS


BACKBONES = ["resnet50", "vit_b_16", "clip_vit_b_32"]
TASK_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = TASK_DIR / "results"
SPLITS_FILE = RESULTS_DIR / "splits.json"
CUE_MANIFEST = RESULTS_DIR / "cue_conflicts" / "manifest.csv"
OUTPUT_DIR = RESULTS_DIR / "representation_analysis"
METRICS_FILE = OUTPUT_DIR / "feature_similarity_metrics.json"
VALUES_FILE = OUTPUT_DIR / "feature_similarity_values.pt"
SUMMARY_PLOT = OUTPUT_DIR / "feature_similarity_summary.png"


def load_tensor_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing required feature file: {path}")
    return torch.load(path, map_location="cpu")


def clean_data(backbone: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = load_tensor_file(RESULTS_DIR / "features" / f"{backbone}.pt")
    return data["test_features"].float(), data["test_labels"].long()


def aligned_pair(
    backbone: str,
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_features, clean_labels = clean_data(backbone)
    transformed = load_tensor_file(path)
    transformed_labels = transformed["labels"].long()

    if not torch.equal(clean_labels, transformed_labels):
        raise ValueError(f"clean and transformed labels do not match for {path}")

    return clean_features, transformed["features"].float(), clean_labels


def accepted_cue_rows() -> dict[str, dict]:
    with CUE_MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["accepted"] == "yes"]

    if len(rows) < 200:
        raise ValueError("at least 200 accepted cue conflicts are required")

    return {row["candidate_id"]: row for row in rows}


def cue_pair(backbone: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_features, clean_labels = clean_data(backbone)
    cue_data = load_tensor_file(
        RESULTS_DIR
        / "cue_conflicts"
        / "evaluation"
        / "features"
        / f"{backbone}.pt"
    )

    with SPLITS_FILE.open(encoding="utf-8") as file:
        test_indices = json.load(file)["test_indices"]

    clean_position = {image_id: position for position, image_id in enumerate(test_indices)}
    rows = accepted_cue_rows()
    positions = []

    for candidate_id in cue_data["candidate_ids"]:
        if candidate_id not in rows:
            raise ValueError(f"candidate {candidate_id} is not accepted in the manifest")

        content_index = int(rows[candidate_id]["content_index"])

        if content_index not in clean_position:
            raise ValueError(f"content image {content_index} is outside the test subset")

        positions.append(clean_position[content_index])

    selected_positions = torch.tensor(positions)
    selected_clean = clean_features[selected_positions]
    selected_labels = clean_labels[selected_positions]
    content_labels = cue_data["content_labels"].long()

    if not torch.equal(selected_labels, content_labels):
        raise ValueError("cue content labels do not match their clean counterparts")

    return selected_clean, cue_data["features"].float(), content_labels


def visualization_pairs(
    backbone: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return {
        "grayscale": aligned_pair(
            backbone,
            RESULTS_DIR / "color" / "features" / "grayscale" / f"{backbone}.pt",
        ),
        "cue conflict": cue_pair(backbone),
        "translation 32px right": aligned_pair(
            backbone,
            RESULTS_DIR
            / "translation"
            / "features"
            / "shift_32_right"
            / f"{backbone}.pt",
        ),
        "patch shuffle": aligned_pair(
            backbone,
            RESULTS_DIR / "patch_shuffle" / "features" / f"{backbone}.pt",
        ),
    }


def cosine_values(clean_features: torch.Tensor, transformed_features: torch.Tensor):
    if clean_features.shape != transformed_features.shape:
        raise ValueError(
            "paired feature shapes do not match: "
            f"{tuple(clean_features.shape)} and {tuple(transformed_features.shape)}"
        )

    return F.cosine_similarity(clean_features, transformed_features, dim=1)


def summarize(values: torch.Tensor) -> dict:
    return {
        "samples": len(values),
        "mean": values.mean().item(),
        "standard_deviation": values.std(unbiased=False).item(),
        "median": values.median().item(),
        "minimum": values.min().item(),
        "maximum": values.max().item(),
    }


def calculate_metrics() -> tuple[dict, dict]:
    results = {}
    saved_values = {}

    for backbone in BACKBONES:
        pairs = visualization_pairs(backbone)
        model_results = {}
        model_values = {}

        for condition in ("grayscale", "cue conflict", "patch shuffle"):
            clean_features, transformed_features, _ = pairs[condition]
            values = cosine_values(clean_features, transformed_features)
            model_results[condition] = summarize(values)
            model_values[condition] = values

        clean_features, clean_labels = clean_data(backbone)
        translation_results = {
            "0": {
                "average_across_directions": summarize(torch.ones(len(clean_features))),
                "directions": {},
            }
        }
        translation_values = {"0": torch.ones(len(clean_features))}

        for pixels in TRANSLATION_PIXELS[1:]:
            direction_results = {}
            direction_values = {}

            for direction in TRANSLATION_DIRECTIONS:
                data = load_tensor_file(
                    RESULTS_DIR
                    / "translation"
                    / "features"
                    / f"shift_{pixels:02d}_{direction}"
                    / f"{backbone}.pt"
                )

                if not torch.equal(clean_labels, data["labels"].long()):
                    raise ValueError("clean and translated labels do not match")

                values = cosine_values(clean_features, data["features"].float())
                direction_results[direction] = summarize(values)
                direction_values[direction] = values

            combined = torch.cat(list(direction_values.values()))
            translation_results[str(pixels)] = {
                "average_across_directions": summarize(combined),
                "directions": direction_results,
            }
            translation_values[str(pixels)] = direction_values

        model_results["translation"] = translation_results
        model_values["translation"] = translation_values
        results[backbone] = model_results
        saved_values[backbone] = model_values

    metrics = {
        "settings": {
            "metric": "paired cosine similarity",
            "translation_displacements": TRANSLATION_PIXELS,
            "translation_directions": TRANSLATION_DIRECTIONS,
            "cue_clean_counterpart": "content image",
            "visualization_translation": "32 pixels right",
        },
        "results": results,
    }
    return metrics, saved_values


def make_summary_plot(metrics: dict) -> None:
    labels = {
        "resnet50": "ResNet-50",
        "vit_b_16": "ViT-B/16",
        "clip_vit_b_32": "CLIP ViT-B/32",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    conditions = ["grayscale", "cue conflict", "patch shuffle"]
    x_positions = torch.arange(len(conditions), dtype=torch.float32)
    width = 0.24

    for offset, backbone in enumerate(BACKBONES):
        values = [metrics["results"][backbone][name]["mean"] for name in conditions]
        axes[0].bar(
            x_positions.numpy() + (offset - 1) * width,
            values,
            width=width,
            label=labels[backbone],
        )

        translation = [
            metrics["results"][backbone]["translation"][str(pixels)][
                "average_across_directions"
            ]["mean"]
            for pixels in TRANSLATION_PIXELS
        ]
        axes[1].plot(
            TRANSLATION_PIXELS,
            translation,
            marker="o",
            linewidth=2,
            label=labels[backbone],
        )

    axes[0].set_xticks(x_positions.numpy(), conditions)
    axes[0].set_ylabel("mean cosine stability")
    axes[0].set_title("representation stability by intervention")
    axes[1].set_xlabel("translation displacement in pixels")
    axes[1].set_ylabel("mean cosine stability")
    axes[1].set_title("translation representation stability")
    axes[1].set_xticks(TRANSLATION_PIXELS)

    for axis in axes:
        axis.set_ylim(0, 1.01)
        axis.grid(axis="y", alpha=0.3)

    axes[1].legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(SUMMARY_PLOT, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, saved_values = calculate_metrics()

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    torch.save(saved_values, VALUES_FILE)
    make_summary_plot(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_FILE}")
    print(f"saved {VALUES_FILE}")
    print(f"saved {SUMMARY_PLOT}")


if __name__ == "__main__":
    main()
