"""evaluate color changes and shape texture cue conflicts"""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import STL10
from tqdm.auto import tqdm

from task1.data.transforms import HUE_FACTOR, get_transform
from task1.models.backbones import (
    COMMON_TRANSFORM,
    FEATURE_DIMS,
    get_clip_tokenizer,
    load_backbone,
)


BATCH_SIZE = 32
NUM_CLASSES = 10
BACKBONES = ["resnet50", "vit_b_16", "clip_vit_b_32"]
CONDITIONS = ["grayscale", "hue_rotation"]
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
FEATURE_DIR = RESULTS_DIR / "color" / "features"
PREDICTION_DIR = RESULTS_DIR / "color" / "predictions"
METRICS_FILE = RESULTS_DIR / "color" / "color_metrics.json"
CUE_DIR = RESULTS_DIR / "cue_conflicts"
CUE_MANIFEST = CUE_DIR / "manifest.csv"
CUE_EVALUATION_DIR = CUE_DIR / "evaluation"
CUE_FEATURE_DIR = CUE_EVALUATION_DIR / "features"
CUE_PREDICTION_DIR = CUE_EVALUATION_DIR / "predictions"
CUE_METRICS_FILE = CUE_EVALUATION_DIR / "cue_conflict_metrics.json"


class CueConflictDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(TASK_DIR / row["output_path"]).convert("RGB")

        return (
            COMMON_TRANSFORM(image),
            int(row["content_class_id"]),
            int(row["style_class_id"]),
            row["candidate_id"],
        )


def make_loaders() -> dict[str, DataLoader]:
    with SPLITS_FILE.open(encoding="utf-8") as file:
        test_indices = json.load(file)["test_indices"]

    loaders = {}

    for condition in CONDITIONS:
        dataset = STL10(
            root=DATA_DIR,
            split="test",
            transform=get_transform(condition),
            download=False,
        )
        loaders[condition] = DataLoader(
            Subset(dataset, test_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def extract_features(model, loader: DataLoader, device: torch.device):
    feature_batches = []
    label_batches = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, leave=False):
            features = model(images.to(device, non_blocking=True))
            feature_batches.append(features.cpu())
            label_batches.append(labels.long())

    return torch.cat(feature_batches), torch.cat(label_batches)


def extract_color_features(device: torch.device) -> None:
    loaders = make_loaders()

    # each backbone stays loaded while both color conditions are processed
    for name in BACKBONES:
        missing_conditions = [
            condition
            for condition in CONDITIONS
            if not (FEATURE_DIR / condition / f"{name}.pt").exists()
        ]

        if not missing_conditions:
            print(f"skipping {name} because its color features already exist")
            continue

        print(f"loading {name}")
        model = load_backbone(name, device)

        for condition in missing_conditions:
            print(f"extracting {name} {condition} features")
            features, labels = extract_features(model, loaders[condition], device)

            if features.shape != (500, FEATURE_DIMS[name]):
                raise ValueError(f"unexpected feature shape {tuple(features.shape)}")

            output_file = FEATURE_DIR / condition / f"{name}.pt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "labels": labels}, output_file)
            print(f"saved {output_file}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


def load_head(name: str, device: torch.device) -> nn.Module:
    checkpoint_file = RESULTS_DIR / "checkpoints" / f"{name}_head.pt"
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    head = nn.Linear(FEATURE_DIMS[name], NUM_CLASSES)
    head.load_state_dict(checkpoint["model_state"])
    return head.eval().to(device)


def save_head_predictions(device: torch.device) -> None:
    for name in BACKBONES:
        head = load_head(name, device)

        for condition in CONDITIONS:
            output_file = PREDICTION_DIR / f"{condition}_{name}_head.pt"

            if output_file.exists():
                continue

            data = torch.load(
                FEATURE_DIR / condition / f"{name}.pt",
                map_location="cpu",
            )

            with torch.inference_mode():
                logits = head(data["features"].to(device)).cpu()

            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            torch.save(
                {
                    "logits": logits,
                    "labels": data["labels"],
                    "predictions": predictions,
                    "confidence": confidence,
                },
                output_file,
            )

        del head
        torch.cuda.empty_cache()


def save_zero_shot_predictions(device: torch.device) -> None:
    output_files = {
        condition: PREDICTION_DIR / f"{condition}_clip_zero_shot.pt"
        for condition in CONDITIONS
    }

    if all(path.exists() for path in output_files.values()):
        return

    clip_model = load_backbone("clip_vit_b_32", device)
    tokenizer = get_clip_tokenizer()
    prompts = [f"a photo of a {class_name}." for class_name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokens)

        for condition, output_file in output_files.items():
            if output_file.exists():
                continue

            data = torch.load(
                FEATURE_DIR / condition / "clip_vit_b_32.pt",
                map_location="cpu",
            )
            image_features = data["features"].to(device)
            logits = (clip_model.logit_scale() * image_features @ text_features.T).cpu()
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            torch.save(
                {
                    "logits": logits,
                    "labels": data["labels"],
                    "predictions": predictions,
                    "confidence": confidence,
                },
                output_file,
            )

    del clip_model
    torch.cuda.empty_cache()


def compare_with_clean(clean_file: Path, transformed_file: Path) -> dict:
    clean = torch.load(clean_file, map_location="cpu")
    transformed = torch.load(transformed_file, map_location="cpu")

    if not torch.equal(clean["labels"], transformed["labels"]):
        raise ValueError("clean and transformed labels do not match")

    labels = transformed["labels"]
    predictions = transformed["predictions"]
    transformed_accuracy = (predictions == labels).float().mean().item()
    clean_accuracy = (clean["predictions"] == labels).float().mean().item()

    return {
        "clean_accuracy": clean_accuracy,
        "accuracy": transformed_accuracy,
        "accuracy_change": transformed_accuracy - clean_accuracy,
        "prediction_consistency": (
            predictions == clean["predictions"]
        ).float().mean().item(),
        "macro_f1": float(
            f1_score(labels.numpy(), predictions.numpy(), average="macro")
        ),
        "mean_maximum_confidence": transformed["confidence"].mean().item(),
    }


def calculate_metrics() -> dict:
    results = {}

    for condition in CONDITIONS:
        condition_results = {}

        for name in BACKBONES:
            model_name = f"{name}_head"
            condition_results[model_name] = compare_with_clean(
                RESULTS_DIR / "predictions" / f"{model_name}.pt",
                PREDICTION_DIR / f"{condition}_{model_name}.pt",
            )

        condition_results["clip_zero_shot"] = compare_with_clean(
            RESULTS_DIR / "predictions" / "clip_zero_shot.pt",
            PREDICTION_DIR / f"{condition}_clip_zero_shot.pt",
        )
        results[condition] = condition_results

    return {
        "settings": {
            "hue_factor": HUE_FACTOR,
            "test_images": 500,
        },
        "results": results,
    }


def load_accepted_cue_rows() -> list[dict]:
    with CUE_MANIFEST.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if any(row["accepted"] not in {"yes", "no"} for row in rows):
        raise ValueError("finish reviewing every cue conflict before evaluation")

    accepted_rows = [row for row in rows if row["accepted"] == "yes"]

    if len(accepted_rows) < 200:
        raise ValueError("at least 200 accepted cue conflicts are required")

    for row in accepted_rows:
        if not (TASK_DIR / row["output_path"]).exists():
            raise FileNotFoundError(TASK_DIR / row["output_path"])

    return accepted_rows


def make_cue_loader(rows: list[dict]) -> DataLoader:
    return DataLoader(
        CueConflictDataset(rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def extract_cue_features(rows: list[dict], device: torch.device) -> None:
    loader = make_cue_loader(rows)

    for name in BACKBONES:
        output_file = CUE_FEATURE_DIR / f"{name}.pt"

        if output_file.exists():
            print(f"skipping {name} because its cue conflict features already exist")
            continue

        print(f"extracting {name} cue conflict features")
        model = load_backbone(name, device)
        feature_batches = []
        content_batches = []
        style_batches = []
        candidate_ids = []

        with torch.inference_mode():
            for images, content_labels, style_labels, ids in tqdm(loader, leave=False):
                feature_batches.append(model(images.to(device, non_blocking=True)).cpu())
                content_batches.append(content_labels.long())
                style_batches.append(style_labels.long())
                candidate_ids.extend(ids)

        features = torch.cat(feature_batches)

        if features.shape != (len(rows), FEATURE_DIMS[name]):
            raise ValueError(f"unexpected feature shape {tuple(features.shape)}")

        torch.save(
            {
                "features": features,
                "content_labels": torch.cat(content_batches),
                "style_labels": torch.cat(style_batches),
                "candidate_ids": candidate_ids,
            },
            output_file,
        )
        print(f"saved {output_file}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


def save_cue_head_predictions(device: torch.device) -> None:
    for name in BACKBONES:
        output_file = CUE_PREDICTION_DIR / f"{name}_head.pt"

        if output_file.exists():
            continue

        data = torch.load(CUE_FEATURE_DIR / f"{name}.pt", map_location="cpu")
        head = load_head(name, device)

        with torch.inference_mode():
            logits = head(data["features"].to(device)).cpu()

        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        torch.save(
            {
                "predictions": predictions,
                "confidence": confidence,
                "content_labels": data["content_labels"],
                "style_labels": data["style_labels"],
                "candidate_ids": data["candidate_ids"],
            },
            output_file,
        )

        del head
        torch.cuda.empty_cache()


def save_cue_zero_shot_predictions(device: torch.device) -> None:
    output_file = CUE_PREDICTION_DIR / "clip_zero_shot.pt"

    if output_file.exists():
        return

    data = torch.load(CUE_FEATURE_DIR / "clip_vit_b_32.pt", map_location="cpu")
    clip_model = load_backbone("clip_vit_b_32", device)
    tokenizer = get_clip_tokenizer()
    prompts = [f"a photo of a {class_name}." for class_name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokens)
        image_features = data["features"].to(device)
        logits = (clip_model.logit_scale() * image_features @ text_features.T).cpu()

    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    torch.save(
        {
            "predictions": predictions,
            "confidence": confidence,
            "content_labels": data["content_labels"],
            "style_labels": data["style_labels"],
            "candidate_ids": data["candidate_ids"],
        },
        output_file,
    )

    del clip_model
    torch.cuda.empty_cache()


def cue_prediction_metrics(prediction_file: Path) -> dict:
    data = torch.load(prediction_file, map_location="cpu")
    predictions = data["predictions"]
    content_labels = data["content_labels"]
    style_labels = data["style_labels"]
    shape_count = int((predictions == content_labels).sum())
    texture_count = int((predictions == style_labels).sum())
    other_count = len(predictions) - shape_count - texture_count
    covered_count = shape_count + texture_count

    return {
        "total": len(predictions),
        "shape_count": shape_count,
        "texture_count": texture_count,
        "other_count": other_count,
        "shape_bias": shape_count / covered_count if covered_count else 0.0,
        "coverage": covered_count / len(predictions),
        "mean_maximum_confidence": data["confidence"].mean().item(),
    }


def evaluate_cue_conflicts(device: torch.device) -> None:
    CUE_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    CUE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_accepted_cue_rows()
    print(f"evaluating {len(rows)} accepted cue conflicts")

    extract_cue_features(rows, device)
    save_cue_head_predictions(device)
    save_cue_zero_shot_predictions(device)

    model_names = [f"{name}_head" for name in BACKBONES]
    model_names.append("clip_zero_shot")
    metrics = {
        "accepted_images": len(rows),
        "results": {
            name: cue_prediction_metrics(CUE_PREDICTION_DIR / f"{name}.pt")
            for name in model_names
        },
    }

    with CUE_METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved {CUE_METRICS_FILE}")


def evaluate_color(device: torch.device) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    extract_color_features(device)
    save_head_predictions(device)
    save_zero_shot_predictions(device)
    metrics = calculate_metrics()

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue-conflicts", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    if args.cue_conflicts:
        evaluate_cue_conflicts(device)
    else:
        evaluate_color(device)


if __name__ == "__main__":
    main()
