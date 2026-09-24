"""evaluate prediction stability after deterministic patch shuffling"""

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from tqdm.auto import tqdm

from task1.data.transforms import PATCH_GRID_SIZE, PATCH_SEED, clean, patch_shuffle
from task1.models.backbones import (
    FEATURE_DIMS,
    get_clip_tokenizer,
    load_backbone,
)


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
PATCH_DIR = RESULTS_DIR / "patch_shuffle"
FEATURE_DIR = PATCH_DIR / "features"
PREDICTION_DIR = PATCH_DIR / "predictions"
METRICS_FILE = PATCH_DIR / "patch_shuffle_metrics.json"
SAMPLE_GRID_FILE = PATCH_DIR / "sample_grid.png"


class PatchShuffleDataset(Dataset):
    def __init__(self, dataset: STL10, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        image_id = self.indices[position]
        image, label = self.dataset[image_id]
        return patch_shuffle(image, image_id), label, image_id


def load_test_indices() -> list[int]:
    with SPLITS_FILE.open(encoding="utf-8") as file:
        indices = json.load(file)["test_indices"]

    if len(indices) != 500 or len(set(indices)) != 500:
        raise ValueError("expected 500 unique saved test indices")

    return indices


def make_dataset() -> PatchShuffleDataset:
    dataset = STL10(
        root=DATA_DIR,
        split="test",
        download=False,
    )
    return PatchShuffleDataset(dataset, load_test_indices())


def make_loader(dataset: PatchShuffleDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def save_sample_grid(dataset: PatchShuffleDataset) -> None:
    if SAMPLE_GRID_FILE.exists():
        return

    figure, axes = plt.subplots(4, 2, figsize=(6, 11))

    for row in range(4):
        image_id = dataset.indices[row]
        original, label = dataset.dataset[image_id]
        original_tensor = clean(original)
        shuffled_tensor = patch_shuffle(original, image_id)

        axes[row, 0].imshow(original_tensor.permute(1, 2, 0))
        axes[row, 1].imshow(shuffled_tensor.permute(1, 2, 0))
        axes[row, 0].set_ylabel(CLASS_NAMES[label])

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title("clean")
    axes[0, 1].set_title("4 x 4 patch shuffle")
    figure.tight_layout()
    figure.savefig(SAMPLE_GRID_FILE, dpi=200, bbox_inches="tight")
    plt.close(figure)


def extract_features(model, loader: DataLoader, device: torch.device):
    feature_batches = []
    label_batches = []
    image_id_batches = []

    with torch.inference_mode():
        for images, labels, image_ids in tqdm(loader, leave=False):
            features = model(images.to(device, non_blocking=True))
            feature_batches.append(features.cpu())
            label_batches.append(labels.long())
            image_id_batches.append(image_ids.long())

    return (
        torch.cat(feature_batches),
        torch.cat(label_batches),
        torch.cat(image_id_batches),
    )


def extract_all_features(dataset: PatchShuffleDataset, device: torch.device) -> None:
    loader = make_loader(dataset)

    # process one backbone at a time to limit accelerator memory use
    for name in BACKBONES:
        output_file = FEATURE_DIR / f"{name}.pt"

        if output_file.exists():
            print(f"skipping {name} because its patch features already exist")
            continue

        print(f"extracting {name} patch-shuffle features")
        model = load_backbone(name, device)
        features, labels, image_ids = extract_features(model, loader, device)

        if features.shape != (len(dataset), FEATURE_DIMS[name]):
            raise ValueError(f"unexpected feature shape {tuple(features.shape)}")

        torch.save(
            {
                "features": features,
                "labels": labels,
                "image_ids": image_ids,
            },
            output_file,
        )

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


def save_prediction(
    output_file: Path,
    logits: torch.Tensor,
    labels: torch.Tensor,
    image_ids: torch.Tensor,
) -> None:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    torch.save(
        {
            "logits": logits,
            "labels": labels,
            "image_ids": image_ids,
            "predictions": predictions,
            "confidence": confidence,
        },
        output_file,
    )


def save_head_predictions(device: torch.device) -> None:
    for name in BACKBONES:
        output_file = PREDICTION_DIR / f"{name}_head.pt"

        if output_file.exists():
            continue

        data = torch.load(FEATURE_DIR / f"{name}.pt", map_location="cpu")
        head = load_head(name, device)

        with torch.inference_mode():
            logits = head(data["features"].to(device)).cpu()

        save_prediction(output_file, logits, data["labels"], data["image_ids"])
        del head
        torch.cuda.empty_cache()


def save_zero_shot_predictions(device: torch.device) -> None:
    output_file = PREDICTION_DIR / "clip_zero_shot.pt"

    if output_file.exists():
        return

    data = torch.load(FEATURE_DIR / "clip_vit_b_32.pt", map_location="cpu")
    clip_model = load_backbone("clip_vit_b_32", device)
    tokenizer = get_clip_tokenizer()
    prompts = [f"a photo of a {class_name}." for class_name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokens)
        image_features = data["features"].to(device)
        logits = (clip_model.logit_scale() * image_features @ text_features.T).cpu()

    save_prediction(output_file, logits, data["labels"], data["image_ids"])
    del clip_model
    torch.cuda.empty_cache()


def model_metrics(clean_file: Path, shuffled_file: Path) -> dict:
    clean_data = torch.load(clean_file, map_location="cpu")
    shuffled_data = torch.load(shuffled_file, map_location="cpu")

    if not torch.equal(clean_data["labels"], shuffled_data["labels"]):
        raise ValueError("clean and patch-shuffled labels do not match")

    labels = shuffled_data["labels"]
    clean_predictions = clean_data["predictions"]
    shuffled_predictions = shuffled_data["predictions"]
    clean_accuracy = (clean_predictions == labels).float().mean().item()
    shuffled_accuracy = (shuffled_predictions == labels).float().mean().item()

    return {
        "clean_accuracy": clean_accuracy,
        "patch_shuffle_accuracy": shuffled_accuracy,
        "accuracy_change": shuffled_accuracy - clean_accuracy,
        "accuracy_drop": clean_accuracy - shuffled_accuracy,
        "prediction_consistency": (
            shuffled_predictions == clean_predictions
        ).float().mean().item(),
        "macro_f1": float(
            f1_score(
                labels.numpy(),
                shuffled_predictions.numpy(),
                average="macro",
            )
        ),
        "mean_maximum_confidence": shuffled_data["confidence"].mean().item(),
    }


def calculate_metrics() -> dict:
    model_names = [f"{name}_head" for name in BACKBONES]
    model_names.append("clip_zero_shot")

    return {
        "settings": {
            "grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
            "seed": PATCH_SEED,
            "permutation": "one deterministic non-identity permutation per image",
            "test_images": 500,
        },
        "results": {
            name: model_metrics(
                RESULTS_DIR / "predictions" / f"{name}.pt",
                PREDICTION_DIR / f"{name}.pt",
            )
            for name in model_names
        },
    }


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    dataset = make_dataset()
    save_sample_grid(dataset)
    extract_all_features(dataset, device)
    save_head_predictions(device)
    save_zero_shot_predictions(device)
    metrics = calculate_metrics()

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_FILE}")
    print(f"saved {SAMPLE_GRID_FILE}")


if __name__ == "__main__":
    main()
