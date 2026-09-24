"""extract and save frozen backbone features"""

import gc
import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import STL10
from tqdm.auto import tqdm

from task1.models.backbones import (
    COMMON_TRANSFORM,
    FEATURE_DIMS,
    get_clip_tokenizer,
    load_backbone,
)


SEED = 6304
BATCH_SIZE = 32
HEAD_BATCH_SIZE = 256
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
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
SPLITS_FILE = TASK_DIR / "results" / "splits.json"
FEATURE_DIR = TASK_DIR / "results" / "features"
CHECKPOINT_DIR = TASK_DIR / "results" / "checkpoints"
PREDICTION_DIR = TASK_DIR / "results" / "predictions"
METRICS_FILE = TASK_DIR / "results" / "clean_baseline.json"


def make_loaders() -> dict[str, DataLoader]:
    with SPLITS_FILE.open(encoding="utf-8") as file:
        splits = json.load(file)

    train_dataset = STL10(
        root=DATA_DIR,
        split="train",
        transform=COMMON_TRANSFORM,
        download=False,
    )
    test_dataset = STL10(
        root=DATA_DIR,
        split="test",
        transform=COMMON_TRANSFORM,
        download=False,
    )

    datasets = {
        "train": Subset(train_dataset, splits["train_indices"]),
        "validation": Subset(train_dataset, splits["validation_indices"]),
        "test": Subset(test_dataset, splits["test_indices"]),
    }

    # shuffling is disabled so saved features keep the split index order
    return {
        name: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }


def extract_features(model, loader: DataLoader, device: torch.device):
    feature_batches = []
    label_batches = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            features = model(images)

            feature_batches.append(features.cpu())
            label_batches.append(labels.long())

    return torch.cat(feature_batches), torch.cat(label_batches)


def extract_all_features(device: torch.device) -> None:
    loaders = make_loaders()

    # models are processed one at a time to keep gpu memory usage low
    for name in BACKBONES:
        output_file = FEATURE_DIR / f"{name}.pt"

        # existing files let the script resume after an interrupted run
        if output_file.exists():
            print(f"skipping {name} because its features already exist")
            continue

        print(f"extracting {name} features")
        model = load_backbone(name, device)
        saved_data = {}

        for split_name, loader in loaders.items():
            features, labels = extract_features(model, loader, device)
            expected_dim = FEATURE_DIMS[name]

            if features.shape[1] != expected_dim:
                raise ValueError(f"expected {expected_dim} features but found {features.shape[1]}")

            saved_data[f"{split_name}_features"] = features
            saved_data[f"{split_name}_labels"] = labels
            print(f"{split_name} {tuple(features.shape)}")

        torch.save(saved_data, output_file)
        print(f"saved {output_file}")

        del model
        del saved_data
        gc.collect()
        torch.cuda.empty_cache()


def validation_accuracy(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    head.eval()

    with torch.inference_mode():
        predictions = head(features.to(device)).argmax(dim=1).cpu()

    return (predictions == labels).float().mean().item()


def load_or_train_head(name: str, data: dict, device: torch.device):
    checkpoint_file = CHECKPOINT_DIR / f"{name}_head.pt"

    # resetting the seed gives every classifier comparison the required seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    head = nn.Linear(FEATURE_DIMS[name], NUM_CLASSES).to(device)

    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        head.load_state_dict(checkpoint["model_state"])
        print(f"loaded {checkpoint_file}")
        return head, checkpoint["best_epoch"], checkpoint["best_val_accuracy"]

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(data["train_features"], data["train_labels"]),
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(features), labels)
            loss.backward()
            optimizer.step()

        accuracy = validation_accuracy(
            head,
            data["validation_features"],
            data["validation_labels"],
            device,
        )
        print(f"{name} epoch {epoch} validation accuracy {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == PATIENCE:
            print(f"stopping {name} early")
            break

    head.load_state_dict(best_state)
    checkpoint = {
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
    }
    torch.save(checkpoint, checkpoint_file)
    print(f"saved {checkpoint_file}")

    return head, best_epoch, best_accuracy


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)

    return {
        "top1_accuracy": (predictions == labels).float().mean().item(),
        "macro_f1": f1_score(labels.numpy(), predictions.numpy(), average="macro"),
        "mean_maximum_confidence": confidence.mean().item(),
    }


def evaluate_head(name: str, head: nn.Module, data: dict, device: torch.device) -> dict:
    prediction_file = PREDICTION_DIR / f"{name}_head.pt"

    if prediction_file.exists():
        saved_output = torch.load(prediction_file, map_location="cpu")
        return metrics_from_logits(saved_output["logits"], saved_output["labels"])

    head.eval()

    with torch.inference_mode():
        logits = head(data["test_features"].to(device)).cpu()

    labels = data["test_labels"]
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    torch.save(
        {
            "logits": logits,
            "labels": labels,
            "predictions": predictions,
            "confidence": confidence,
        },
        prediction_file,
    )

    return metrics_from_logits(logits, labels)


def evaluate_zero_shot_clip(device: torch.device) -> dict:
    prediction_file = PREDICTION_DIR / "clip_zero_shot.pt"

    if prediction_file.exists():
        saved_output = torch.load(prediction_file, map_location="cpu")
        return metrics_from_logits(saved_output["logits"], saved_output["labels"])

    data = torch.load(FEATURE_DIR / "clip_vit_b_32.pt", map_location="cpu")
    clip_model = load_backbone("clip_vit_b_32", device)
    tokenizer = get_clip_tokenizer()
    prompts = [f"a photo of a {class_name}." for class_name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        text_features = clip_model.encode_text(tokens)
        image_features = data["test_features"].to(device)
        logits = (clip_model.logit_scale() * image_features @ text_features.T).cpu()

    labels = data["test_labels"]
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    torch.save(
        {
            "logits": logits,
            "labels": labels,
            "predictions": predictions,
            "confidence": confidence,
        },
        prediction_file,
    )

    del clip_model
    torch.cuda.empty_cache()

    return metrics_from_logits(logits, labels)


def train_and_evaluate_heads(device: torch.device) -> dict:
    results = {}

    for name in BACKBONES:
        data = torch.load(FEATURE_DIR / f"{name}.pt", map_location="cpu")
        head, best_epoch, best_accuracy = load_or_train_head(name, data, device)
        metrics = evaluate_head(name, head, data, device)
        metrics["best_epoch"] = best_epoch
        metrics["best_validation_accuracy"] = best_accuracy
        results[f"{name}_head"] = metrics

        del head
        del data
        torch.cuda.empty_cache()

    results["clip_zero_shot"] = evaluate_zero_shot_clip(device)
    return results


def main() -> None:
    torch.manual_seed(SEED)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    extract_all_features(device)
    results = train_and_evaluate_heads(device)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"saved {METRICS_FILE}")


if __name__ == "__main__":
    main()
