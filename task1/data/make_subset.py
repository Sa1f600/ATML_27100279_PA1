import json
from pathlib import Path

import numpy as np
from torchvision.datasets import STL10


SEED = 6304
TEST_IMAGES_PER_CLASS = 50

TASK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK_DIR.parent / "data"
OUTPUT_FILE = TASK_DIR / "results" / "splits.json"


# create a stratified training and validation split
def make_train_val_split(labels: np.ndarray) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(SEED)
    train_indices = []
    val_indices = []

    # split each class separately to preserve class balance
    for class_id in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        split_position = int(len(class_indices) * 0.8)
        train_indices.extend(class_indices[:split_position].tolist())
        val_indices.extend(class_indices[split_position:].tolist())

    return train_indices, val_indices


# create a balanced subset from the official test split
def make_test_subset(labels: np.ndarray) -> list[int]:
    rng = np.random.default_rng(SEED)
    test_indices = []

    for class_id in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) < TEST_IMAGES_PER_CLASS:
            raise ValueError(f"class {class_id} has fewer than {TEST_IMAGES_PER_CLASS} images")

        # sampling without replacement prevents duplicate images
        selected = rng.choice(class_indices, size=TEST_IMAGES_PER_CLASS, replace=False)
        test_indices.extend(selected.tolist())

    return test_indices


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = STL10(root=DATA_DIR, split="train", download=True)
    test_dataset = STL10(root=DATA_DIR, split="test", download=True)

    train_labels = np.asarray(train_dataset.labels)
    test_labels = np.asarray(test_dataset.labels)

    train_indices, val_indices = make_train_val_split(train_labels)
    test_indices = make_test_subset(test_labels)

    # save the exact indices so every model uses identical images
    splits = {
        "dataset": "STL10",
        "seed": SEED,
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2)

    print(f"saved {len(train_indices)} training indices")
    print(f"saved {len(val_indices)} validation indices")
    print(f"saved {len(test_indices)} test indices")
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
