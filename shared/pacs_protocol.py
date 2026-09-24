"""shared, deterministic source splits for tasks 2 and 3."""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 6304
SOURCES = ("photo", "art_painting", "cartoon")
TARGET = "sketch"
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
SPLIT_FILE = Path(__file__).parent / "splits/pacs_sketch_seed6304.json"


def image_paths(root, domain):
    return sorted(p.relative_to(root).as_posix() for p in (root / domain).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def class_label(path):
    return CLASSES.index(Path(path).parts[1])


def make_splits(root, output=SPLIT_FILE):
    root, output = Path(root), Path(output)
    result = {"seed": SEED, "classes": list(CLASSES), "sources": {}}
    for domain in SOURCES:
        paths = image_paths(root, domain)
        if not paths:
            raise ValueError(f"no images in {root / domain}")
        labels = [class_label(p) for p in paths]
        if set(labels) != set(range(7)):
            raise ValueError(f"missing classes in {domain}")
        train, val = train_test_split(paths, test_size=0.2, stratify=labels,
                                     random_state=SEED)
        result["sources"][domain] = {"train": sorted(train), "val": sorted(val)}
    # target paths carry no numeric labels; training never decodes their class folders
    result["target"] = image_paths(root, TARGET)
    if not result["target"]:
        raise ValueError(f"no images in {root / TARGET}")
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError("existing split differs from this dataset; do not overwrite it")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default=str(SPLIT_FILE))
    args = parser.parse_args()
    make_splits(args.data_root, args.output)
