"""small helpers for reproducible runs and saved evidence."""

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def code_hash():
    paths = sorted(p for directory in (REPO / "shared", REPO / "task2")
                   for p in directory.rglob("*") if p.suffix in {".py", ".yaml"})
    return hashlib.sha256("".join(str(p.relative_to(REPO)) + digest(p)
                                  for p in paths).encode()).hexdigest()


def experiment_configs():
    directory = REPO / "task2/configs"
    base = yaml.safe_load((directory / "base.yaml").read_text())
    configs = {name: base | yaml.safe_load((directory / f"{name}.yaml").read_text())
               for name in ("source_only", "dan", "dann", "cdan")}
    for weight in (0.1, 10.0):
        configs[f"dan_lambda_{weight:g}"] = configs["dan"] | {"mmd_weight": weight}
    return configs
