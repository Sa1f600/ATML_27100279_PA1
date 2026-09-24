"""per-class accuracy, dominant confusions, and traceable failure examples."""

import numpy as np
from sklearn.metrics import confusion_matrix

from shared.pacs_protocol import CLASSES


def class_analysis(labels, predictions, baseline, paths):
    matrix = confusion_matrix(labels, predictions, labels=list(range(7)))
    rows, failures = [], []
    for index, name in enumerate(CLASSES):
        selected = labels == index
        accuracy = float((predictions[selected] == index).mean())
        previous = float((baseline[selected] == index).mean())
        errors = matrix[index].copy()
        errors[index] = 0
        rows.append({"class": name, "count": int(selected.sum()), "accuracy": accuracy,
                     "delta_pp": 100 * (accuracy - previous),
                     "dominant_confusion": CLASSES[errors.argmax()] if errors.sum() else None,
                     "dominant_confusion_count": int(errors.max())})
        # retain fixed-order failures plus examples gained or lost relative to source-only
        masks = {"failure": selected & (predictions != labels),
                 "gained": selected & (predictions == labels) & (baseline != labels),
                 "lost": selected & (predictions != labels) & (baseline == labels)}
        for kind, mask in masks.items():
            for i in np.flatnonzero(mask)[:3]:
                failures.append({"kind": kind, "path": paths[i], "true_class": name,
                                 "predicted_class": CLASSES[predictions[i]],
                                 "source_only_prediction": CLASSES[baseline[i]]})
    return rows, matrix, failures
