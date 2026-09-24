"""classification metrics and deterministic feature extraction."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels, predictions):
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(7)),
                                      average="macro", zero_division=0))}


@torch.inference_mode()
def collect(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    features, logits, labels = [], [], []
    for images, target in loader:
        feature = backbone(images.to(device))
        features.append(feature.cpu().numpy())
        logits.append(classifier(feature).cpu().numpy())
        labels.append(target.numpy())
    return {"features": np.concatenate(features), "logits": np.concatenate(logits),
            "labels": np.concatenate(labels)}
