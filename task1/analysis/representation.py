"""create joint clean and transformed t-sne visualizations"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from task1.analysis.feature_similarity import BACKBONES, visualization_pairs


SEED = 6304
PERPLEXITY = 30
PCA_COMPONENTS = 50
TSNE_ITERATIONS = 1000
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
MODEL_LABELS = {
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
    "clip_vit_b_32": "CLIP ViT-B/32",
}

TASK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = TASK_DIR / "results" / "representation_analysis"
SETTINGS_FILE = OUTPUT_DIR / "tsne_settings.json"


def joint_embedding(clean_features, transformed_features) -> np.ndarray:
    combined = np.concatenate(
        [clean_features.numpy(), transformed_features.numpy()],
        axis=0,
    )

    # normalize before fitting so the projection emphasizes feature direction
    combined = normalize(combined, norm="l2")
    components = min(PCA_COMPONENTS, combined.shape[0] - 1, combined.shape[1])
    reduced = PCA(n_components=components, random_state=SEED).fit_transform(combined)

    # fit one projection to both conditions so their coordinates are comparable
    return TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        max_iter=TSNE_ITERATIONS,
        random_state=SEED,
    ).fit_transform(reduced)


def plot_backbone(backbone: str) -> None:
    pairs = visualization_pairs(backbone)
    figure, axes = plt.subplots(1, len(pairs), figsize=(20, 5))
    colors = ListedColormap(plt.get_cmap("tab10").colors)
    color_norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), colors.N)
    last_scatter = None

    for axis, (condition, pair) in zip(axes, pairs.items()):
        clean_features, transformed_features, labels = pair
        embedding = joint_embedding(clean_features, transformed_features)
        sample_count = len(labels)
        label_values = labels.numpy()

        last_scatter = axis.scatter(
            embedding[:sample_count, 0],
            embedding[:sample_count, 1],
            c=label_values,
            cmap=colors,
            norm=color_norm,
            marker="o",
            s=16,
            alpha=0.55,
            linewidths=0,
        )
        axis.scatter(
            embedding[sample_count:, 0],
            embedding[sample_count:, 1],
            c=label_values,
            cmap=colors,
            norm=color_norm,
            marker="x",
            s=18,
            alpha=0.75,
            linewidths=0.7,
        )
        axis.set_title(condition)
        axis.set_xticks([])
        axis.set_yticks([])

    condition_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="gray",
            markeredgecolor="none",
            label="clean",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            linestyle="none",
            label="transformed",
            markersize=7,
        ),
    ]
    axes[0].legend(handles=condition_legend, loc="best", fontsize=8)
    colorbar = figure.colorbar(
        last_scatter,
        ax=axes,
        ticks=range(10),
        fraction=0.02,
        pad=0.02,
    )
    colorbar.ax.set_yticklabels(CLASS_NAMES)
    figure.suptitle(f"{MODEL_LABELS[backbone]} joint clean/transformed t-SNE")
    figure.subplots_adjust(left=0.03, right=0.88, bottom=0.08, top=0.86, wspace=0.12)
    output_file = OUTPUT_DIR / f"tsne_{backbone}.png"
    figure.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = {
        "method": "t-SNE",
        "seed": SEED,
        "perplexity": PERPLEXITY,
        "iterations": TSNE_ITERATIONS,
        "preprocessing": "l2 normalization followed by joint 50-component PCA",
        "conditions": [
            "grayscale",
            "cue conflict",
            "translation 32px right",
            "patch shuffle",
        ],
        "note": "each panel fits one projection to combined clean and transformed features",
    }

    with SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    for backbone in BACKBONES:
        plot_backbone(backbone)

    print(f"saved {SETTINGS_FILE}")


if __name__ == "__main__":
    main()
