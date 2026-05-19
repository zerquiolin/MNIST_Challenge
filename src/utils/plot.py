import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import HTML, display
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.eda import denormalize_mnist
from src.utils.lib.schema import TrainHistory


def plot_train_history(history: TrainHistory) -> None:
    """Plot training and validation loss/accuracy curves."""
    # Figure
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history.epoch, history.train_loss, "bo-", label="Training loss")
    ax1.plot(history.epoch, history.eval_loss, "ro-", label="Validation loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history.epoch, history.train_accuracy, "bo-", label="Training acc")
    ax2.plot(history.epoch, history.eval_accuracy, "ro-", label="Validation acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[int | str],
) -> None:
    """Plot a confusion matrix for predicted and true labels."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def print_classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    classes: Sequence[int | str],
) -> None:
    """Print precision, recall, and F1 metrics for each class."""
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))


def plot_predictions(
    images: torch.Tensor,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n: int,
) -> None:
    """Plot sample images with true and predicted labels."""
    fig = plt.figure(figsize=(16, 8))

    for idx in range(n):
        ax = fig.add_subplot(4, 8, idx + 1, xticks=[], yticks=[])
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap="gray")

        true_label = y_true[idx]
        pred_label = y_pred[idx]

        color = "green" if true_label == pred_label else "red"

        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(df: pd.DataFrame) -> None:
    splits = df["split"].unique()
    class_colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), sharey=False)

    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits, strict=False):
        subset = df[df["split"] == split]

        colors = [
            class_colors[int(class_label) % len(class_colors)] for class_label in subset["class"]
        ]
        bars = ax.bar(subset["class"], subset["count"], color=colors)
        ax.set_xlabel("Digit class")
        ax.set_title(f"Class distribution - {split}")
        ax.set_xticks(range(10))

        max_count = subset["count"].max()
        ax.set_ylim(0, max_count * 1.15)

        for bar, percentage in zip(bars, subset["percentage"], strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes:
        ax.set_ylabel("Count")
    fig.suptitle("Class distribution by split")
    fig.tight_layout()
    plt.show()


def plot_samples(loader, name: str, n: int = 12) -> None:
    images, labels = next(iter(loader))

    images = denormalize_mnist(images)
    images = images.clamp(0, 1)

    n = min(n, len(images))

    plt.figure(figsize=(12, 4))

    for i in range(n):
        plt.subplot(2, 6, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")

    plt.suptitle(f"Sample images - {name}")
    plt.tight_layout()
    plt.show()


def plot_hw_samples(
    image_paths: Sequence[str | Path],
    n: int = 16,
    cols: int = 8,
) -> None:
    """Plot handwritten images with optional true and predicted labels."""
    n = min(n, len(image_paths))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(2 * cols, 2 * rows))

    for i in range(n):
        image_path = Path(image_paths[i])
        img = plt.imread(image_path)

        ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap="gray")

        label = image_path.stem[0]
        ax.set_title(f"Label: {label}", fontsize=10)

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_nn_architecture(svg_path: str) -> None:

    svg_path = Path(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")

    # Remove fixed width/height from original SVG tag
    svg_text = re.sub(r'\swidth="[^"]*"', "", svg_text, count=1)
    svg_text = re.sub(r'\sheight="[^"]*"', "", svg_text, count=1)

    # Make SVG responsive inside the container
    svg_text = re.sub(
        r"<svg ",
        '<svg style="width:100%; height:auto; display:block;" ',
        svg_text,
        count=1,
    )

    display(
        HTML(f"""
    <div style="
        width:100%;
        height:auto;
        background:white;
        padding:0px;
        border-radius:8px;
        margin:0 auto;
    ">
    {svg_text}
    </div>
    """)
    )
