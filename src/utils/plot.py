from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

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
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
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

        ax.set_title(
            f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=11
        )

    plt.tight_layout()
    plt.show()
