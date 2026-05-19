import pandas as pd
import torch


def inspect_dataloader(loader, name: str) -> None:
    images, labels = next(iter(loader))

    print(f"\n{name} loader")
    print("-" * 40)
    print(f"Number of batches: {len(loader)}")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Image min: {images.min().item():.4f}")
    print(f"Image max: {images.max().item():.4f}")
    print(f"Labels in first batch: {labels.unique().tolist()}")


def get_class_distribution(loader, name: str) -> pd.DataFrame:
    all_labels = []

    for _, labels in loader:
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)

    counts = torch.bincount(all_labels, minlength=10)
    percentages = counts / counts.sum() * 100

    df = pd.DataFrame(
        {
            "split": name,
            "class": list(range(10)),
            "count": counts.tolist(),
            "percentage": percentages.tolist(),
        }
    )

    return df


def denormalize_mnist(tensor: torch.Tensor) -> torch.Tensor:
    mean = 0.1307
    std = 0.3081
    return tensor * std + mean
