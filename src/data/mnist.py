import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.data.augmentation import test_transforms, train_transforms


def load_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    torch_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int | list[str]]]:
    """Load MNIST datasets and create train, validation, and test dataloaders.

    The MNIST training split is divided into a training subset and a validation
    subset according to `val_split`. Training data uses the training transform
    pipeline, while the evaluation and test set uses the evaluation transform pipeline.

    Args:
        data_dir: Directory where the MNIST dataset is stored or downloaded.
        batch_size: Number of samples per batch.
        val_split: Fraction of the original training set used for validation. Must be between 0 and 1.
        torch_seed: Random seed used to make the train/validation split reproducible.

    Returns:
        A tuple containing:
            train_loader: DataLoader for the training subset.
            val_loader: DataLoader for the validation subset.
            test_loader: DataLoader for the test set.
            metadata: Dictionary with dataset sizes and class labels.
    """
    if not 0 <= val_split <= 1:
        raise ValueError("The validation split must be between [0,1].")

    # Generator
    generator = torch.Generator().manual_seed(torch_seed)

    # Make sure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Train dataset
    training_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transforms()
    )
    training_dataset_validation = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=test_transforms()
    )
    # Test dataset
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=test_transforms()
    )

    # Split train/val sizes
    val_size = int(len(training_dataset) * val_split)
    train_size = len(training_dataset) - val_size

    # Split train/val dataset
    train_subset, validation_subset = random_split(
        dataset=range(len(training_dataset)),
        lengths=[train_size, val_size],
        generator=generator,
    )
    train_dataset = torch.utils.data.Subset(
        training_dataset, indices=train_subset.indices
    )
    validation_dataset = torch.utils.data.Subset(
        training_dataset_validation, indices=validation_subset.indices
    )

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Metadata
    metadata = {
        "train_size": train_size,
        "test_size": len(test_dataset),
        "validation_size": val_size,
        "classes": training_dataset.classes,
    }

    return train_loader, val_loader, test_loader, metadata
