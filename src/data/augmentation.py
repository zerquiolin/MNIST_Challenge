import torchvision.transforms as transforms


def train_transforms() -> transforms.Compose:
    """Return augmentation and normalization transforms for training images."""
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomInvert(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def test_transforms() -> transforms.Compose:
    """Return deterministic resize and normalization transforms for evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
