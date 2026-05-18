import torch
from torch import nn


class MNISTCNN(nn.Module):
    """MNIST Convolutional Neural Network

    Architecture:
        1. Conv Layer (1,32) -> ReLu -> MaxPool (2,2).
        2. Conv Layer (32, 64) -> Relu -> MaxPool (2,2) -> Dropout (0.25).
        3. Flatten (64 * 7 * 7) -> FC1 (64 * 7 * 7, 128) -> ReLu -> Dropout (0.5).
        4. FC2 (128, 10) -> Logits.

    Input Shape:
        tensor: Shape (batch_size, 1, 28, 28).

    Returns:
        logits: 10 logits for the 10 digits to categorize.
    """

    def __init__(self) -> None:
        """Initialize convolutional and fully connected layers."""
        # Call Super Class Constructor
        super().__init__()

        # --- Building Blocks ---
        # Fully Connected
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7, out_features=128
        )  # Transition between matrix and vectors.
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Logits output

        # Convolutional
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )  # Entry point
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # Following processing point

        # Pooling
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.5)

        # ReLu
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute digit logits for a batch shaped as (N, 1, 28, 28)."""
        # First pass
        x = self.relu(self.conv1(x))
        x = self.pooling(x)

        # Second pass
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.dropout2d(x)

        # Third pass (classifier)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # Fourth pass (output logits)
        x = self.fc2(x)

        # Logits
        return x
