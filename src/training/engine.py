import torch
import numpy as np
from numpy.typing import NDArray
from src.utils.lib.schema import TrainHistory
from tqdm import tqdm


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    # Metadata
    total = 0
    correct = 0
    running_loss = 0.0

    # Loop
    for X, y in tqdm(train_dataloader, desc="Training", leave=False, unit="batch"):
        # Move tensors to device
        X, y = X.to(device), y.to(device)
        # Reset gradients
        # optimizer.zero_grad(set_to_none=True)  # Small optimization
        optimizer.zero_grad()
        # Forward pass
        output = model(X)
        # Compute loss
        loss = criterion(output, y)
        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * X.size(
            0
        )  # Loss mean * X size (remove the partial average)
        predicted = output.argmax(dim=1)

        correct += (predicted == y).sum().item()
        total += y.size(0)

    # Metadata
    if total == 0:
        raise ValueError("Training dataloader is empty.")

    epoch_accuracy = correct / total
    epoch_loss = running_loss / total

    return epoch_loss, epoch_accuracy


def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model for one epoch and return loss and accuracy."""
    model.eval()
    # Metadata
    total = 0
    correct = 0
    running_loss = 0.0

    # Loop
    with torch.inference_mode():
        for X, y in tqdm(eval_dataloader, desc="Evaluation", leave=False, unit="batch"):
            # Move tensors to device
            X, y = X.to(device), y.to(device)
            # Prediction
            output = model(X)
            # Evaluation
            loss = criterion(output, y)

            # Metrics
            running_loss += loss.item() * X.size(
                0
            )  # Loss mean * X size (remove the partial average)
            predicted = output.argmax(dim=1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    # Metadata
    if total == 0:
        raise ValueError("Evaluation dataloader is empty.")

    epoch_accuracy = correct / total
    epoch_loss = running_loss / total

    return epoch_loss, epoch_accuracy


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 20,
) -> tuple[torch.nn.Module, TrainHistory]:
    """Train a model and collect per-epoch training and validation metrics."""
    # Safe Guard
    assert epochs > 0, "The number of epochs should be greater than 0."

    # Metadata
    history: TrainHistory = TrainHistory()

    # Model Device
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
        # Training Loop
        train_loss, train_accuracy = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            device=device,
        )
        # Evaluation Loop
        eval_loss, eval_accuracy = eval_epoch(
            model=model,
            criterion=criterion,
            eval_dataloader=eval_dataloader,
            device=device,
        )
        # Update History
        history.epoch.append(epoch + 1)
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.eval_loss.append(eval_loss)
        history.eval_accuracy.append(eval_accuracy)

        # Logs
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"eval_loss={eval_loss:.4f} | "
            f"eval_acc={eval_accuracy:.4f}"
        )

    # Return trained Model
    return model, history


def test_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.int64], torch.Tensor]:
    """Return test labels, predictions, and images for reporting utilities."""
    model.eval()

    # Metadata
    preds = []
    labels = []
    images = []

    with torch.inference_mode():
        for X, y in test_dataloader:
            # Set tensors to device
            X, y = X.to(device), y.to(device)

            # Predictions
            output = model(X)
            predicted = output.argmax(dim=1)

            # Extend Metadata
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
            images.extend(X.cpu())

    if not images:
        raise ValueError("Test dataloader is empty.")

    return np.array(labels), np.array(preds), torch.cat(images)
