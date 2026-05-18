import argparse
from pathlib import Path

import torch

from src.data.mnist import load_mnist_dataloaders
from src.models.mnist_cnn import MNISTCNN
from src.training.engine import train
from src.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "mnist_classifier.pt"


def generate_model(
    output_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 22,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> Path:
    """Train the MNIST classifier and save its weights to `output_path`."""
    train_loader, eval_loader, _, _ = load_mnist_dataloaders(batch_size=batch_size)
    model = MNISTCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    device = get_device()

    trained_model, _ = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        epochs=epochs,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model artifact generation."""
    parser = argparse.ArgumentParser(description="Train and export the MNIST model artifact.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path where the model artifact will be written. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--epochs", type=int, default=22, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Optimizer learning rate."
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for creating the trained model artifact."""
    args = parse_args()
    output_path = generate_model(
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(f"Model artifact written to {output_path}")


if __name__ == "__main__":
    main()
