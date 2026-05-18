from pathlib import Path

import torch


def save_model(model: torch.nn.Module, output_dir: str | Path) -> None:
    """Save a model state dictionary into the provided output directory."""
    # Make sure data directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Save model
    torch.save(model.state_dict(), output_path / f"{model.__class__.__name__}.pt")


def load_model(
    model: type[torch.nn.Module] | torch.nn.Module,
    model_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a PyTorch state dictionary into a model class or model instance."""
    # Create instance of class
    if isinstance(model, type):
        model = model()
    # Check class coherence
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Expected a torch.nn.Module class or instance, got {type(model).__name__}."
        )
    # Update wegihts, device, and mode
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
