from pydantic import BaseModel, Field


class TrainHistory(BaseModel):
    """Training and validation metrics collected after each epoch."""

    epoch: list[float] = Field(default_factory=list, description="Epochs recorded.")
    train_accuracy: list[float] = Field(
        default_factory=list,
        description="Training accuracy recorded after each epoch.",
    )
    train_loss: list[float] = Field(
        default_factory=list,
        description="Training loss recorded after each epoch.",
    )
    eval_accuracy: list[float] = Field(
        default_factory=list,
        description="Validation accuracy recorded after each epoch.",
    )
    eval_loss: list[float] = Field(
        default_factory=list,
        description="Validation loss recorded after each epoch.",
    )
