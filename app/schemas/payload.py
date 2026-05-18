from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for digit prediction."""

    image_base64: str = Field(
        ...,
        max_length=5_000_000,
        description="Base64 encoded string of the handwritten digit image. Max size ~3.7MB.",
    )


class PredictResponse(BaseModel):
    """Prediction result returned by the API."""

    prediction: int = Field(..., description="The predicted digit class (0-9).")
    confidence: float | None = Field(
        None, description="The confidence probability of the prediction."
    )


class ErrorResponse(BaseModel):
    """Error response shape returned by the API."""

    detail: str = Field(..., description="A detailed message explaining the error.")
