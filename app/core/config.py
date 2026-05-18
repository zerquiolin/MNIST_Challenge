from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    api_title: str = Field(default="MNIST Digit Classifier API", alias="API_TITLE")
    api_version: str = Field(default="1.0.0", alias="API_VERSION")
    api_description: str = Field(
        default="API for Handwritten Digit Classification using a PyTorch CNN trained on MNIST.",
        alias="API_DESCRIPTION",
    )
    environment: str = Field(default="development", alias="ENVIRONMENT")
    allowed_origins: list[str] = Field(default=["*"], alias="ALLOWED_ORIGINS")
    allowed_methods: list[str] = Field(default=["*"], alias="ALLOWED_METHODS")
    allowed_headers: list[str] = Field(default=["*"], alias="ALLOWED_HEADERS")

    # Path to the exported model weights
    model_path: str = Field(
        default=str(PROJECT_ROOT / "artifacts" / "mnist_classifier.pt"),
        alias="MODEL_PATH",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
