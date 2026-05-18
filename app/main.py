import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

logger = logging.getLogger("mnist-classifier")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=settings.allowed_origins,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )

    @app.middleware("http")
    async def add_process_time_header(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add request processing time to every HTTP response."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(
            f"{request.method} {request.url.path} completed in {process_time:.4f}s"
        )
        return response

    app.include_router(router)

    @app.get("/")
    def root() -> dict[str, str]:
        """Return a basic API welcome message."""
        return {
            "message": "Welcome to the MNIST Digit Classifier API. Use POST /predict to classify digits."
        }

    @app.get("/status")
    def status() -> dict[str, str]:
        """Return model artifact availability for health checks."""
        return (
            {"Status": "Ok"}
            if Path(settings.model_path).exists()
            else {"Status": "Error", "Message": "Model not found."}
        )

    return app


app = create_app()
