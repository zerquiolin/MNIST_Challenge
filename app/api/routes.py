import logging

from fastapi import APIRouter, HTTPException

from app.schemas.payload import ErrorResponse, PredictRequest, PredictResponse
from app.services.inference import predict_digit
from app.services.validation import process_base64_image, validate_base64_image

router = APIRouter()
logger = logging.getLogger("mnist-classifier")


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """Preprocess a base64 image and classify it with the MNIST CNN."""
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 field is empty.")

    ok, message = validate_base64_image(payload.image_base64)
    if not ok:
        raise HTTPException(status_code=422, detail=message)

    try:
        tensor_img = process_base64_image(payload.image_base64)
        prediction, confidence = predict_digit(tensor_img)
        return PredictResponse(prediction=prediction, confidence=confidence)

    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        logger.exception("Model loading or inference error")
        raise HTTPException(
            status_code=500,
            detail="Internal model error.",
        ) from e
    except Exception as e:
        logger.exception("Unexpected inference error")
        raise HTTPException(
            status_code=500, detail="Unexpected error during prediction."
        ) from e
