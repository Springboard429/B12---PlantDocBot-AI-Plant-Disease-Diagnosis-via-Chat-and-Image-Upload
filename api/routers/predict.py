from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image
import io

from api.models import (
    ImagePredictionResponse,
    TextPredictionResponse,
    CombinedPredictionResponse,
    ErrorResponse,
)
from api.services.image_model import get_image_model
from api.services.text_model import get_text_model

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post(
    "/image",
    response_model=ImagePredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPEG or PNG.",
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        model = get_image_model()
        disease, confidence = model.predict(image)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ImagePredictionResponse(disease=disease, confidence=round(confidence, 2))


@router.post(
    "/text",
    response_model=TextPredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_text(text: str = Form(...)):
    try:
        model = get_text_model()
        disease, confidence = model.predict(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TextPredictionResponse(disease=disease, confidence=round(confidence, 2))


@router.post(
    "/combined",
    response_model=CombinedPredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_combined(
    file: UploadFile = File(default=None),
    text: str = Form(default=None),
):
    if file is None and text is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least an image file or a text description.",
        )

    image_response = None
    text_response = None

    if file is not None:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format. Use JPEG or PNG.",
            )
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        try:
            img_model = get_image_model()
            disease, confidence = img_model.predict(image)
            image_response = ImagePredictionResponse(
                disease=disease, confidence=round(confidence, 2)
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    if text is not None:
        try:
            txt_model = get_text_model()
            disease, confidence = txt_model.predict(text)
            text_response = TextPredictionResponse(
                disease=disease, confidence=round(confidence, 2)
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    return CombinedPredictionResponse(
        image_prediction=image_response,
        text_prediction=text_response,
    )
