from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io

from api.services.image_model import predict_image
from api.services.text_model import predict_text

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# =========================
# IMAGE PREDICTION (FIXED FORMAT)
# =========================
@router.post("/image")
async def predict_image_route(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = predict_image(image)

    return {
        "disease": result.get("disease") or result.get("label"),
        "confidence": float(
            result.get("confidence") if result.get("confidence") is not None
            else result.get("score", 0) * 100
        ),
        "model": "cnn"
    }


# =========================
# TEXT PREDICTION (FIXED - PREVENTS CRASH)
# =========================
@router.post("/text")
async def predict_text_route(text: str = Form(...)):
    result = predict_text(text)

    return {
        "disease": result.get("disease") or result.get("label"),
        "confidence": float(
            result.get("confidence") if result.get("confidence") is not None
            else result.get("score", 0) * 100
        ),
        "model": "distilbert"
    }


# =========================
# COMBINED PREDICTION (FIXED)
# =========================
@router.post("/combined")
async def predict_combined(
    file: UploadFile = File(default=None),
    text: str = Form(default=None),
):
    if file is None and text is None:
        raise HTTPException(status_code=400, detail="Provide image or text")

    response = {
        "image_prediction": None,
        "text_prediction": None
    }

    # IMAGE PART
    if file is not None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(image)

        response["image_prediction"] = {
            "disease": result.get("disease") or result.get("label"),
            "confidence": float(
                result.get("confidence") if result.get("confidence") is not None
                else result.get("score", 0) * 100
            ),
            "model": "cnn"
        }

    # TEXT PART
    if text is not None:
        result = predict_text(text)

        response["text_prediction"] = {
            "disease": result.get("disease") or result.get("label"),
            "confidence": float(
                result.get("confidence") if result.get("confidence") is not None
                else result.get("score", 0) * 100
            ),
            "model": "distilbert"
        }

    return response