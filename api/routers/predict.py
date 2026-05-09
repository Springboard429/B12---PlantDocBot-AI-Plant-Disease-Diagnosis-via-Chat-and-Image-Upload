from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io

from api.services.image_model import predict_image
from api.services.text_model import predict_text

router = APIRouter(prefix="/predict", tags=["predict"])

# IMAGE
@router.post("/image")
async def image_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    return predict_image(image)


# TEXT
@router.post("/text")
async def text_predict(text: str = Form(...)):
    return predict_text(text)


# COMBINED
@router.post("/combined")
async def combined_predict(
    file: UploadFile = File(default=None),
    text: str = Form(default=None),
):
    if file is None and text is None:
        raise HTTPException(status_code=400, detail="Provide input")

    response = {}

    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        response["image"] = predict_image(image)

    if text:
        response["text"] = predict_text(text)

    return response