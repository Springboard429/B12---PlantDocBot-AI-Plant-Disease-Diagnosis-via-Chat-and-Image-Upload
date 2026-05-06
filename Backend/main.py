import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# ✅ Your model imports
from cnninference import predict_image, class_names
from textinference import predict_disease

# ---------- APP ----------
app = FastAPI(title="Plant Disease API")

# ---------- CORS (IMPORTANT for React) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True
    }

# ---------- CLASSES ----------
@app.get("/classes")
def get_classes():
    return {
        "classes": class_names
    }

# ---------- TEXT ----------
@app.post("/predict/text")
def predict_text_api(text: str = Form(...)):
    try:
        result = predict_disease(text)

        # handle flexible return
        if isinstance(result, tuple):
            disease, confidence = result
        else:
            disease = result
            confidence = 0.0

        return {
            "disease": str(disease),
            "confidence": float(confidence) * 100,
            "model": "text-model"
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- IMAGE ----------
@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...)):
    try:
        # save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        prediction, confidence = predict_image(temp_path)

        os.remove(temp_path)

        return {
            "disease": str(prediction),
            "confidence": float(confidence),
            "model": "cnn-model"
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- COMBINED ----------
@app.post("/predict/combined")
async def predict_combined(
    file: UploadFile = File(None),
    text: str = Form("")
):
    image_result = None
    text_result = None

    try:
        # ---------- IMAGE ----------
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                contents = await file.read()
                temp.write(contents)
                temp_path = temp.name

            prediction, confidence = predict_image(temp_path)
            os.remove(temp_path)

            image_result = {
                "disease": str(prediction),
                "confidence": float(confidence),
                "model": "cnn-model"
            }

        # ---------- TEXT ----------
        if text.strip():
            result = predict_disease(text)

            if isinstance(result, tuple):
                disease, confidence = result
            else:
                disease = result
                confidence = 0.0

            text_result = {
                "disease": str(disease),
                "confidence": float(confidence) * 100,
                "model": "text-model"
            }

        return {
            "image_prediction": image_result,
            "text_prediction": text_result
        }

    except Exception as e:
        return {"error": str(e)}