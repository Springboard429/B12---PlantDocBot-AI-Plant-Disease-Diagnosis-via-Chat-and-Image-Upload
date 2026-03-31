from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict
from api.services.image_model import load_image_model
from api.services.text_model import load_text_model

app = FastAPI(title="Plant Disease Detection API")


# =========================
# CORS (REQUIRED)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD MODELS AT STARTUP
# =========================
@app.on_event("startup")
def load_models():
    load_image_model("models/image_model/best_cnn_plantdoc.pth")
    load_text_model("models/text_model/plant_disease_modelc")
    print("✅ All models loaded successfully")


# =========================
# HEALTH (REQUIRED FOR FRONTEND)
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": True
    }


# =========================
# CLASSES (REQUIRED FOR FRONTEND)
# =========================
@app.get("/classes")
def get_classes():
    return {
        "classes": [
            "Tomato Bacterial spot",
            "Tomato Leaf Mold",
            "Tomato mosaic virus"
        ]
    }


# =========================
# INCLUDE ROUTES
# =========================
app.include_router(predict.router)


# =========================
# ROOT CHECK
# =========================
@app.get("/")
def home():
    return {"message": "API is running"}
