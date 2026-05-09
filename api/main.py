from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict
from api.services.image_model import load_image_model
from api.services.text_model import load_text_model

app = FastAPI(title="PlantDocBot API")

# CORS (important for frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
@app.on_event("startup")
def load_models():
    load_image_model("models/best_model_full.pth")
    load_text_model("distilbert_model")
    print("✅ Models loaded")

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(predict.router)