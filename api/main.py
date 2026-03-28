from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict
from api.models import HealthResponse, ClassesResponse
from api.services.image_model import get_image_model
from api.services.text_model import get_text_model

DISEASE_CLASSES = [
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two spotted spider mite",
    "Tomato Target Spot",
    "Tomato YellowLeaf Curl Virus",
    "Tomato healthy",
    "Tomato mosaic virus",
]

_models_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_ready
    try:
        get_image_model()
        get_text_model()
        _models_ready = True
    except Exception as e:
        _models_ready = False
        print(f"Warning: Models failed to load at startup: {e}")
    yield
    _models_ready = False


app = FastAPI(
    title="PlantDoc API",
    description="Plant disease detection API powered by SimpleCNN (image) and DistilBERT (text).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", models_loaded=_models_ready)


@app.get("/classes", response_model=ClassesResponse)
async def list_classes():
    return ClassesResponse(classes=DISEASE_CLASSES)
