from pydantic import BaseModel, Field


# =========================
# IMAGE RESPONSE
# =========================
class ImagePredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "simplecnn"


# =========================
# TEXT RESPONSE
# =========================
class TextPredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "distilbert"


# =========================
# TEXT INPUT (optional use)
# =========================
class TextInput(BaseModel):
    text: str = Field(..., max_length=512, description="Plant disease description text")


# =========================
# COMBINED RESPONSE
# =========================
class CombinedPredictionResponse(BaseModel):
    image_prediction: ImagePredictionResponse | None = None
    text_prediction: TextPredictionResponse | None = None


# =========================
# ERROR RESPONSE
# =========================
class ErrorResponse(BaseModel):
    detail: str


# =========================
# HEALTH CHECK (optional)
# =========================
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# =========================
# CLASSES RESPONSE (optional)
# =========================
class ClassesResponse(BaseModel):
    classes: list[str]