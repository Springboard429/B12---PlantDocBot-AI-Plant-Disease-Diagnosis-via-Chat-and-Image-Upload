from pydantic import BaseModel, Field


class TextPredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "distilbert"


class ImagePredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "simplecnn"


class TextInput(BaseModel):
    text: str = Field(..., max_length=512, description="Plant disease description text")


class CombinedPredictionResponse(BaseModel):
    image_prediction: ImagePredictionResponse | None = None
    text_prediction: TextPredictionResponse | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class ClassesResponse(BaseModel):
    classes: list[str]


class ErrorResponse(BaseModel):
    detail: str
