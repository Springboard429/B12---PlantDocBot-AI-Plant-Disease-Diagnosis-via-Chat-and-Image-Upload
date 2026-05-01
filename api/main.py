"""
Plant Doc Bot — FastAPI
Endpoints:
  POST /predict/image  — SimpleCNN image classification
  POST /predict/text   — DistilBERT symptom text classification
  GET  /health         — health check
"""

import io
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent.parent
IMAGE_MODEL_PATH = BASE_DIR / "models" / "image_model" / "best_model.pth"
TEXT_MODEL_PATH  = BASE_DIR / "models" / "text_model"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Doc Bot API",
    description="Plant disease detection via image (SimpleCNN) and text (DistilBERT)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Image transforms ──────────────────────────────────────────────────────────
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# SimpleCNN — matches your saved best_model.pth exactly
# Confirmed keys: features.0, features.3, features.6 (conv weights)
#                 classifier.1, classifier.4, classifier.6 (fc weights)
# Shapes:  features.0: [32,3,3,3]  features.3: [64,32,3,3]  features.6: [128,64,3,3]
#          classifier.1: [512,100352]  classifier.4: [256,512]  classifier.6: [38,256]
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # features.0
            nn.ReLU(),                                       # features.1
            nn.MaxPool2d(2, 2),                              # features.2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # features.3
            nn.ReLU(),                                       # features.4
            nn.MaxPool2d(2, 2),                              # features.5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # features.6
            nn.ReLU(),                                       # features.7
            nn.MaxPool2d(2, 2),                              # features.8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                    # classifier.0
            nn.Linear(128 * 28 * 28, 512),                  # classifier.1 → 100352→512
            nn.ReLU(),                                       # classifier.2
            nn.Dropout(0.5),                                 # classifier.3
            nn.Linear(512, 256),                             # classifier.4
            nn.ReLU(),                                       # classifier.5
            nn.Linear(256, num_classes),                     # classifier.6
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model state holders
# ─────────────────────────────────────────────────────────────────────────────
class ImageModelWrapper:
    model: nn.Module = None
    class_names: list = []
    num_classes: int = 38


class TextModelWrapper:
    model: DistilBertForSequenceClassification = None
    tokenizer: DistilBertTokenizerFast = None
    id2label: dict = {}
    max_len: int = 128


image_state = ImageModelWrapper()
text_state  = TextModelWrapper()


# ─────────────────────────────────────────────────────────────────────────────
# Startup — load both models once when server starts
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():

    # ── Image model ──────────────────────────────────────────────────────────
    print(f"Loading image model from {IMAGE_MODEL_PATH} ...")
    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Image model not found: {IMAGE_MODEL_PATH}")

    state_dict = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)

    # It's a plain OrderedDict of weights — load directly
    img_model = SimpleCNN(num_classes=38)
    img_model.load_state_dict(state_dict, strict=False)
    img_model.eval()
    img_model.to(DEVICE)

    image_state.model       = img_model
    image_state.class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']  # no class names saved — returns index numbers
    image_state.num_classes = 38
    print(f"Image model ready — 38 classes on {DEVICE}")

    # ── Text model ───────────────────────────────────────────────────────────
    print(f"Loading text model from {TEXT_MODEL_PATH} ...")
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Text model not found: {TEXT_MODEL_PATH}")

    txt_model = DistilBertForSequenceClassification.from_pretrained(str(TEXT_MODEL_PATH))
    txt_model.eval()
    txt_model.to(DEVICE)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TEXT_MODEL_PATH))

    meta_path = TEXT_MODEL_PATH / "inference_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        max_len  = meta.get("max_len", 128)
    else:
        id2label = {i: str(i) for i in range(txt_model.config.num_labels)}
        max_len  = 128

    text_state.model     = txt_model
    text_state.tokenizer = tokenizer
    text_state.id2label  = id2label
    text_state.max_len   = max_len
    print(f"Text model ready — {len(id2label)} classes on {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    disease: str
    confidence: float
    confidence_pct: str
    top3: list[dict]
    model_used: str
    inference_time_ms: float


class TextRequest(BaseModel):
    text: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "image_model_loaded": image_state.model is not None,
        "text_model_loaded":  text_state.model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict/image", response_model=PredictionResult, tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a plant leaf image → returns predicted disease + confidence.
    Accepts JPG or PNG. Returns top prediction + top 3 results.
    """
    if image_state.model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded.")

    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Use JPG or PNG."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = image_transforms(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = image_state.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top3_indices = probs.topk(3).indices.cpu().tolist()
    top3_probs   = probs.topk(3).values.cpu().tolist()

    top3 = [
        {
            "disease":        image_state.class_names[i] if image_state.class_names else f"class_{i}",
            "confidence_pct": f"{top3_probs[rank]*100:.2f}%",
        }
        for rank, i in enumerate(top3_indices)
    ]

    best_idx  = top3_indices[0]
    best_prob = top3_probs[0]
    best_name = image_state.class_names[best_idx] if image_state.class_names else f"class_{best_idx}"

    return PredictionResult(
        disease           = best_name,
        confidence        = round(best_prob, 4),
        confidence_pct    = f"{best_prob*100:.2f}%",
        top3              = top3,
        model_used        = "SimpleCNN",
        inference_time_ms = round(elapsed_ms, 2),
    )


@app.post("/predict/text", response_model=PredictionResult, tags=["Prediction"])
def predict_text(request: TextRequest):
    """
    Send a symptom description → returns predicted disease + confidence.
    Example: {"text": "Leaves show yellow spots with brown edges and curling tips"}
    """
    if text_state.model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded.")

    text = request.text.strip()
    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Text too short. Describe the symptoms.")

    enc = text_state.tokenizer(
        text,
        max_length=text_state.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = text_state.model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).logits
        probs = torch.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top3_indices = probs.topk(3).indices.cpu().tolist()
    top3_probs   = probs.topk(3).values.cpu().tolist()

    top3 = [
        {
            "disease":        text_state.id2label.get(i, str(i)),
            "confidence_pct": f"{top3_probs[rank]*100:.2f}%",
        }
        for rank, i in enumerate(top3_indices)
    ]

    best_idx  = top3_indices[0]
    best_prob = top3_probs[0]
    best_name = text_state.id2label.get(best_idx, str(best_idx))

    return PredictionResult(
        disease           = best_name,
        confidence        = round(best_prob, 4),
        confidence_pct    = f"{best_prob*100:.2f}%",
        top3              = top3,
        model_used        = "DistilBERT",
        inference_time_ms = round(elapsed_ms, 2),
    )