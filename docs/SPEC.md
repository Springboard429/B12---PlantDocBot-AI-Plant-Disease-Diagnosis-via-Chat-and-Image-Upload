# PlantDoc API — Specification

## 1. Overview

**Project**: PlantDoc API
**Type**: FastAPI REST backend
**Summary**: A plant disease detection API that serves two trained ML models — an image classifier (SimpleCNN) and a text classifier (DistilBERT) — to diagnose plant diseases from either leaf images or text descriptions.
**Target users**: Frontend applications, mobile apps, or external services that need plant disease predictions.

---

## 2. Functionality

### Core Features

#### 2.1 Image Disease Classification
- **Endpoint**: `POST /predict/image`
- **Input**: Multipart form-upload (`file` field), accepts JPEG/PNG images
- **Behavior**: Loads the image, runs it through the SimpleCNN model, returns the predicted disease class and confidence score
- **Output**:
  ```json
  {
    "disease": "Tomato Late blight",
    "confidence": 94.2,
    "model": "simplecnn"
  }
  ```

#### 2.2 Text Disease Classification
- **Endpoint**: `POST /predict/text`
- **Input**: JSON body `{"text": "A tomato leaf showing dark brown lesions..."}`
- **Behavior**: Tokenizes the text, runs it through the DistilBERT model, returns the predicted disease class and confidence score
- **Output**:
  ```json
  {
    "disease": "Tomato Late blight",
    "confidence": 87.5,
    "model": "distilbert"
  }
  ```

#### 2.3 Combined Prediction
- **Endpoint**: `POST /predict/combined`
- **Input**: Multipart form-upload with optional text field (`{"text": "..."}` alongside image file)
- **Behavior**: Runs both models and returns both predictions; if only one input is provided, runs only that model
- **Output**:
  ```json
  {
    "image_prediction": {
      "disease": "Tomato Late blight",
      "confidence": 94.2
    },
    "text_prediction": {
      "disease": "Tomato Late blight",
      "confidence": 87.5
    }
  }
  ```

#### 2.4 Health Check
- **Endpoint**: `GET /health`
- **Output**: `{"status": "ok", "models_loaded": true}`

#### 2.5 List Available Disease Classes
- **Endpoint**: `GET /classes`
- **Output**:
  ```json
  {
    "classes": [
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
      "Tomato mosaic virus"
    ]
  }
  ```

### Models

| Model | File | Type | Input |
|-------|------|------|-------|
| SimpleCNN | `assets/models/simplecnn/best_simplecnn_plant_disease.pth` | Image classifier | RGB image (224×224) |
| DistilBERT | `assets/models/distilbert_plantdisease_model/` | Text classifier | Disease description text |

Both models share the same 15-class label set.

### Error Handling
- Invalid image format → 400 with `{"detail": "Unsupported image format. Use JPEG or PNG."}`
- Missing file → 400 with `{"detail": "No image file provided."}`
- Text too long (>512 tokens) → 400 with `{"detail": "Text exceeds maximum length."}`
- Model not loaded → 503 with `{"detail": "Model not ready. Try again later."}`

---

## 3. Technical Approach

### Stack
- **Framework**: FastAPI (Python 3.8+)
- **ML Runtime**: PyTorch 2.0+
- **Image Processing**: PIL/Pillow, torchvision transforms
- **Text Processing**: HuggingFace Transformers (DistilBERT tokenizer + model)
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic for request/response schemas

### Project Structure
```
plantdoc/
├── assets/
│   └── models/
│       ├── simplecnn/
│       │   └── best_simplecnn_plant_disease.pth
│       └── distilbert_plantdisease_model/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, routes, startup
│   ├── models.py         # Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py     # /predict/* routes
│   └── services/
│       ├── __init__.py
│       ├── image_model.py   # SimpleCNN loading & inference
│       └── text_model.py   # DistilBERT loading & inference
├── docs/
│   └── SPEC.md
└── requirements.txt
```

### Startup Behavior
- On application startup (`@app.on_event("startup")`), load both models into memory once
- Keep models loaded for the lifetime of the process (no per-request loading)
- If GPU is available, move models to CUDA; fall back to CPU

### Performance Considerations
- Image predictions are stateless — no session or cache needed
- Both models can run on CPU; GPU is optional
- Max image size: 10 MB

---

## 4. Acceptance Criteria

1. `POST /predict/image` accepts a JPEG/PNG upload and returns a disease prediction with confidence
2. `POST /predict/text` accepts a JSON body with text and returns a disease prediction with confidence
3. `POST /predict/combined` accepts an image + optional text and returns predictions from both models
4. `GET /health` returns a 200 response when the server is running
5. `GET /classes` returns all 15 disease class names
6. Invalid inputs return appropriate 4xx errors with error messages
7. Models are loaded once at startup, not on every request
8. The API can be started with `uvicorn api.main:app`
