# import required libraries
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
import io
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import webbrowser
import threading
from typing import Optional


# initialize fastapi app
app = FastAPI()


# open browser automatically on startup
@app.on_event("startup")
def startup_event():
    def open_once():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000/docs")

    threading.Thread(target=open_once).start()


# helper function to format disease names
def format_disease_name(name: str) -> str:
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name.title()


# load text model and tokenizer
TEXT_MODEL_PATH = "backend/models/plant_disease_text_model"

text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)

text_model.eval()

with open("backend/models/label_encoder.pkl", "rb") as f:
    text_encoder = pickle.load(f)


# define cnn model architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# load image model
IMAGE_MODEL_PATH = "backend/models/best_cnn_model.pth"

checkpoint = torch.load(IMAGE_MODEL_PATH, map_location="cpu")

class_names = checkpoint["class_names"]
num_classes = len(class_names)

image_model = CNN(num_classes=num_classes)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model.eval()


# define image transforms
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# define request schema
class TextInput(BaseModel):
    text: str = Field(..., max_length=512)


# define response schemas
class TextPredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "distilbert"


class ImagePredictionResponse(BaseModel):
    disease: str
    confidence: float
    model: str = "simplecnn"


class CombinedPredictionResponse(BaseModel):
    image_prediction: Optional[ImagePredictionResponse] = None
    text_prediction: Optional[TextPredictionResponse] = None


# function to check confidence and gap
def is_valid_prediction(probs):
    top2_probs, _ = torch.topk(probs, 2)
    top1 = top2_probs[0][0].item()
    top2 = top2_probs[0][1].item()
    gap = top1 - top2
    return top1, gap


# text prediction endpoint
@app.post("/predict-text", response_model=TextPredictionResponse)
def predict_text(data: TextInput):

    # keyword validation
    keywords = ["leaf", "plant", "spot", "disease", "yellow", "brown", "green"]
    if not any(word in data.text.lower() for word in keywords):
        return TextPredictionResponse(
            disease="input not related to plant disease",
            confidence=0
        )

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = text_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

    confidence, gap = is_valid_prediction(probs)

    if confidence < 0.6 or gap < 0.15:
        return TextPredictionResponse(
            disease="invalid or unclear description",
            confidence=round(confidence * 100, 2)
        )

    raw_label = text_encoder.inverse_transform([predicted_class])[0]
    clean_label = format_disease_name(raw_label)

    return TextPredictionResponse(
        disease=clean_label,
        confidence=round(confidence * 100, 2)
    )


# image prediction endpoint
@app.post("/predict-image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image_tensor)

    probs = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

    confidence, gap = is_valid_prediction(probs)

    if confidence < 0.6 or gap < 0.15:
        return ImagePredictionResponse(
            disease="invalid image or not a leaf",
            confidence=round(confidence * 100, 2)
        )

    class_name = class_names[predicted_class]
    clean_label = format_disease_name(class_name)

    return ImagePredictionResponse(
        disease=clean_label,
        confidence=round(confidence * 100, 2)
    )


# combined prediction endpoint
@app.post("/predict-combined", response_model=CombinedPredictionResponse)
async def predict_combined(
    text: Optional[str] = Form(None),
    file: UploadFile = File(None)
):

    image_result = None
    text_result = None

    # text prediction
    if text:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = text_model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

            confidence, gap = is_valid_prediction(probs)

            if confidence >= 0.6 and gap >= 0.15:
                raw_label = text_encoder.inverse_transform([predicted_class])[0]
                clean_label = format_disease_name(raw_label)

                text_result = TextPredictionResponse(
                    disease=clean_label,
                    confidence=round(confidence * 100, 2)
                )
            else:
                text_result = TextPredictionResponse(
                    disease="invalid or unclear description",
                    confidence=round(confidence * 100, 2)
                )

        except:
            text_result = TextPredictionResponse(
                disease="invalid or unclear description",
                confidence=0
            )

    # image prediction (unchanged)
    if file:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = image_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = image_model(image_tensor)

        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

        confidence, gap = is_valid_prediction(probs)

        if confidence >= 0.6 and gap >= 0.15:
            class_name = class_names[predicted_class]
            clean_label = format_disease_name(class_name)

            image_result = ImagePredictionResponse(
                disease=clean_label,
                confidence=round(confidence * 100, 2)
            )
        else:
            image_result = ImagePredictionResponse(
                disease="invalid image or not a leaf",
                confidence=round(confidence * 100, 2)
            )

    return CombinedPredictionResponse(
        image_prediction=image_result,
        text_prediction=text_result
    )


# enable cors
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)