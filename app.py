from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import torch
import torch.nn as nn
import pickle
from PIL import Image
import io

from transformers import AutoModelForSequenceClassification, DistilBertTokenizer

import webbrowser
import threading
import os

# ---------------------------
# Open browser
# ---------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/docs")

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    def open_once():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000/docs")

    threading.Thread(target=open_once).start()

# ---------------------------
# LOAD TEXT MODEL
# ---------------------------
TEXT_MODEL_PATH = "models/plant_disease_text_model"

text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

with open("models/label_encoder.pkl", "rb") as f:
    text_encoder = pickle.load(f)

# ---------------------------
# DEFINE CNN MODEL
# ---------------------------
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

# ---------------------------
# LOAD IMAGE MODEL
# ---------------------------
image_model = CNN(num_classes=38)

image_model.load_state_dict(
    torch.load("models/best_cnn_model.pth", map_location="cpu")
)

image_model.eval()

# ---------------------------
# IMAGE TRANSFORMS
# ---------------------------
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# REQUEST SCHEMA
# ---------------------------
class TextInput(BaseModel):
    text: str

# ---------------------------
# TEXT PREDICTION
# ---------------------------
@app.post("/predict-text")
def predict_text(data: TextInput):

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = text_model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    disease = text_encoder.inverse_transform([predicted_class])[0]

    return {
        "input_text": data.text,
        "prediction": disease
    }

# ---------------------------
# IMAGE PREDICTION
# ---------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image_tensor)

    predicted_class = torch.argmax(outputs, dim=1).item()

    return {
        "filename": file.filename,
        "predicted_class_index": predicted_class
    }