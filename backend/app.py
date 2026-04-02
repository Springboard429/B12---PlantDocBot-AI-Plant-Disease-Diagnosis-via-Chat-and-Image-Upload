from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from PIL import Image
import io

from transformers import AutoModelForSequenceClassification, DistilBertTokenizer

import webbrowser
import threading
import os

# open browser automatically
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/docs")

# initialize fastapi
app = FastAPI()

@app.on_event("startup")
def startup_event():
    def open_once():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000/docs")

    threading.Thread(target=open_once).start()

# helper function to clean disease names
def format_disease_name(name: str) -> str:
    # replace underscores with spaces
    name = name.replace("_", " ")

    # remove multiple spaces
    name = " ".join(name.split())

    # fix common patterns
    name = name.replace("  ", " ")

    # capitalize nicely
    name = name.title()

    return name

# load text model
TEXT_MODEL_PATH = "backend/models/plant_disease_text_model"

text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

with open("backend/models/label_encoder.pkl", "rb") as f:
    text_encoder = pickle.load(f)

# define cnn model (same as training)
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
image_model = CNN(num_classes=38)

image_model.load_state_dict(
    torch.load("backend/models/best_cnn_model.pth", map_location="cpu")
)

image_model.eval()

# image transforms
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load image class names safely
def load_image_classes():
    # try to load from pickle if exists
    path = "backend/models/image_class_names.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # fallback: create dummy class names
    # this ensures api does not crash
    return [f"class_{i}" for i in range(38)]

image_classes = load_image_classes()

# request schema
class TextInput(BaseModel):
    text: str

# text prediction endpoint
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

    # get raw label from encoder
    raw_label = text_encoder.inverse_transform([predicted_class])[0]

    # format nicely
    clean_label = format_disease_name(raw_label)

    # calculate confidence
    probs = F.softmax(logits, dim=1)
    confidence = probs[0][predicted_class].item()

    return {
        "input_text": data.text,
        "prediction": clean_label,
        "confidence": round(confidence * 100, 2)
    }

# image prediction endpoint
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = image_model(image_tensor)

    # add this debug
    probs = torch.softmax(outputs, dim=1)
    print("probabilities:", probs)

    predicted_class = torch.argmax(probs, dim=1).item()

    predicted_class = torch.argmax(outputs, dim=1).item()

    # get class name using index
    class_name = image_classes[predicted_class]

    # format nicely
    clean_label = format_disease_name(class_name)

    # calculate confidence
    probs = F.softmax(outputs, dim=1)
    confidence = probs[0][predicted_class].item()

    return {
        "filename": file.filename,
        "prediction": clean_label,
        "confidence": round(confidence * 100, 2)
    }

# allow frontend connection
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
