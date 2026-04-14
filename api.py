from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Plant Disease Detection API")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- CNN MODEL PATH --------------------
cnn_model_path = "./best_simplecnn_plant_disease.pth"

# -------------------- LOAD CNN MODEL + CLASSES --------------------
checkpoint = torch.load(cnn_model_path, map_location=device)

class_names_cnn = checkpoint['class_names']
num_classes = len(class_names_cnn)

print(f"✅ Loaded {num_classes} CNN classes.")

# -------------------- CNN MODEL --------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -------------------- LOAD CNN --------------------
cnn_model = CNN(num_classes=num_classes)
cnn_model.load_state_dict(checkpoint['model_state_dict'])
cnn_model.to(device)
cnn_model.eval()

print("✅ CNN model loaded successfully.")

# -------------------- IMAGE TRANSFORM --------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- TEXT MODEL --------------------
text_model_dir = "./distilbert_plant_model"

try:
    tokenizer_text = AutoTokenizer.from_pretrained(text_model_dir)
    model_text = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
    model_text.to(device)
    model_text.eval()
    print("✅ Text model loaded successfully.")
except Exception as e:
    print(f"❌ Text model error: {e}")

# -------------------- IMAGE ENDPOINT --------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(image_tensor)
            pred_id = torch.argmax(output, dim=1).item()

            if pred_id < len(class_names_cnn):
                pred_class = class_names_cnn[pred_id]
            else:
                pred_class = f"Unknown Disease (Index {pred_id})"

        return {
            "filename": file.filename,
            "predicted_disease": pred_class
        }

    except Exception as e:
        print(f"Image Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- TEXT ENDPOINT (FIXED) --------------------
@app.post("/predict-text")
async def predict_text(description: str = Form(...)):
    try:
        inputs = tokenizer_text(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model_text(**inputs)
            pred_id = outputs.logits.argmax(dim=-1).item()

            # ✅ FIXED LABEL MAPPING
            labels = ["Healthy", "Unhealthy"]  # 🔥 adjust if needed
            pred_class = labels[pred_id]

        return {
            "description": description,
            "predicted_disease": pred_class
        }

    except Exception as e:
        print(f"Text Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- ROOT --------------------
@app.get("/")
async def root():
    return {"message": "API is Running"}