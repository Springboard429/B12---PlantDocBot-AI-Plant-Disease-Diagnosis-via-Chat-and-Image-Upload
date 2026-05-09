import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# CNN Model (same as training)
# ------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ------------------------
# Load Model
# ------------------------
MODEL_PATH = "best_model_full.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = CNN(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class_names = checkpoint["class_names"]

print("✅ Model loaded successfully")

# ------------------------
# Transform (IMPORTANT)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ------------------------
# Prediction Function
# ------------------------
def predict(image_path):
    if not os.path.exists(image_path):
        print("❌ Image not found")
        return

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = confidence.item() * 100

    print("Prediction:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

# ------------------------
# Run from terminal
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide image path")
    else:
        predict(sys.argv[1])