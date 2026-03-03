import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# CNN Model Definition
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Load Class Names from Dataset
# (Ensures correct order)
# -----------------------------
dataset_path = "data/raw/plantvillage dataset/color"

class_names = sorted(os.listdir(dataset_path))


# -----------------------------
# Load Best Model
# -----------------------------
model_path = "models/cnn_best_model.pth"

if not os.path.exists(model_path):
    print("Model file not found!")
    exit()

model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model Loaded Successfully!")


# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image_path):

    if not os.path.exists(image_path):
        print("Image not found!")
        return

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_disease = class_names[predicted.item()]

    print("Predicted Disease:", predicted_disease)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # 🔽 Paste your image path here 🔽
    image_path = "data/raw/plantvillage dataset/color/Peach___healthy/1a07ce54-f4fd-41cf-b088-144f6bf71859___Rutg._HL 3543.JPG"

    predict_image(image_path)