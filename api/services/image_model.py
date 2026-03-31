import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# MODEL DEFINITION
# =========================
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
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

# =========================
# GLOBAL VARIABLES
# =========================
model = None
class_names = None

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD MODEL FUNCTION
# =========================
def load_image_model(model_path: str):
    global model, class_names

    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model_instance = CNN(num_classes)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.to(device)
    model_instance.eval()

    model = model_instance

    print("✅ Image model loaded successfully")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image: Image.Image):
    global model, class_names

    if model is None:
        raise Exception("Model not loaded")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    predicted_class = class_names[pred_idx.item()]
    confidence = confidence.item() * 100

    return {
        "disease": predicted_class,      # ✅ FIXED
        "confidence": round(confidence, 2)
    }
