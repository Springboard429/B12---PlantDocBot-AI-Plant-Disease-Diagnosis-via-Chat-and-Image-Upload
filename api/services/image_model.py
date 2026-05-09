import torch
import torch.nn as nn
from torchvision import transforms

device = torch.device("cpu")

model = None
class_names = None

# CNN (same as your training)
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
            nn.Dropout(0.3),   # ✅ THIS WAS MISSING
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_image_model(path):
    global model, class_names

    checkpoint = torch.load(path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model_instance = CNN(num_classes)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()

    model = model_instance
    print("✅ Image model loaded")


def predict_image(image):
    global model, class_names

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return {
    "disease": class_names[pred.item()],
    "confidence": round(float(conf.item() * 100), 2)
}
    