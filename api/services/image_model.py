import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageModel:
    def __init__(self):
        self.model: SimpleCNN | None = None
        self.class_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self) -> None:
        checkpoint_path = PROJECT_ROOT / "assets/models/simplecnn/best_simplecnn_plant_disease.pth"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]
        self.model = SimpleCNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Image model not loaded")
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_prob, pred_idx = torch.max(probabilities, 1)
        pred_class = self.class_names[pred_idx.item()]
        confidence = pred_prob.item() * 100
        return pred_class, confidence


_image_model: ImageModel | None = None


def get_image_model() -> ImageModel:
    global _image_model
    if _image_model is None:
        _image_model = ImageModel()
        _image_model.load()
    return _image_model
