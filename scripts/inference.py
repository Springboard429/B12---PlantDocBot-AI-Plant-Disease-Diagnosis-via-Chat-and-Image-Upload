import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
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
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

checkpoint = torch.load(
    PROJECT_ROOT / 'assets/models/simplecnn/best_simplecnn_plant_disease.pth',
    weights_only=True,
)
class_names = checkpoint['class_names']
num_classes = checkpoint['num_classes']

model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_prob, pred_idx = torch.max(probabilities, 1)
    
    pred_class = class_names[pred_idx.item()]
    confidence = pred_prob.item() * 100
    
    plt.imshow(image)
    plt.title(f'Prediction: {pred_class}\nConfidence: {confidence:.1f}%')
    plt.axis('off')
    plt.show()
    
    return pred_class, confidence

image_path = PROJECT_ROOT / 'dataset/archive/PlantVillage/val/Tomato___Bacterial_spot/0a6d40e4-75d6-4659-8bc1-22f47cdb2ca8___GCREC_Bact.Sp 6247.JPG'
predicted_class, confidence = predict_image(image_path)
print(f'Predicted: {predicted_class} ({confidence:.1f}%)')