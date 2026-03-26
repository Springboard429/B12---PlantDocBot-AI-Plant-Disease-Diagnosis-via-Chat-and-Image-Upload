import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from transformers import pipeline

# --- 1. SETTINGS & DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL_PATH = "best_model.pth"  # Your .pth file
TEXT_MODEL_PATH = "fine_tuned_model"            # Or your custom spaCy model folder

# The standard 38 PlantVillage classes (alphabetical order)
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# The 15 classes your Hugging Face text classification model was trained on
TEXT_CLASSES = [
    "Pepper bell Bacterial spot", "Pepper bell healthy", "Potato Early blight",
    "Potato Late blight", "Potato healthy", "Tomato Bacterial spot",
    "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
    "Tomato Septoria leaf spot", "Tomato Spider mites Two spotted spider mite",
    "Tomato Target Spot", "Tomato YellowLeaf Curl Virus", "Tomato healthy",
    "Tomato mosaic virus"
]

# --- 2. LOAD HUGGING FACE TEXT CLASSIFICATION MODEL ---
try:
    text_classifier = pipeline("text-classification", model=TEXT_MODEL_PATH, tokenizer=TEXT_MODEL_PATH)
    print(f"Successfully loaded text model from {TEXT_MODEL_PATH}")
except Exception as e:
    print(f"Error loading Hugging Face text model: {e}")
    text_classifier = None

def get_text_predictions(text: str):
    if text_classifier is None:
        return {"label": "Model Not Loaded", "score": 0.0}
        
    try:
        # The pipeline outputs a list like: [{'label': 'LABEL_9', 'score': 0.95}]
        result = text_classifier(text)[0]
        label_str = result["label"]
        
        # Convert "LABEL_X" back to the actual class name
        if label_str.startswith("LABEL_"):
            try:
                class_idx = int(label_str.replace("LABEL_", ""))
                if 0 <= class_idx < len(TEXT_CLASSES):
                    result["label"] = TEXT_CLASSES[class_idx]
            except Exception:
                pass
                
        return result
    except Exception as e:
        print(f"Inference error: {e}")
        return {"label": "Error", "score": 0.0}

# --- 3. PYTORCH MODEL ARCHITECTURE ---
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(3 * 224 * 224, 256), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

def load_leaf_model():
    # Load the checkpoint dictionary
    try:
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
        best_model_name = checkpoint.get('model_name', 'CNN')
        num_classes = checkpoint.get('num_classes', 38)
        
        # Instantiate the correct model architecture
        if best_model_name == 'CNN':
            model = SimpleCNN(num_classes=num_classes)
        else:
            model = MLP(num_classes=num_classes)
            
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded {best_model_name} model weights from {IMAGE_MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load model weights from {IMAGE_MODEL_PATH}. Error: {e}")
        model = SimpleCNN(num_classes=38) # Fallback to empty model
        
    model.to(DEVICE)
    model.eval()
    return model

leaf_model = load_leaf_model()

def get_image_prediction(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = leaf_model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
        
    class_idx = predicted.item()
    class_name = PLANT_CLASSES[class_idx] if 0 <= class_idx < len(PLANT_CLASSES) else "Unknown"
        
    return class_name, confidence, class_idx