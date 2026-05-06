
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# 1. Define the SimpleCNN Model (must be identical to the training model)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
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

# 2. Configuration for loading the model
MODEL_PATH = "/content/drive/MyDrive/Infosys/best_simplecnn_plant_disease.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the model is saved to Google Drive and the path is correct.")
    model = None # Set model to None if not found
    class_names = []
else:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    num_classes = checkpoint['num_classes']
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    class_names = checkpoint['class_names']
    print("Model loaded successfully.")

# 3. Define the image transformation pipeline
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Prediction function
def predict_image(image_path, actual_class_name=None):
    if model is None:
        return "Model not loaded", 0.0
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}", 0.0

    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    # Plotting the image is typically not done in a standalone inference script,
    # but for completeness, I'll keep the plotting code if matplotlib is imported.
    # You can comment out or remove this part for a pure inference script.
    try:
        import matplotlib.pyplot as plt
        plt.imshow(image)
        title_text = f"Pred: {predicted_class_name}
Confidence: {confidence_score:.2f}%"
        if actual_class_name:
            title_text = f"True: {actual_class_name}
" + title_text
        plt.title(title_text)
        plt.axis("off")
        plt.show()
    except ImportError:
        pass # matplotlib not available, skip plotting

    return predicted_class_name, confidence_score

# Example usage (you would replace this with actual image paths when running standalone)
if __name__ == '__main__':
    print("
--- Standalone Example Prediction --")
    # Placeholder for a real image path
    # For a real inference, uncomment the line below and provide a valid path:
    # example_image_path = "path/to/your/new/image.jpg"

    # This part would typically be removed or modified to accept command-line arguments
    # or hardcoded paths for specific new images.
    print("To run an example prediction, please modify the `if __name__ == '____':` block")
    print("and provide a path to an image for prediction.")

    # For instance, if you have a test image at 'test_image.jpg' in the same directory:
    # test_image_path = 'test_image.jpg'
    # if os.path.exists(test_image_path):
    #     predicted_class, confidence = predict_image(test_image_path)
    #     print(f"Predicted Class for {test_image_path}: {predicted_class}")
    #     print(f"Confidence: {confidence:.2f}%")
    # else:
    #     print(f"Please place a test image at {test_image_path} to try the example.")
