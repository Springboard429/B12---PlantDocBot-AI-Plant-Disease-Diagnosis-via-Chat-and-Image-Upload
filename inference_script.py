
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys

# --- Constants derived from dataset ---
NUM_CLASSES = 38
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# 1. Define the CNN model class
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened features size for the first fully connected layer
        flattened_size = 256 * 14 * 14

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Define the image transformation pipeline for inference
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Implement a function to predict a single image
def predict_image(image_path, model, device, transform, class_names):
    """
    Loads an image, preprocesses it, and makes a prediction using the trained model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx.item()].item()

    return predicted_class, confidence


if __name__ == '__main__':
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = CNN(num_classes=NUM_CLASSES)

    # 4. Load the saved model state dictionary
    model_path = 'cnn_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in the current directory.")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    print(f"Model loaded from {model_path} and set to evaluation mode on {device}.")

    # 5. Add example usage
    print("
--- Example Usage ---")
    print("To use, place a sample image in the same directory as this script")
    print("or provide a full path to an image.")
    sample_image_path = input("Enter path to a sample image (e.g., 'sample.jpg'): ")

    if not sample_image_path:
        print("No image path provided. Exiting example.")
    elif not os.path.exists(sample_image_path):
        print(f"Error: Sample image '{sample_image_path}' not found.")
    else:
        try:
            print(f"Making prediction for: {sample_image_path}")
            predicted_class, confidence = predict_image(sample_image_path, model, device, inference_transforms, CLASS_NAMES)
            print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
