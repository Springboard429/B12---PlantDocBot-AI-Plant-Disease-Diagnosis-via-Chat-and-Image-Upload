import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "models/text_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


# -----------------------------
# Load label encoder
# -----------------------------
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

# convert logits → probabilities
    probs = torch.softmax(outputs.logits, dim=1)

# get predicted class
    predicted_class = torch.argmax(probs, dim=1).item()

# get confidence (probability of predicted class)
    confidence = probs[0][predicted_class].item()

    disease = encoder.inverse_transform([predicted_class])[0]

    return {
    "disease": disease,
    "confidence": confidence
}


# -----------------------------
# Test Example
# -----------------------------
if __name__ == "__main__":

    text = input("Enter plant disease description: ")

    prediction = predict_disease(text)

    print("Predicted Disease:", prediction)