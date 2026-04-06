from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib

# Model folder
MODEL_PATH = "models/distilbert_model"

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

# Load trained model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")

model.eval()


def predict_disease(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    disease = label_encoder.inverse_transform([predicted_class])[0]

    return disease


if __name__ == "__main__":

    user_input = input("Enter plant leaf description: ")

    prediction = predict_disease(user_input)

    print("Predicted Disease:", prediction)