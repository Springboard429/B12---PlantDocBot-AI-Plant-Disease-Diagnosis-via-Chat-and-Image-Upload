
import torch
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load saved model
model_path = "/content/drive/MyDrive/Infosys/distilbert_plant_disease_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load label encoder
encoder = joblib.load("/content/drive/MyDrive/Infosys/label_encoder.pkl")

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

    prediction = torch.argmax(outputs.logits, dim=1).item()

    disease = encoder.inverse_transform([prediction])[0]

    return disease


if __name__ == "__main__":

    text = input("Enter plant disease description: ")

    result = predict_disease(text)

    print("Predicted Disease:", result)
