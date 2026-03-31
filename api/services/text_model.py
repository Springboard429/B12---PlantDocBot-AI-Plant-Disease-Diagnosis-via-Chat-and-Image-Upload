import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# =========================
# GLOBAL VARIABLES
# =========================
model = None
tokenizer = None
class_names = None

SUPPORTED_PLANTS = ["tomato", "potato", "pepper"]

# =========================
# LOAD MODEL
# =========================
def load_text_model(model_path):
    global model, tokenizer, class_names

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    with open(f"{model_path}/classes.json") as f:
        class_names = json.load(f)

    model.eval()

    print("✅ Text model loaded")

# =========================
# HELPER FUNCTION
# =========================
def detect_plant(text):
    text = text.lower()
    for plant in SUPPORTED_PLANTS:
        if plant in text:
            return plant
    return None

# =========================
# PREDICTION FUNCTION
# =========================
def predict_text(user_input: str):
    global model, tokenizer, class_names

    if model is None:
        raise Exception("Model not loaded")

    plant = detect_plant(user_input)

    # ✅ Prevent frontend crash
    if plant is None:
        return {
            "disease": "Unknown",
            "confidence": 0.0
        }

    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    top_probs, top_indices = torch.topk(probs, 3)

    predictions = []
    for i in range(3):
        disease = class_names[top_indices[0][i]]
        confidence = top_probs[0][i].item()
        predictions.append({
            "disease": disease,
            "confidence": round(confidence * 100, 2)
        })

    # ✅ Return ONLY what frontend needs
    best = predictions[0]

    return {
        "disease": best["disease"],
        "confidence": best["confidence"]
    }
