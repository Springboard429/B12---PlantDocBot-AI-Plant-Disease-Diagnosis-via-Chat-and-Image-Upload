import torch
import torch.nn.functional as F
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================
# GLOBALS
# =========================

model = None
tokenizer = None
class_names = None


# =========================
# LOAD MODEL
# =========================

def load_text_model(path):

    global model, tokenizer, class_names

    print("Loading DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(path)

    with open(f"{path}/classes.json", "r") as f:
        class_names = json.load(f)

    model.eval()

    print("✅ Text model loaded successfully")


# =========================
# PREDICT FUNCTION
# =========================

def predict_text(text):

    global model, tokenizer, class_names

    # -------------------------
    # CLEAN INPUT
    # -------------------------

    text = text.lower().strip()

    # add plant context automatically
    if "tomato" in text and "leaf" not in text:
        text = f"tomato leaf {text}"

    elif "potato" in text and "leaf" not in text:
        text = f"potato leaf {text}"

    elif "pepper" in text and "leaf" not in text:
        text = f"pepper leaf {text}"

    elif "leaf" not in text:
        text = f"plant leaf {text}"

    # -------------------------
    # TOKENIZE
    # -------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    # -------------------------
    # MODEL
    # -------------------------

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # temperature scaling
    probs = F.softmax(logits / 1.5, dim=-1)

    probs = probs.squeeze(0)

    # -------------------------
    # TOP 3
    # -------------------------

    top_probs, top_indices = torch.topk(probs, 3)

    top_3 = []

    for i in range(3):

        idx = top_indices[i].item()

        disease = class_names[idx]

        confidence = round(
            float(top_probs[i].item() * 100),
            2
        )

        top_3.append({
            "disease": disease,
            "confidence": confidence
        })

    # -------------------------
    # BEST RESULT
    # -------------------------

    best = top_3[0]

    return {
        "disease": best["disease"],
        "confidence": best["confidence"],
        "top_3": top_3
    }


print("🔥 NEW TEXT MODEL LOADED")