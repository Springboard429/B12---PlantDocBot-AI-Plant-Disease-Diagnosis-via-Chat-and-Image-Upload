import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TextModel:
    def __init__(self):
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 128

    def load(self) -> None:
        model_path = PROJECT_ROOT / "assets/models/distilbert_plantdisease_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text model not loaded")
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            pred_prob, pred_idx = torch.max(probabilities, dim=1)
        pred_class = self.model.config.id2label[str(pred_idx.item())]
        confidence = pred_prob.item() * 100
        return pred_class, confidence


_text_model: TextModel | None = None


def get_text_model() -> TextModel:
    global _text_model
    if _text_model is None:
        _text_model = TextModel()
        _text_model.load()
    return _text_model
