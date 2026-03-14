from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import os

# ═══════════════════════════════════════
# Custom Model Definition (must match train.py)
# ═══════════════════════════════════════

class SentimentClassifier(nn.Module):
    """Same architecture used during training."""
    def __init__(self, pretrained_model_name="distilbert-base-uncased", dropout=0.3):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.distilbert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# ═══════════════════════════════════════
# Load Model & Tokenizer
# ═══════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), "sentiment_model")
DEVICE = torch.device("cpu")  # Use CPU for serving (stable & works everywhere)

def load_custom_model():
    """Load the custom-trained model from disk."""
    model_path = os.path.join(MODEL_DIR, "model.pt")

    if os.path.exists(model_path):
        print(f"✓ Loading custom-trained model from {MODEL_DIR}")
        model = SentimentClassifier()
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        print(f"✓ Custom model loaded (best F1: {checkpoint.get('best_f1', 'N/A')})")
        return model, tokenizer, True
    else:
        print(f"⚠ No custom model found at {MODEL_DIR}. Using HuggingFace pre-trained model as fallback.")
        print(f"  Run 'python train.py' to train your own model!")
        from transformers import pipeline
        return None, None, False

model, tokenizer, is_custom = load_custom_model()

# Fallback pipeline if custom model doesn't exist yet
if not is_custom:
    from transformers import pipeline
    fallback_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ═══════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment: str
    score: float
    model_type: str  # "custom" or "pretrained"

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Sentiment Analysis API is running.",
        "model": "custom-trained" if is_custom else "pretrained-fallback",
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    if is_custom:
        # ─── Use our custom-trained model ───
        encoding = tokenizer(
            request.text,
            add_special_tokens=True,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        sentiment = "Positive" if predicted_class == 1 else "Negative"
        return AnalyzeResponse(sentiment=sentiment, score=confidence, model_type="custom")
    else:
        # ─── Fallback to pre-trained pipeline ───
        result = fallback_pipeline(request.text)[0]
        label = result["label"]
        sentiment = "Positive" if label == "POSITIVE" else "Negative"
        return AnalyzeResponse(sentiment=sentiment, score=result["score"], model_type="pretrained")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
