"""
Sentiment Agent: CardiffNLP twitter-roberta-base-sentiment-latest.

Outputs a continuous score in [-1.0, 1.0] (negative = -1, positive = +1)
plus a discrete label {negative, neutral, positive}.

Score is computed as p(positive) - p(negative) so the planner can bucket it
the same way the dataset's sentiment_score was bucketed.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings
from app.core.schemas import Sentiment


@lru_cache(maxsize=1)
def _load():
    name = settings.sentiment_model
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if (settings.device == "cuda" and torch.cuda.is_available()) else "cpu"
    mdl = mdl.to(device).eval()
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    return tok, mdl, device, id2label


def analyze(text: str) -> Sentiment:
    text = (text or "").strip()
    if not text:
        return Sentiment(score=0.0, label="neutral")
    tok, mdl, device, id2label = _load()
    with torch.no_grad():
        enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        probs = F.softmax(mdl(**enc).logits, dim=-1)[0].cpu().numpy()

    p_neg = float(probs[[i for i, l in id2label.items() if l == "negative"][0]])
    p_pos = float(probs[[i for i, l in id2label.items() if l == "positive"][0]])
    top_idx = int(probs.argmax())
    label = id2label.get(top_idx, "neutral")
    score = max(-1.0, min(1.0, p_pos - p_neg))
    return Sentiment(score=score, label=label)
