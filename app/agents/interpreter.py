"""
Interpreter Agent.

Predicts (intent, customer_tag) from a transcript chunk.

Two modes:
  - "finetuned": loads two heads fine-tuned on the dataset (one per label set).
                 Set INTERPRETER_MODE=finetuned and provide INTERPRETER_INTENT_DIR /
                 INTERPRETER_CTAG_DIR (paths to saved model dirs).
  - "zeroshot":  uses an NLI model (DeBERTa-v3 NLI by default) for zero-shot
                 classification over the fixed label sets. Default mode so the
                 system runs out of the box.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from app.core.config import settings
from app.core.schemas import Interpretation


INTENTS: list[str] = [
    "Returns_and_Refunds", "Technical_Issues", "Upgrades_and_Promotions",
    "Loyalty Program", "Delay Management", "Delivery Delays",
    "Product Feedback", "Cancellation Policies", "Booking Errors",
    "Churn Prediction", "Cross-Brand Mentions", "Brand Loyalty",
    "Service Complaints", "Price Sensitivity", "Feature Requests",
    "Account Issues", "Billing Issues", "Refund Status",
    "Order Tracking", "Subscription Issues", "Promotion Inquiry",
    "General Inquiry", "Complaint Resolution", "Onboarding",
]

CUSTOMER_TAGS: list[str] = [
    "CUSTOMER_EXPRESSES_SATISFACTION", "CUSTOMER_EXPRESSES_FRUSTRATION",
    "CUSTOMER_STATES_ISSUE", "CUSTOMER_REQUESTS_FINANCIAL_RELIEF",
    "CUSTOMER_EXPRESSES_CONFUSION", "CUSTOMER_OTHER",
    "CUSTOMER_OBJECTS_TO_POLICY", "CUSTOMER_REQUESTS_ESCALATION",
    "CUSTOMER_PROVIDES_INFO", "CUSTOMER_MENTIONS_COMPETITOR",
    "CUSTOMER_THREATENS_CHURN",
]


def _device_index() -> int:
    return 0 if (settings.device == "cuda" and torch.cuda.is_available()) else -1


@lru_cache(maxsize=1)
def _zeroshot():
    nli_model = os.getenv("INTERPRETER_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    return pipeline("zero-shot-classification", model=nli_model, device=_device_index())


def _humanize(label: str) -> str:
    return label.replace("CUSTOMER_", "").replace("_", " ").lower()


_CTAG_HUMAN_TO_RAW = {_humanize(t): t for t in CUSTOMER_TAGS}


@lru_cache(maxsize=1)
def _finetuned():
    intent_dir = os.getenv("INTERPRETER_INTENT_DIR")
    ctag_dir = os.getenv("INTERPRETER_CTAG_DIR")
    if not intent_dir or not ctag_dir:
        raise RuntimeError("Set INTERPRETER_INTENT_DIR and INTERPRETER_CTAG_DIR for finetuned mode")
    device = "cuda" if (settings.device == "cuda" and torch.cuda.is_available()) else "cpu"

    tok_i = AutoTokenizer.from_pretrained(intent_dir)
    mdl_i = AutoModelForSequenceClassification.from_pretrained(intent_dir).to(device).eval()
    tok_c = AutoTokenizer.from_pretrained(ctag_dir)
    mdl_c = AutoModelForSequenceClassification.from_pretrained(ctag_dir).to(device).eval()
    return (tok_i, mdl_i, tok_c, mdl_c, device)


def _classify_finetuned(text: str) -> Interpretation:
    tok_i, mdl_i, tok_c, mdl_c, device = _finetuned()

    with torch.no_grad():
        ei = tok_i(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        oi = mdl_i(**ei).logits.softmax(-1)[0]
        ii = int(oi.argmax().item())
        intent = mdl_i.config.id2label.get(ii, INTENTS[ii] if ii < len(INTENTS) else "UNKNOWN")
        intent_conf = float(oi[ii].item())

        ec = tok_c(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        oc = mdl_c(**ec).logits.softmax(-1)[0]
        ic = int(oc.argmax().item())
        ctag = mdl_c.config.id2label.get(ic, CUSTOMER_TAGS[ic] if ic < len(CUSTOMER_TAGS) else "CUSTOMER_OTHER")
        ctag_conf = float(oc[ic].item())

    return Interpretation(
        intent=intent,
        customer_tag=ctag,
        intent_confidence=intent_conf,
        customer_tag_confidence=ctag_conf,
    )


def _classify_zeroshot(text: str) -> Interpretation:
    zs = _zeroshot()

    intent_labels = [i.replace("_", " ") for i in INTENTS]
    r1 = zs(text, candidate_labels=intent_labels, multi_label=False,
            hypothesis_template="The customer is calling about {}.")
    raw_intent = INTENTS[intent_labels.index(r1["labels"][0])]
    intent_conf = float(r1["scores"][0])

    ctag_labels = [_humanize(t) for t in CUSTOMER_TAGS]
    r2 = zs(text, candidate_labels=ctag_labels, multi_label=False,
            hypothesis_template="The customer {}.")
    ctag = _CTAG_HUMAN_TO_RAW[r2["labels"][0]]
    ctag_conf = float(r2["scores"][0])

    return Interpretation(
        intent=raw_intent,
        customer_tag=ctag,
        intent_confidence=intent_conf,
        customer_tag_confidence=ctag_conf,
    )


def interpret(text: str) -> Interpretation:
    text = (text or "").strip()
    if not text:
        return Interpretation(intent="General Inquiry", customer_tag="CUSTOMER_OTHER")
    mode = os.getenv("INTERPRETER_MODE", "zeroshot").lower()
    if mode == "finetuned":
        return _classify_finetuned(text)
    return _classify_zeroshot(text)
