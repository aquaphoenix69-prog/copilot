"""
Speech-to-text using NVIDIA Parakeet-TDT-0.6B-v2.

Loads via NeMo's ASRModel.from_pretrained. Audio in: numpy float32 mono 16kHz
or a path to a wav/flac file. Text out.
"""

from __future__ import annotations

import io
import tempfile
from functools import lru_cache
from typing import Union

import numpy as np
import soundfile as sf

from app.core.config import settings


@lru_cache(maxsize=1)
def _load_model():
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name=settings.stt_model)
    model.eval()
    if settings.device == "cuda":
        try:
            model = model.cuda()
        except Exception:
            pass
    return model


def transcribe(audio: Union[str, bytes, np.ndarray], sample_rate: int = 16000) -> str:
    model = _load_model()

    if isinstance(audio, str):
        path = audio
        result = model.transcribe([path])
    else:
        if isinstance(audio, bytes):
            arr, sr = sf.read(io.BytesIO(audio), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
        else:
            arr = audio.astype("float32")
            sr = sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, sr, subtype="PCM_16")
            path = tmp.name
        result = model.transcribe([path])

    if not result:
        return ""
    first = result[0]
    if isinstance(first, str):
        return first
    return getattr(first, "text", str(first))
