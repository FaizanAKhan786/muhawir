import hashlib
import re
from typing import Any
import numpy as np
from src.retrieval.chunking import normalize_arabic, normalize_english


def simple_hash_vector(text: str, dim: int = 128) -> list[float]:
    """Generate a deterministic normalized vector based on character n-grams.

    Useful for offline testing and fast mock fallbacks.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        tokens = [text]
    for token in tokens:
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def preprocess_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Clean and normalize a document dictionary."""
    cleaned = dict(doc)
    text = cleaned.get("text", "")
    lang = cleaned.get("language", "en")

    if lang == "ar":
        cleaned_text = normalize_arabic(text)
    else:
        cleaned_text = normalize_english(text)

    # Standardize whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned["text"] = cleaned_text

    if "title" in cleaned and cleaned["title"]:
        title_lang = cleaned.get("language", "en")
        cleaned["title"] = (
            normalize_arabic(cleaned["title"])
            if title_lang == "ar"
            else normalize_english(cleaned["title"])
        )

    return cleaned
