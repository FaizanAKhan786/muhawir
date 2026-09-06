import hashlib
import re
import unicodedata
from typing import Any
import numpy as np


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


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by standardizing alef variants, removing tashkeel and tatweel."""
    if not text:
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove tashkeel / harakat (diacritics: Fatha, Damma, Kasra, Sukun, Tanwin, Shadda, etc.)
    # Unicode range for Arabic diacritics: U+064B - U+065F, U+0670
    tashkeel_regex = re.compile(r"[\u064B-\u065F\u0670]")
    text = tashkeel_regex.sub("", text)

    # Remove tatweel / kashida (ـ U+0640)
    text = re.sub(r"\u0640", "", text)

    # Normalize alef variants (أ, إ, آ, ٱ -> ا)
    text = re.sub(r"[أإآٱ]", "ا", text)

    # Normalize taa marbuta (ة -> ه) and alif maqsura (ى -> ي) optionally for general matching
    # Standard rule: preserve ة or convert if desired, but alef variants is mandatory
    # Also standardize common punctuation
    text = text.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_english(text: str) -> str:
    """Normalize English text by lowercasing and Unicode NFKC normalization."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str, language: str = "auto") -> list[str]:
    """Split text into sentences respecting Arabic and English punctuation boundaries."""
    # Arabic sentence terminators: . (full stop), ۔ (Arabic full stop), ، (Arabic comma for long clauses), ؟ (question mark), !
    # English sentence terminators: ., ?, !
    sentence_delimiters = r"(?<=[.!?۔؟])\s+"
    raw_sentences = re.split(sentence_delimiters, text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict[str, Any]]:
    """Semantic chunking with sliding window respecting sentence boundaries.

    Each document dict should have: doc_id, text, language (optional), metadata (optional).
    Returns list of chunk dicts:
      chunk_id, doc_id, text, language, chunk_index, section_title, metadata
    """
    all_chunks: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = doc.get("doc_id", "doc_unknown")
        doc_text = doc.get("text", "")
        lang = doc.get("language", "en")
        base_meta = doc.get("metadata", {})
        section_title = doc.get("title") or base_meta.get("title", "")

        if not doc_text.strip():
            continue

        sentences = split_into_sentences(doc_text, language=lang)

        current_chunk_words: list[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size and we already have content
            if current_len + sentence_len > chunk_size and current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_id}_c{chunk_idx}",
                        "doc_id": doc_id,
                        "text": chunk_text,
                        "language": lang,
                        "chunk_index": chunk_idx,
                        "section_title": section_title,
                        "metadata": {
                            **base_meta,
                            "source_doc": doc_id,
                            "chunk_index": chunk_idx,
                            "section_title": section_title,
                            "char_count": len(chunk_text),
                        },
                    }
                )
                chunk_idx += 1

                # Sliding window overlap
                overlap_words: list[str] = []
                overlap_len = 0
                for w in reversed(current_chunk_words):
                    if overlap_len + len(w) + 1 <= overlap:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break

                current_chunk_words = list(overlap_words)
                current_len = sum(len(w) for w in current_chunk_words) + max(
                    0, len(current_chunk_words) - 1
                )

            current_chunk_words.extend(sentence_words)
            current_len += sentence_len + 1

        # Emit trailing chunk
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "language": lang,
                    "chunk_index": chunk_idx,
                    "section_title": section_title,
                    "metadata": {
                        **base_meta,
                        "source_doc": doc_id,
                        "chunk_index": chunk_idx,
                        "section_title": section_title,
                        "char_count": len(chunk_text),
                    },
                }
            )

    return all_chunks
