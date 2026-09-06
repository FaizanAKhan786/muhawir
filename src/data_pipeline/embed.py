import logging
from typing import Any
from src.data_pipeline.preprocess import simple_hash_vector

logger = logging.getLogger(__name__)

_EMBED_MODEL_CACHE: dict[str, Any] = {}


def get_embedding_model(model_name: str = "BAAI/bge-m3"):
    """Load and cache SentenceTransformer embedding model."""
    global _EMBED_MODEL_CACHE
    if model_name not in _EMBED_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", model_name)
            _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer (%s), fallback to hash embeddings.", e)
            _EMBED_MODEL_CACHE[model_name] = None
    return _EMBED_MODEL_CACHE[model_name]


def batch_embed(
    chunks: list[dict[str, Any]],
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Embed document chunks in batches, adding the 'embedding' field to each chunk."""
    if not chunks:
        return []

    texts = [c.get("text", "") for c in chunks]
    model = get_embedding_model(model_name)

    if model is not None:
        try:
            logger.info("Encoding %d chunks with %s...", len(texts), model_name)
            all_embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for chunk, emb in zip(chunks, all_embeddings):
                chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            return chunks
        except Exception as e:
            logger.warning("Batch encoding failed (%s), using deterministic fallback.", e)

    # Fallback to deterministic pseudo-embeddings (1024-dim to match BGE-M3)
    for chunk in chunks:
        chunk["embedding"] = simple_hash_vector(chunk.get("text", ""), dim=1024)

    return chunks
