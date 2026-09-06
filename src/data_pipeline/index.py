import logging
from typing import Any

try:
    from qdrant_client.http import models as rest_models
except ImportError:
    rest_models = None  # type: ignore

from src.retrieval.hybrid import BM25Index

logger = logging.getLogger(__name__)


def create_qdrant_collection(
    client: Any,
    collection_name: str,
    vector_size: int = 1024,
) -> bool:
    """Create a Qdrant collection with Cosine distance if it doesn't already exist."""
    if client is None:
        logger.warning("Qdrant client is None, skipping collection creation.")
        return False

    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest_models.VectorParams(
                    size=vector_size,
                    distance=rest_models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (size=%d)", collection_name, vector_size)
        else:
            logger.info("Collection '%s' already exists.", collection_name)
        return True
    except Exception as e:
        logger.error("Error creating Qdrant collection '%s': %s", collection_name, e)
        return False


def index_chunks(
    client: Any,
    collection_name: str,
    chunks_with_embeddings: list[dict[str, Any]],
) -> int:
    """Upsert chunk records with embeddings and payloads into Qdrant."""
    if client is None or not chunks_with_embeddings:
        return 0

    points = []
    for idx, chunk in enumerate(chunks_with_embeddings):
        chunk_id = chunk.get("chunk_id", f"chk_{idx}")
        # Convert string ID to uuid or integer hash for Qdrant compatibility if needed
        import uuid

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        embedding = chunk.get("embedding")
        if not embedding:
            continue

        payload = {
            "chunk_id": chunk_id,
            "doc_id": chunk.get("doc_id", ""),
            "text": chunk.get("text", ""),
            "language": chunk.get("language", "en"),
            "section_title": chunk.get("section_title", ""),
            "metadata": chunk.get("metadata", {}),
        }

        points.append(
            rest_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    try:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )
        logger.info("Indexed %d chunks into Qdrant collection '%s'", len(points), collection_name)
        return len(points)
    except Exception as e:
        logger.error("Error indexing chunks into Qdrant: %s", e)
        return 0


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Index:
    """Build and return an in-memory BM25 index from chunks."""
    index = BM25Index(corpus=chunks)
    logger.info("Constructed BM25 index with %d documents.", len(chunks))
    return index
