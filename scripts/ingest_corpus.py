import argparse
import logging
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_settings
from src.data_pipeline.embed import batch_embed
from src.data_pipeline.index import build_bm25_index, create_qdrant_collection, index_chunks
from src.data_pipeline.ingest import ingest_documents
from src.data_pipeline.preprocess import preprocess_document
from src.retrieval.chunking import chunk_documents
from src.retrieval.hybrid import set_retrieval_components

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_corpus")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest, preprocess, chunk, embed, and index document corpus.")
    parser.add_argument("--source", type=str, default="data/sample_docs/", help="Source directory containing documents")
    parser.add_argument("--collection", type=str, default=None, help="Target Qdrant collection name")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()
    collection_name = args.collection or settings.QDRANT_COLLECTION

    logger.info("Starting ingestion from '%s' into collection '%s'...", args.source, collection_name)

    # 1. Ingest raw documents
    raw_docs = ingest_documents(args.source)
    if not raw_docs:
        logger.warning("No documents found in '%s'. Ingestion finished with 0 documents.", args.source)
        return

    logger.info("Found %d raw documents.", len(raw_docs))

    # 2. Preprocess documents
    clean_docs = [preprocess_document(doc) for doc in raw_docs]

    # 3. Chunk documents
    chunks = chunk_documents(clean_docs, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    logger.info("Generated %d semantic chunks.", len(chunks))

    # 4. Batch embed chunks
    chunks_with_embeddings = batch_embed(chunks, model_name=settings.EMBEDDING_MODEL)

    # 5. Index into Qdrant if client accessible
    try:
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)
        create_qdrant_collection(qdrant_client, collection_name, vector_size=1024)
        indexed_count = index_chunks(qdrant_client, collection_name, chunks_with_embeddings)
        logger.info("Indexed %d chunks into Qdrant.", indexed_count)
    except Exception as e:
        logger.warning("Could not connect to live Qdrant instance (%s). Proceeding with local BM25 index.", e)

    # 6. Build in-memory BM25 index
    bm25_idx = build_bm25_index(chunks)

    # Register in document store
    doc_store = {c["chunk_id"]: c for c in chunks}
    set_retrieval_components(bm25=bm25_idx, document_store=doc_store)

    logger.info("Ingestion completed successfully for %d chunks.", len(chunks))


if __name__ == "__main__":
    main()
