from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from src.agents.graph import compile_graph, get_langfuse_client
from src.agents.nodes import set_openai_client, set_reranker
from src.api.middleware import RateLimitMiddleware, StructuredLoggingMiddleware
from src.api.routes import router as api_router
from src.config import get_settings
from src.retrieval.cache import SemanticCache
from src.retrieval.hybrid import BM25Index, VectorSearcher, set_retrieval_components
from src.retrieval.reranker import NeuralReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("muhawir")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("Initializing Muḥāwir knowledge engine...")

    # 1. Initialize OpenAI client pointing to vLLM
    openai_client = AsyncOpenAI(
        base_url=settings.VLLM_BASE_URL,
        api_key=settings.API_KEY or "none",
    )
    set_openai_client(openai_client)
    app.state.openai_client = openai_client

    # 2. Initialize Qdrant client
    qdrant_client = None
    try:
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=5.0)
        app.state.qdrant_client = qdrant_client
        logger.info("Connected to Qdrant at %s:%d", settings.QDRANT_HOST, settings.QDRANT_PORT)
    except Exception as e:
        logger.warning("Could not connect to Qdrant at startup: %s", e)
        app.state.qdrant_client = None

    # 3. Initialize Redis client & Semantic Cache
    redis_client = None
    semantic_cache = None
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False,
        )
        app.state.redis_client = redis_client
        semantic_cache = SemanticCache(
            redis_client=redis_client,
            threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        )
        app.state.semantic_cache = semantic_cache
        logger.info("Connected to Redis at %s:%d", settings.REDIS_HOST, settings.REDIS_PORT)
    except Exception as e:
        logger.warning("Could not connect to Redis at startup: %s", e)
        app.state.redis_client = None
        app.state.semantic_cache = None

    # 4. Initialize Neural Reranker & Vector Searcher
    reranker = NeuralReranker(model_name=settings.RERANKER_MODEL)
    set_reranker(reranker)

    vector_searcher = VectorSearcher(
        qdrant_client=qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
    )
    bm25_index = BM25Index()
    set_retrieval_components(bm25=bm25_index, vector_searcher=vector_searcher)

    # 5. Compile LangGraph workflow & warm up Langfuse
    app.state.agent = compile_graph()
    get_langfuse_client()

    logger.info("Muḥāwir application startup complete. Ready to serve bilingual queries.")

    yield

    # Shutdown
    logger.info("Shutting down Muḥāwir application...")
    if redis_client is not None:
        try:
            await redis_client.close()
        except Exception as e:
            logger.debug("Error closing Redis client: %s", e)
    if qdrant_client is not None:
        try:
            qdrant_client.close()
        except Exception as e:
            logger.debug("Error closing Qdrant client: %s", e)


def create_app() -> FastAPI:
    """Build FastAPI application with middleware and routes."""
    app = FastAPI(
        title="Muḥāwir (محاور) API",
        description="Production-grade Bilingual (Arabic/English) Agentic Knowledge Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom Middleware
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=30, window_sec=60)

    # Routes
    app.include_router(api_router)

    return app


app = create_app()
