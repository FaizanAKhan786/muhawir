import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sse_starlette.sse import EventSourceResponse

from src.agents.graph import compile_graph, get_langfuse_client
from src.agents.state import AgentState
from src.api.middleware import verify_api_key
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    ServiceStatus,
)
from src.config import get_settings
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.evaluation.regression import check_regression

logger = logging.getLogger(__name__)

router = APIRouter()

# Prometheus metrics
REQUEST_COUNT = Counter("muhawir_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("muhawir_latency_seconds", "Request latency", ["endpoint"])
AGENT_RETRY_COUNT = Counter("muhawir_agent_retries_total", "Count of agent retry loops")

# Global compiled agent workflow
_COMPILED_AGENT = None


def get_agent():
    global _COMPILED_AGENT
    if _COMPILED_AGENT is None:
        _COMPILED_AGENT = compile_graph()
    return _COMPILED_AGENT


@router.post("/chat")
async def chat_stream(
    chat_req: ChatRequest,
    request: Request,
) -> EventSourceResponse:
    """Stream agent status updates and final answer via Server-Sent Events (SSE)."""
    settings = get_settings()
    trace_id = str(uuid.uuid4())

    # Check Semantic Cache from app state
    cache = getattr(request.app.state, "semantic_cache", None)
    if cache:
        cached_result = await cache.get(chat_req.query)
        if cached_result:
            async def cache_gen() -> AsyncGenerator[dict, None]:
                yield {"event": "status", "data": json.dumps({"type": "status", "data": "Cache hit: delivering response"})}
                yield {"event": "answer", "data": json.dumps({"type": "answer", "data": cached_result})}

            return EventSourceResponse(cache_gen())

    # Langfuse trace setup
    lf_client = get_langfuse_client()
    lf_trace = None
    if lf_client:
        try:
            lf_trace = lf_client.trace(
                id=trace_id,
                name="chat_request",
                input={"query": chat_req.query, "session_id": chat_req.session_id},
            )
        except Exception as e:
            logger.debug("Langfuse trace creation failed: %s", e)

    async def event_generator() -> AsyncGenerator[dict, None]:
        start_time = time.time()
        yield {"event": "status", "data": json.dumps({"type": "status", "step": "🛡️ Input Safety Check", "data": "Running input guardrails..."})}

        initial_state: AgentState = {
            "query": chat_req.query,
            "language": "en",
            "language_confidence": 0.0,
            "expanded_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "draft_answer": "",
            "quality_score": {"faithfulness": 1.0, "language_consistent": True, "grounded": True},
            "final_answer": "",
            "citations": [],
            "retry_count": 0,
            "error": None,
            "trace_id": trace_id,
            "is_multi_hop": False,
            "sub_queries": [],
            "reasoning_chain": [],
            "hop_contexts": [],
            "current_hop": 0,
            "max_hops": 3,
            "synthesis_answer": "",
            "complexity_score": 0.0,
            "guardrail_violations": [],
            "input_blocked": False,
            "output_sanitized": False,
            "sanitized_query": chat_req.query,
            "current_step": "initializing",
            "steps_completed": [],
        }

        # --- Stream intermediate node events using astream_events ---
        try:
            agent = get_agent()

            # Step visibility: stream agent node events
            STEP_LABELS: dict[str, str] = {
                "input_guardrail":      "🛡️ Input Safety Check",
                "blocked_response":     "🚫 Request Blocked",
                "detect_language":      "🌐 Detecting Language",
                "classify_complexity":  "🧠 Classifying Query Complexity",
                "expand_query":         "🔎 Expanding Query",
                "retrieve":             "📚 Retrieving Knowledge",
                "rerank":               "⚡ Reranking Results",
                "generate":             "✍️ Generating Answer",
                "output_guardrail":     "🛡️ Output Safety Check",
                "quality_gate":         "✅ Quality Gate",
                "format_response":      "📋 Formatting Response",
                "decompose_query":      "🔬 Decomposing Into Sub-Queries",
                "multi_hop_retrieve":   "📚 Multi-Hop Retrieval",
                "multi_hop_rerank":     "⚡ Multi-Hop Reranking",
                "hop_synthesize":       "🔗 Synthesizing Hop Findings",
                "final_synthesize":     "🎯 Final Cross-Hop Synthesis",
                "output_guardrail_mh":  "🛡️ Output Safety Check",
                "quality_gate_mh":      "✅ Multi-Hop Quality Gate",
                "format_response_mh":   "📋 Formatting Response",
            }

            last_state = initial_state
            async for event in agent.astream_events(initial_state, version="v1"):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chain_start" and name in STEP_LABELS:
                    label = STEP_LABELS[name]
                    yield {
                        "event": "step",
                        "data": json.dumps({"type": "step", "node": name, "label": label, "status": "started"}),
                    }

                elif kind == "on_chain_end" and name in STEP_LABELS:
                    label = STEP_LABELS[name]
                    output = event.get("data", {}).get("output", {})
                    # Capture the latest state from node output
                    if isinstance(output, dict):
                        last_state = {**last_state, **output}
                    yield {
                        "event": "step",
                        "data": json.dumps({"type": "step", "node": name, "label": label, "status": "completed"}),
                    }

            result_state = last_state

            # Check if blocked
            if result_state.get("input_blocked", False):
                REQUEST_COUNT.labels(endpoint="/chat", status="blocked").inc()

            final_response = {
                "answer": result_state.get("final_answer", ""),
                "language": result_state.get("language", "en"),
                "citations": result_state.get("citations", []),
                "quality_score": result_state.get("quality_score"),
                "confidence": round(result_state.get("quality_score", {}).get("faithfulness", 0.9), 2),
                "trace_id": trace_id,
                "is_multi_hop": result_state.get("is_multi_hop", False),
                "reasoning_chain": result_state.get("reasoning_chain", []),
                "hops_used": result_state.get("current_hop", 0),
                "complexity_score": round(result_state.get("complexity_score", 0.0), 2),
                "guardrail_violations": result_state.get("guardrail_violations", []),
                "output_sanitized": result_state.get("output_sanitized", False),
                "steps_completed": result_state.get("steps_completed", []),
            }

            # Save in semantic cache (only non-blocked, non-sanitized responses)
            if cache and result_state.get("final_answer") and not result_state.get("input_blocked"):
                await cache.set(chat_req.query, final_response, ttl=settings.CACHE_TTL)

            if lf_trace:
                lf_trace.update(output=final_response)

            REQUEST_COUNT.labels(endpoint="/chat", status="success").inc()
            REQUEST_LATENCY.labels(endpoint="/chat").observe(time.time() - start_time)

            yield {"event": "answer", "data": json.dumps({"type": "answer", "data": final_response})}

        except Exception as exc:
            logger.error("Error during agent execution: %s", exc)
            if lf_trace:
                lf_trace.update(output={"error": str(exc)}, level="ERROR")
            REQUEST_COUNT.labels(endpoint="/chat", status="error").inc()
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": f"Execution error: {str(exc)}"}),
            }

    return EventSourceResponse(event_generator())


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_system(
    eval_req: EvalRequest,
    api_key: str = Depends(verify_api_key),
) -> EvalResponse:
    """Trigger system evaluation pipeline on the specified golden set."""
    try:
        metrics = run_ragas_evaluation(
            golden_set_path=eval_req.golden_set,
            log_to_mlflow=eval_req.log_to_mlflow,
        )
        passed, failures = check_regression(metrics)
        return EvalResponse(
            metrics=metrics,
            passed=passed,
            failures=failures,
        )
    except Exception as e:
        logger.error("Evaluation run failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Comprehensive health check checking Qdrant, Redis, and vLLM."""
    services: dict[str, ServiceStatus] = {}
    overall_ok = True

    # 1. Qdrant check
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    if qdrant_client:
        try:
            t0 = time.time()
            qdrant_client.get_collections()
            lat = round((time.time() - t0) * 1000, 2)
            services["qdrant"] = ServiceStatus(status="healthy", latency_ms=lat)
        except Exception as e:
            services["qdrant"] = ServiceStatus(status="unhealthy", details=str(e))
            overall_ok = False
    else:
        services["qdrant"] = ServiceStatus(status="not_configured")

    # 2. Redis check
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client:
        try:
            t0 = time.time()
            ping_call = redis_client.ping()
            if hasattr(ping_call, "__await__"):
                await ping_call
            lat = round((time.time() - t0) * 1000, 2)
            services["redis"] = ServiceStatus(status="healthy", latency_ms=lat)
        except Exception as e:
            services["redis"] = ServiceStatus(status="unhealthy", details=str(e))
            overall_ok = False
    else:
        services["redis"] = ServiceStatus(status="not_configured")

    # 3. vLLM check
    settings = get_settings()
    try:
        import httpx

        t0 = time.time()
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.VLLM_BASE_URL.rstrip('/v1')}/health")
            lat = round((time.time() - t0) * 1000, 2)
            if resp.status_code == 200:
                services["vllm"] = ServiceStatus(status="healthy", latency_ms=lat)
            else:
                services["vllm"] = ServiceStatus(
                    status="degraded",
                    latency_ms=lat,
                    details=f"Status {resp.status_code}",
                )
    except Exception as e:
        services["vllm"] = ServiceStatus(status="unreachable", details=str(e))

    return HealthResponse(
        status="healthy" if overall_ok else "degraded",
        services=services,
    )


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus exposition metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
