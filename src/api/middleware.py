import hashlib
import hmac
import inspect
import json
import logging
import time
from typing import Any, Callable
from fastapi import HTTPException, Request, Response, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In-memory fallback rate limiter
_IN_MEMORY_RATE_LIMITS: dict[str, list[float]] = {}


async def verify_api_key(
    key: str = Security(api_key_header),
) -> str:
    """Validate API key from X-API-Key header."""
    settings = get_settings()
    if not settings.API_KEY or settings.API_KEY.lower() in {"changeme", "your-api-key-here"}:
        raise HTTPException(status_code=503, detail="Server API key has not been securely configured")
    if not key or not hmac.compare_digest(key, settings.API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key (X-API-Key header).",
        )
    return key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter enforcing 30 requests per minute per API key via Redis."""

    def __init__(self, app: Any, redis_client: Any = None, max_requests: int = 30, window_sec: int = 60):
        super().__init__(app)
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_sec = window_sec

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exclude health and metrics from rate limits
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        api_key = request.headers.get(API_KEY_NAME, "anonymous")
        now = time.time()
        key_name = f"ratelimit:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        is_rate_limited = False

        if self.redis is not None:
            try:
                # Sliding window log in Redis
                pipe = self.redis.pipeline() if hasattr(self.redis, "pipeline") else None
                if pipe:
                    pipe.zremrangebyscore(key_name, 0, now - self.window_sec)
                    pipe.zadd(key_name, {str(now): now})
                    pipe.zcard(key_name)
                    pipe.expire(key_name, self.window_sec)
                    pending = pipe.execute()
                    res = await pending if inspect.isawaitable(pending) else pending
                    current_count = res[2]
                else:
                    current_count = 1

                if current_count > self.max_requests:
                    is_rate_limited = True
            except Exception as e:
                logger.warning("Redis rate limiter error, falling back to in-memory: %s", e)
                is_rate_limited = self._in_memory_rate_limit(api_key, now)
        else:
            is_rate_limited = self._in_memory_rate_limit(api_key, now)

        if is_rate_limited:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded (30 requests/minute)."}),
                status_code=429,
                media_type="application/json",
            )

        return await call_next(request)

    def _in_memory_rate_limit(self, key: str, now: float) -> bool:
        timestamps = _IN_MEMORY_RATE_LIMITS.setdefault(key, [])
        # Expire older timestamps
        _IN_MEMORY_RATE_LIMITS[key] = [t for t in timestamps if t > now - self.window_sec]
        if len(_IN_MEMORY_RATE_LIMITS[key]) >= self.max_requests:
            return True
        _IN_MEMORY_RATE_LIMITS[key].append(now)
        return False


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Structured request logging middleware with sensitive data redaction."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # Redact query params / auth headers
        headers_log = dict(request.headers)
        if API_KEY_NAME.lower() in headers_log:
            headers_log[API_KEY_NAME.lower()] = "***REDACTED***"
        if "authorization" in headers_log:
            headers_log["authorization"] = "***REDACTED***"

        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
        }
        logger.info("HTTP Request: %s", json.dumps(log_data))
        return response
