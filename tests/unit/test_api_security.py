"""HTTP access-control regression checks without starting model infrastructure."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import get_settings


@pytest.fixture
def secure_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "ci-only-secret-value")
    get_settings.cache_clear()
    # Do not enter TestClient's context manager: OpenAPI and unauthorized
    # requests need no model weights, Redis, Qdrant or FastText startup.
    client = TestClient(create_app())
    yield client
    client.close()
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("method", "url", "json_body"),
    [
        ("post", "/chat", {"query": "Hello"}),
        ("post", "/evaluate", {}),
        ("get", "/sessions/private-session/history", None),
        ("delete", "/sessions/private-session", None),
        ("get", "/users/other-person/memories", None),
        ("delete", "/users/other-person/memories", None),
    ],
)
def test_model_and_memory_routes_require_api_key(secure_client, method, url, json_body):
    request = getattr(secure_client, method)
    kwargs = {"json": json_body} if json_body is not None else {}
    response = request(url, **kwargs)
    assert response.status_code == 401, (method, url, response.text)


def test_openapi_is_available_without_inference(secure_client):
    response = secure_client.get("/openapi.json")
    assert response.status_code == 200
    assert "/chat" in response.json()["paths"]


def test_missing_server_side_api_key_cannot_authenticate(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from src.api.middleware import verify_api_key

    monkeypatch.delenv("API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(verify_api_key(key="changeme"))
        assert exc.value.status_code == 503
    finally:
        get_settings.cache_clear()
