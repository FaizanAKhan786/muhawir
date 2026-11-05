"""Smoke-check the actual Muḥāwir FastAPI schema and capture its rendered Swagger UI.

Runs with Uvicorn lifespan explicitly disabled. This verifies the public API
surface and authentication boundary, not LLM inference or external services.
"""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
with urlopen(BASE + "/openapi.json", timeout=15) as resp:
    schema = json.load(resp)
assert "/chat" in schema["paths"]
assert "/evaluate" in schema["paths"]
assert "/sessions/{session_id}/history" in schema["paths"]
# Requests without credentials must not reach expensive agent logic.
request = Request(
    BASE + "/chat",
    data=b'{"query":"Hello"}',
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urlopen(request, timeout=15)
except HTTPError as exc:
    assert exc.code == 401, f"Unauthenticated chat returned {exc.code}"
else:
    raise AssertionError("Unauthenticated chat was accepted")

out = Path("docs/screenshots")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    problems = []
    page.on("pageerror", lambda err: problems.append(str(err)))
    response = page.goto(BASE + "/docs", wait_until="domcontentloaded", timeout=25000)
    assert response and response.status == 200
    page.locator(".swagger-ui .info .title").wait_for(timeout=35000)
    assert "Muḥāwir" in page.locator(".swagger-ui .info .title").inner_text()
    assert page.locator("#operations-default-post_chat").count() or page.locator(".opblock").count()
    assert not problems, problems[:3]
    page.screenshot(
        path=str(out / "muhawir-api-docs.png"),
        full_page=True,
        animations="disabled",
    )
    browser.close()
print("VERIFIED: actual FastAPI OpenAPI routes, 401 for anonymous chat, rendered Swagger UI screenshot")
