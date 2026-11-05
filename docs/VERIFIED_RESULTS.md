# Muḥāwir | Evidence-backed verification

## Automated source and unit checks

**Verified GitHub Actions run:** [#42125487335](https://github.com/FaizanAKhan786/muhawir/actions/runs/42125487335)  
**Commit:** `6b1767cd4461c836f9d1e72d42ec3191c475722d`  
**Checked on:** 2026-11-05 (GitHub-hosted Linux runner, Python 3.11)

| Check | Observed result |
| --- | --- |
| Unit tests | **83 passed in 1.11 seconds** |
| Selected fatal Ruff checks and syntax compilation | **Passed** |
| Hardware-backed RAG regression gate | **Not executed**; requires separate configured model infrastructure |

The unit tests include API-key access checks, graph logic, Arabic processing, evaluation score calculations, guardrail behavior, multi-hop pipeline utilities, memory, and retrieval helper tests. These confirm **local code behavior with test fixtures**. They do not certify a running end-to-end production model.

## Evaluation integrity

The previous evaluation implementation had fabricated a rank-1 retrieval list by injecting the expected document ID. This has been corrected. Without a real retriever, the evaluation API now marks retrieval scores `None`; without live generation and suitable evaluation, answer scores remain `None` and fixture-only values appear separately under `proxy_diagnostics`.

See [Evaluation Integrity](EVALUATION_INTEGRITY.md) for how to attach genuine retrieval and generation adapters and collect evidence. **No measured live MRR, Recall@K, BLEU, faithfulness, end-to-end latency or production uptime figures are claimed here.**

## What is demonstrated through the API docs check

The independent [API-only screenshot workflow](../.github/workflows/api-docs-screenshot.yml) starts the real FastAPI application with heavy model startup disabled, checks that `/chat` rejects an anonymous request with HTTP 401 and renders Swagger/OpenAPI if browser assets load. Its screenshots demonstrate the API schema and auth boundary, **not** inference quality.

All verification status is dynamic. For later changes, consult [the Actions history](https://github.com/FaizanAKhan786/muhawir/actions).
