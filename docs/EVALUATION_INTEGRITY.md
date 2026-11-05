# Muḥāwir | Evaluation integrity and evidence boundaries

Muḥāwir provides metric implementations, evaluation fixtures and automated tests. **Fixture-based diagnostic scores must not be presented as measured live retrieval or RAG performance.**

## The critical distinction

| Mode | Input | What can be claimed |
| --- | --- | --- |
| Unit tests | Small synthetic examples and known answers | Correct metric calculations and deterministic code behavior |
| Golden set without a retriever | Reference answers and contexts from JSON | Heuristic proxy diagnostics only. MRR/NDCG/Recall are **not evaluated** |
| Real retriever wired into evaluator | Ranked document IDs actually produced for questions, with known relevance labels | Retrieval MRR, NDCG and Recall@K for that specific corpus and configuration |
| Real generator + authorized evaluator | Actual generated answers, retrieved contexts and judge model | Potentially meaningful answer quality metrics, subject to assessor limitations |

The code now **never inserts expected document IDs into a fabricated rank-1 results list**. If `retriever=None`, retrieval metrics are `None`, `retrieval_source="not_evaluated"`, and regression checks fail rather than claiming success. The `benchmarks_verified` flag also remains false unless a real retriever, a real generator and non-proxy answer evaluation are supplied.

## Running local diagnostic tests

```bash
python -m pytest tests/unit/test_evaluation.py tests/unit/test_api_security.py -v
python scripts/run_eval.py --golden-set data/golden_sets/golden_v1.0.json --output eval_results.json
```

The second command is deliberately expected to **fail the production regression gate** without a live retriever. Its JSON may still contain lexical-proxy scores. Those figures must be labeled **fixture proxies**, not RAGAS accuracy or successful production benchmarks.

For opt-in external RAGAS evaluation, `RAGAS_EVALUATOR_ENABLED=1` attempts to use a properly configured judge model. It still evaluates the input answer/context dataset; that alone does not prove the answers were produced by the deployed pipeline.

## Proper live evaluation procedure

1. Pin the repository commit, model revision, tokenizer, dependency versions and hardware configuration.
2. Prepare a private or public held-out corpus with meaningful `expected_doc_id` ground truth for each question. Do not reuse the training prompts as a test.
3. Run real indexing. Pass a retriever callable or object returning **ranked document IDs**, not your expected IDs.
4. Pass a real generator callable for responses and record the actual retrieved contexts used per request.
5. Evaluate retrieval metrics, relevant language groups and groundedness; keep raw traces and examples of failures.
6. Publish scores **only** with the corpus, query count, model, settings, timestamps and limitations.

## Access control

The API requires a configured `API_KEY` and `X-API-Key` header for chat, evaluation and memory/session data. OpenAPI can be inspected without starting a model; authentication is verified independently in unit tests and browser documentation smoke checks. This is a **single-service API-key boundary**, not full per-user authorization. Do not deploy it as a multi-tenant application without ownership enforcement and identity integration.

## Browser evidence

[API screenshot workflow](../.github/workflows/api-docs-screenshot.yml) serves the **real** FastAPI docs using `--lifespan off`. It checks that anonymous `/chat` returns HTTP 401 and captures the actual Swagger UI if its external assets load. That image is an API-documentation screenshot, **not** proof of a running LLM.
