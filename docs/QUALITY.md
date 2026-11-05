# Muḥāwir | Quality & reproducibility

> **Status rule:** Check the [live GitHub Actions history](https://github.com/FaizanAKhan786/muhawir/actions) for real workflow conclusions. This document does not invent benchmark scores or imply every feature was tested.

## Automated verification

Python compilation; selected fatal Ruff rules; unit tests. GPU-backed RAG evaluation is not part of the ordinary hosted-runner CI path.

- [View the CI workflow](https://github.com/FaizanAKhan786/muhawir/actions/workflows/ci.yml)
- Any failing run is an unresolved result until corrected; a green syntax/unit-check workflow is not proof of deployment readiness.

## Reproduce the checks

Run these commands from the **repository root**, unless the commands change directories:

```bash
python -m pytest tests/unit/ -v
python scripts/run_eval.py --golden-set data/golden_sets/golden_v1.0.json --output eval_results.json
```

## How to interpret the results

**Relevant metrics:** RAG faithfulness, retrieval MRR, language consistency, NDCG/recall and recorded quality gates.

**Requirements and limitations:** Provision the vLLM model endpoint, indexed corpus, Qdrant and any dependencies used by the evaluation. The golden set is a project fixture, not a third-party certification.

Before comparing results, save the commit SHA, command, dataset and split (including seed), model revision, dependency environment, hardware/CPU/GPU, and the raw result file. Use the same evaluation set for all compared runs and distinguish **synthetic tests** from **real-world measurements**.

## Publication policy

Only publish numeric scores alongside the run configuration and raw evidence. A threshold, code assertion or example report must **not** be presented as a measured production result.

[Project README](../README.md) · [GitHub Actions](https://github.com/FaizanAKhan786/muhawir/actions)
