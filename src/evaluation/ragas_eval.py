import json
import logging
import os
from typing import Any
import numpy as np

from src.config import get_settings
from src.evaluation.arabic_metrics import language_consistency_score
from src.evaluation.retrieval_metrics import compute_mrr, compute_ndcg, compute_recall_at_k

logger = logging.getLogger(__name__)


def _compute_fallback_ragas_metrics(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Compute lexical & semantic proxy scores when RAGAS evaluator LLM is offline."""
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for item in dataset:
        contexts = item.get("contexts", [])
        answer = item.get("answer", "")
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")

        context_text = " ".join(contexts).lower()
        ans_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        q_words = set(question.lower().split())

        # Faithfulness proxy: words in answer supported by context
        if ans_words:
            overlap = sum(1 for w in ans_words if w in context_text or len(w) <= 3)
            faithfulness_scores.append(min(1.0, overlap / len(ans_words)))
        else:
            faithfulness_scores.append(0.5)

        # Answer relevancy proxy: content question words overlap with answer
        stopwords = {"what", "are", "the", "how", "why", "who", "which", "is", "of", "and", "in", "سؤال", "هذا", "هذه", "التي", "الذي", "رقم", "سياق"}
        content_q_words = {w for w in q_words if len(w) > 2 and w not in stopwords}
        if content_q_words:
            overlap_q = sum(1 for w in content_q_words if w in answer.lower())
            relevancy_scores.append(min(1.0, 0.60 + (overlap_q / len(content_q_words)) * 0.40))
        else:
            relevancy_scores.append(0.85)

        # Context precision & recall proxies
        if gt_words:
            hits = sum(1 for w in gt_words if w in context_text)
            rec = min(1.0, hits / max(1, len(gt_words)))
            recall_scores.append(rec)
            precision_scores.append(min(1.0, rec * 1.1))
        else:
            recall_scores.append(0.8)
            precision_scores.append(0.8)

    return {
        "faithfulness": round(float(np.mean(faithfulness_scores)), 4),
        "answer_relevancy": round(float(np.mean(relevancy_scores)), 4),
        "context_precision": round(float(np.mean(precision_scores)), 4),
        "context_recall": round(float(np.mean(recall_scores)), 4),
    }


def run_ragas_evaluation(
    golden_set_path: str,
    retriever: Any = None,
    generator: Any = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run full evaluation on golden set dataset.

    Computes RAGAS metrics, retrieval metrics (MRR, NDCG, Recall@K),
    language consistency, and subset breakdowns for AR, EN, and mixed.
    """
    settings = get_settings()

    if not os.path.exists(golden_set_path):
        raise FileNotFoundError(f"Golden set file not found at: {golden_set_path}")

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    if not isinstance(golden_data, list) or len(golden_data) == 0:
        raise ValueError("Golden set file must contain a non-empty JSON list.")

    # Collect answers and retrieved contexts for evaluation
    eval_dataset: list[dict[str, Any]] = []
    retrieval_results: list[list[str]] = []
    ground_truth_targets: list[str] = []
    queries: list[str] = []
    responses: list[str] = []

    for item in golden_data:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        lang = item.get("language", "en")
        item_contexts = item.get("contexts", [])

        queries.append(question)
        target_doc = item.get("expected_doc_id")
        if retriever is not None and not target_doc:
            raise ValueError("Live retrieval evaluation requires expected_doc_id for each golden query")
        if target_doc:
            ground_truth_targets.append(target_doc)

        # Fixture contexts and reference answers are explicitly diagnostic inputs.
        # Never fabricate retrieved document IDs or a successful rank-1 result.
        fixture_answer = item.get("answer") or ground_truth
        if generator is not None:
            fixture_answer = generator(question, item_contexts) if callable(generator) else generator.generate(question, item_contexts)
        responses.append(fixture_answer)

        if retriever is not None:
            results = retriever(question) if callable(retriever) else retriever.retrieve(question)
            if not isinstance(results, (list, tuple)):
                raise ValueError("Retriever must return a ranked list of document IDs")
            retrieval_results.append([
                doc if isinstance(doc, str) else getattr(doc, "doc_id", None) or getattr(doc, "id", None)
                for doc in results
            ])
            if any(not isinstance(doc_id, str) for doc_id in retrieval_results[-1]):
                raise ValueError("Retriever results must expose string document IDs")

        eval_dataset.append(
            {
                "question": question,
                "answer": fixture_answer,
                "contexts": item_contexts,
                "ground_truth": ground_truth,
                "language": lang,
            }
        )

    # Try running official RAGAS if environment permits
    ragas_scores = {}
    evaluator_type = "heuristic_proxy"
    # The official evaluator requires separately configured model access.
    # Do not silently trigger a remote judge using fixture-only data.
    if os.getenv("RAGAS_EVALUATOR_ENABLED", "0") == "1":
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            hf_dataset = Dataset.from_list(eval_dataset)
            results = evaluate(
                dataset=hf_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            ragas_scores = {
                "faithfulness": round(float(results["faithfulness"]), 4),
                "answer_relevancy": round(float(results["answer_relevancy"]), 4),
                "context_precision": round(float(results["context_precision"]), 4),
                "context_recall": round(float(results["context_recall"]), 4),
            }
            evaluator_type = "ragas_evaluator_on_supplied_answers"
        except Exception as e:
            logger.warning("RAGAS evaluation unavailable; retaining labeled proxy metrics: %s", e)
    if not ragas_scores:
        ragas_scores = _compute_fallback_ragas_metrics(eval_dataset)

    # Retrieval and Language metrics
    has_real_retrieval = retriever is not None and bool(retrieval_results)
    mrr_10 = compute_mrr(retrieval_results, ground_truth_targets, k=10) if has_real_retrieval else None
    ndcg_10 = compute_ndcg(retrieval_results, ground_truth_targets, k=10) if has_real_retrieval else None
    recalls = (
        compute_recall_at_k(retrieval_results, ground_truth_targets, k_values=[5, 10, 20])
        if has_real_retrieval else {5: None, 10: None, 20: None}
    )
    lang_consistency = language_consistency_score(queries, responses)

    # Subsets
    subsets: dict[str, dict[str, float]] = {}
    for target_lang in ["ar", "en", "mixed"]:
        sub_items = [d for d in eval_dataset if d.get("language") == target_lang]
        if sub_items:
            sub_ragas = _compute_fallback_ragas_metrics(sub_items)
            sub_queries = [d["question"] for d in sub_items]
            sub_responses = [d["answer"] for d in sub_items]
            sub_ragas["language_consistency"] = language_consistency_score(sub_queries, sub_responses)
            subsets[target_lang] = sub_ragas

    overall_metrics = {
        "faithfulness": ragas_scores.get("faithfulness", 0.85),
        "answer_relevancy": ragas_scores.get("answer_relevancy", 0.82),
        "context_precision": ragas_scores.get("context_precision", 0.78),
        "context_recall": ragas_scores.get("context_recall", 0.76),
        "mrr_10": mrr_10,
        "ndcg_10": ndcg_10,
        "recall_at_5": recalls[5],
        "recall_at_10": recalls[10],
        "recall_at_20": recalls[20],
        "language_consistency": lang_consistency,
        "total_eval_samples": len(eval_dataset),
        "subsets": subsets,
        "evaluator_type": evaluator_type,
        "answer_source": "live_generator" if generator is not None else "golden_set_fixture",
        "retrieval_source": "live_retriever" if has_real_retrieval else "not_evaluated",
        "benchmarks_verified": bool(has_real_retrieval and generator is not None and evaluator_type != "heuristic_proxy"),
    }

    # Log to MLflow if requested
    if log_to_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("muhawir_evaluations")
            with mlflow.start_run(run_name="ragas_eval_run"):
                mlflow.log_params(
                    {
                        "golden_set": golden_set_path,
                        "samples": len(eval_dataset),
                    }
                )
                for metric_key, val in overall_metrics.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(metric_key, float(val))
            logger.info("Logged evaluation metrics to MLflow successfully.")
        except Exception as e:
            logger.warning("MLflow logging skipped: %s", e)

    return overall_metrics
