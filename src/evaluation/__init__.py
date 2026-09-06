"""Evaluation module for Muḥāwir."""

from src.evaluation.arabic_metrics import code_switch_handling_score, language_consistency_score
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.evaluation.regression import THRESHOLDS, check_regression
from src.evaluation.retrieval_metrics import compute_mrr, compute_ndcg, compute_recall_at_k

__all__ = [
    "run_ragas_evaluation",
    "language_consistency_score",
    "code_switch_handling_score",
    "compute_mrr",
    "compute_ndcg",
    "compute_recall_at_k",
    "check_regression",
    "THRESHOLDS",
]
