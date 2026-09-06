import math
from typing import Sequence, Union

GroundTruthType = Union[Sequence[str], Sequence[Sequence[str]]]


def _normalize_gt(gt: Union[str, Sequence[str]]) -> set[str]:
    """Ensure ground truth target is a set of string IDs."""
    if isinstance(gt, str):
        return {gt}
    return set(gt)


def compute_mrr(
    results: Sequence[Sequence[str]],
    ground_truth: Sequence[Union[str, Sequence[str]]],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank (MRR@k).

    results: list of lists of retrieved document/chunk IDs.
    ground_truth: list of target IDs (or list of sets/lists of relevant IDs) per query.
    """
    if not results or not ground_truth or len(results) != len(ground_truth):
        return 0.0

    reciprocal_ranks: list[float] = []

    for retrieved, gt in zip(results, ground_truth):
        targets = _normalize_gt(gt)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in targets:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def compute_ndcg(
    results: Sequence[Sequence[str]],
    ground_truth: Sequence[Union[str, Sequence[str]]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@k) with binary relevance."""
    if not results or not ground_truth or len(results) != len(ground_truth):
        return 0.0

    ndcg_scores: list[float] = []

    for retrieved, gt in zip(results, ground_truth):
        targets = _normalize_gt(gt)
        if not targets:
            ndcg_scores.append(1.0)
            continue

        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            rel = 1.0 if doc_id in targets else 0.0
            if rel > 0:
                dcg += rel / math.log2(rank + 1)

        # Compute Ideal DCG
        idcg = 0.0
        num_ideal = min(len(targets), k)
        for rank in range(1, num_ideal + 1):
            idcg += 1.0 / math.log2(rank + 1)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return round(sum(ndcg_scores) / len(ndcg_scores), 4)


def compute_recall_at_k(
    results: Sequence[Sequence[str]],
    ground_truth: Sequence[Union[str, Sequence[str]]],
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[int, float]:
    """Compute Recall@K across multiple cutoff values."""
    if not results or not ground_truth or len(results) != len(ground_truth):
        return {k: 0.0 for k in k_values}

    recall_totals: dict[int, float] = {k: 0.0 for k in k_values}
    n = len(results)

    for retrieved, gt in zip(results, ground_truth):
        targets = _normalize_gt(gt)
        if not targets:
            for k in k_values:
                recall_totals[k] += 1.0
            continue

        for k in k_values:
            retrieved_k = set(retrieved[:k])
            hits = len(targets.intersection(retrieved_k))
            recall_totals[k] += hits / len(targets)

    return {k: round(total / n, 4) for k, total in recall_totals.items()}
