from typing import Any

THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.65,
    "mrr_10": 0.60,
    "language_consistency": 0.90,
}


def check_regression(
    current_scores: dict[str, Any],
    thresholds: dict[str, float] = THRESHOLDS,
) -> tuple[bool, list[str]]:
    """Check whether evaluation metrics meet or exceed defined production thresholds.

    Returns:
      (passed, list of failure messages)
    """
    failures: list[str] = []

    for metric, threshold in thresholds.items():
        if metric not in current_scores:
            failures.append(f"Missing required metric '{metric}' (required >= {threshold})")
            continue

        raw_score = current_scores[metric]
        if raw_score is None:
            failures.append(f"Metric '{metric}' has not been evaluated")
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            failures.append(f"Metric '{metric}' has no numeric measured value")
            continue
        if not __import__("math").isfinite(score):
            failures.append(f"Metric '{metric}' has a non-finite value")
            continue
        if score < threshold:
            failures.append(
                f"Regression on '{metric}': score {score:.4f} < threshold {threshold:.4f} (diff: {score - threshold:+.4f})"
            )

    passed = len(failures) == 0
    return passed, failures
