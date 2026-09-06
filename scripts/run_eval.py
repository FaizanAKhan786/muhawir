import argparse
import io
import json
import logging
import os
import sys

# Ensure UTF-8 output on Windows terminals
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.ragas_eval import run_ragas_evaluation
from src.evaluation.regression import THRESHOLDS, check_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline on golden benchmark set.")
    parser.add_argument(
        "--golden-set",
        type=str,
        default="data/golden_sets/golden_v1.0.json",
        help="Path to golden benchmark dataset JSON file",
    )
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        default=False,
        help="Log evaluation metrics and artifacts to MLflow server",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output filepath for evaluation results JSON",
    )
    return parser.parse_args()


def print_summary_table(metrics: dict, passed: bool, failures: list[str]) -> None:
    print("\n" + "=" * 75)
    print("                MUḤĀWIR (محاور) EVALUATION REPORT                ")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Current Score':<15} | {'Threshold':<12} | {'Status':<8}")
    print("-" * 75)

    for metric, threshold in THRESHOLDS.items():
        score = metrics.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"{metric:<30} | {score:<15.4f} | {threshold:<12.4f} | {status:<8}")

    print("-" * 75)
    print(f"{'NDCG@10':<30} | {metrics.get('ndcg_10', 0.0):<15.4f} | {'N/A':<12} | INFO")
    print(f"{'Recall@5':<30} | {metrics.get('recall_at_5', 0.0):<15.4f} | {'N/A':<12} | INFO")
    print(f"{'Recall@10':<30} | {metrics.get('recall_at_10', 0.0):<15.4f} | {'N/A':<12} | INFO")
    print(f"{'Recall@20':<30} | {metrics.get('recall_at_20', 0.0):<15.4f} | {'N/A':<12} | INFO")
    print(f"{'Total Test Samples':<30} | {metrics.get('total_eval_samples', 0):<15} | {'N/A':<12} | INFO")
    print("=" * 75)

    if passed:
        print(">>> OVERALL STATUS: [PASSED] - All production quality gates satisfied.")
    else:
        print(">>> OVERALL STATUS: [FAILED] - Regression detected on the following metrics:")
        for failure in failures:
            print(f"    * {failure}")
    print("=" * 75 + "\n")


def main():
    args = parse_args()

    # Ensure golden set exists or seed it
    if not os.path.exists(args.golden_set):
        logger.info("Golden set file %s not found. Seeding golden dataset...", args.golden_set)
        from scripts.seed_golden_set import main as seed_main
        seed_main()

    logger.info("Executing evaluation pipeline against %s...", args.golden_set)
    metrics = run_ragas_evaluation(
        golden_set_path=args.golden_set,
        log_to_mlflow=args.log_to_mlflow,
    )

    passed, failures = check_regression(metrics)

    # Save to eval_results.json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "passed": passed,
                "failures": failures,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved evaluation results to %s", args.output)

    print_summary_table(metrics, passed, failures)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
