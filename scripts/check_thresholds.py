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

from src.evaluation.regression import THRESHOLDS, check_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_thresholds")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "eval_results.json"

    if not os.path.exists(results_path):
        logger.error("Results file '%s' not found. Run scripts/run_eval.py first.", results_path)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    passed, failures = check_regression(metrics, THRESHOLDS)

    print("\n--- Production Quality Gate Verification ---")
    for metric, threshold in THRESHOLDS.items():
        val = metrics.get(metric, 0.0)
        status = "PASSED" if val >= threshold else "FAILED"
        print(f"[{status}] {metric}: {val:.4f} (Threshold >= {threshold:.4f})")

    if passed:
        print("\nAll quality gates passed successfully. Ready for deployment.\n")
        sys.exit(0)
    else:
        print("\nQuality gate check FAILED with the following regressions:")
        for fail in failures:
            print(f"  - {fail}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
