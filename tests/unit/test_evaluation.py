from src.evaluation.arabic_metrics import language_consistency_score
from src.evaluation.regression import check_regression
from src.evaluation.retrieval_metrics import compute_mrr, compute_ndcg, compute_recall_at_k


def test_mrr_computation():
    # Query 1: relevant doc at rank 1 (RR = 1.0)
    # Query 2: relevant doc at rank 2 (RR = 0.5)
    # Query 3: not in top-10 (RR = 0.0)
    results = [
        ["docA", "docB", "docC"],
        ["docX", "docB", "docY"],
        ["doc1", "doc2", "doc3"],
    ]
    ground_truth = ["docA", "docB", "docZ"]

    mrr = compute_mrr(results, ground_truth, k=10)
    # Expected: (1.0 + 0.5 + 0.0) / 3 = 0.50
    assert mrr == 0.50


def test_ndcg_computation():
    # Perfect ranking: target at rank 1 -> NDCG = 1.0
    results = [["doc1", "doc2", "doc3"]]
    ground_truth = ["doc1"]
    ndcg = compute_ndcg(results, ground_truth, k=10)
    assert ndcg == 1.0

    # Sub-optimal ranking: target at rank 2 -> DCG = 1/log2(3) = 0.6309; IDCG = 1/log2(2) = 1.0
    results_sub = [["doc2", "doc1", "doc3"]]
    ndcg_sub = compute_ndcg(results_sub, ground_truth, k=10)
    assert 0.60 < ndcg_sub < 0.65


def test_recall_at_k():
    results = [
        ["d1", "d2", "d3", "d4", "d5"],
        ["d10", "d11", "d12", "d13", "d14"],
    ]
    ground_truth = [
        ["d2", "d5"],  # 2 targets both in top 5 -> recall = 1.0
        ["d10", "d99"],  # 1 hit out of 2 in top 5 -> recall = 0.5
    ]
    recalls = compute_recall_at_k(results, ground_truth, k_values=[3, 5])
    # For k=5: (1.0 + 0.5) / 2 = 0.75
    assert recalls[5] == 0.75


def test_language_consistency_score():
    queries = [
        "ما هي عاصمة قطر؟",
        "What is the capital of Saudi Arabia?",
        "كيف يعمل الذكاء الاصطناعي؟",
    ]
    responses = [
        "عاصمة قطر هي الدوحة.",
        "The capital of Saudi Arabia is Riyadh.",
        "عاصمة وفرنسا هي باريس.",
    ]
    score = language_consistency_score(queries, responses)
    assert score == 1.0

    mismatched_responses = [
        "The capital of Qatar is Doha.",  # AR -> EN mismatch
        "The capital of Saudi Arabia is Riyadh.",
        "عاصمة وفرنسا هي باريس.",
    ]
    score_mismatch = language_consistency_score(queries, mismatched_responses)
    assert score_mismatch == round(2 / 3, 4)


def test_regression_check_passes():
    scores = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
        "mrr_10": 0.65,
        "language_consistency": 0.95,
    }
    passed, failures = check_regression(scores)
    assert passed is True
    assert len(failures) == 0


def test_regression_check_fails():
    scores = {
        "faithfulness": 0.70,  # Below 0.80
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
        "mrr_10": 0.55,  # Below 0.60
        "language_consistency": 0.95,
    }
    passed, failures = check_regression(scores)
    assert passed is False
    assert len(failures) == 2
    assert any("faithfulness" in f for f in failures)
    assert any("mrr_10" in f for f in failures)


def test_fixture_only_evaluation_never_fabricates_retrieval_accuracy(tmp_path, monkeypatch):
    import json
    from src.evaluation.ragas_eval import run_ragas_evaluation

    monkeypatch.setenv("RAGAS_EVALUATOR_ENABLED", "0")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps([{
        "question": "What is the test objective?",
        "ground_truth": "Document-grounded answers.",
        "answer": "Document-grounded answers.",
        "contexts": ["This system aims for document-grounded answers."],
        "expected_doc_id": "doc-a",
        "language": "en",
    }]), encoding="utf-8")

    results = run_ragas_evaluation(str(fixture), log_to_mlflow=False)
    assert results["retrieval_source"] == "not_evaluated"
    assert results["answer_source"] == "golden_set_fixture"
    assert results["evaluator_type"] == "heuristic_proxy"
    assert results["mrr_10"] is None
    assert results["ndcg_10"] is None
    assert results["recall_at_5"] is None
    assert results["benchmarks_verified"] is False

    passed, failures = check_regression(results)
    assert passed is False
    assert any("mrr_10" in error for error in failures)


def test_injected_retriever_uses_real_rank_positions(tmp_path, monkeypatch):
    import json
    from src.evaluation.ragas_eval import run_ragas_evaluation

    monkeypatch.setenv("RAGAS_EVALUATOR_ENABLED", "0")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps([{
        "question": "Where is doc A?",
        "ground_truth": "In a local fixture.",
        "contexts": ["In a local fixture."],
        "expected_doc_id": "doc-a",
        "language": "en",
    }]), encoding="utf-8")

    def actual_retriever(_question):
        return ["not-relevant", "doc-a"]

    result = run_ragas_evaluation(
        str(fixture), retriever=actual_retriever, log_to_mlflow=False
    )
    assert result["retrieval_source"] == "live_retriever"
    assert result["mrr_10"] == 0.5
    assert result["ndcg_10"] > 0.6
    assert result["recall_at_5"] == 1.0
    assert result["benchmarks_verified"] is False
