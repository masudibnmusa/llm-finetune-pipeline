from evaluation.metrics.task_specific_metrics import (
    evaluate_predictions, exact_match, rouge_l, token_f1,
)


def test_exact_match():
    assert exact_match("The cat sat.", "the cat sat") == 1.0
    assert exact_match("a b", "a c") == 0.0


def test_token_f1_partial():
    score = token_f1("the cat sat", "the cat ran")
    assert 0.0 < score < 1.0


def test_rouge_identical():
    assert rouge_l("hello world", "hello world") > 0.99


def test_evaluate_predictions_keys():
    out = evaluate_predictions(["a b c"], ["a b c"])
    assert set(out) == {"rougeL", "token_f1", "exact_match"}
    assert out["exact_match"] == 1.0