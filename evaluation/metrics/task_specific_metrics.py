"""Task metrics: ROUGE-L, token F1, exact match."""
import re
from collections import Counter

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _tokens(text: str):
    return re.findall(r"\w+", text.lower())


def exact_match(pred: str, ref: str) -> float:
    return float(" ".join(_tokens(pred)) == " ".join(_tokens(ref)))


def token_f1(pred: str, ref: str) -> float:
    p, r = Counter(_tokens(pred)), Counter(_tokens(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    return _scorer.score(ref, pred)["rougeL"].fmeasure


def evaluate_predictions(preds, refs) -> dict:
    n = max(len(preds), 1)
    return {
        "rougeL": sum(rouge_l(p, r) for p, r in zip(preds, refs)) / n,
        "token_f1": sum(token_f1(p, r) for p, r in zip(preds, refs)) / n,
        "exact_match": sum(exact_match(p, r) for p, r in zip(preds, refs)) / n,
    }