"""Perplexity of base vs fine-tuned model on test text."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, max_length: int = 2048) -> float:
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        n = enc["input_ids"].shape[1] - 1
        if n < 1:
            continue
        out = model(**enc, labels=enc["input_ids"])
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(total_loss / max(total_tokens, 1))


if __name__ == "__main__":
    from datasets import load_dataset
    from data_pipeline.formatting.chat_template import apply_chat_template
    from evaluation.benchmark_runner import load_models

    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default="data/checkpoints/final")
    p.add_argument("--test", default="data/splits/test.jsonl")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()

    model, tok = load_models(a.base, a.adapter, load_4bit=True)
    ds = load_dataset("json", data_files=a.test, split="train").select(range(a.limit))
    texts = [apply_chat_template(ex, tok)["text"] for ex in ds]

    ft = compute_perplexity(model, tok, texts)
    with model.disable_adapter():
        base = compute_perplexity(model, tok, texts)
    print(json.dumps({"base_perplexity": round(base, 3), "finetuned_perplexity": round(ft, 3)}, indent=2))