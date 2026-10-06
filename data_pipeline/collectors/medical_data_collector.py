"""Collect medical Q&A pairs into data/raw/medical.jsonl."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def collect(limit: int = 2000, out: str = "data/raw/medical.jsonl"):
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if n >= limit:
                break
            rec = {
                "domain": "medical",
                "source": "medical_flashcards",
                "question": row["input"],
                "answer": row["output"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} medical records to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", default="data/raw/medical.jsonl")
    a = p.parse_args()
    collect(limit=a.limit, out=a.out)