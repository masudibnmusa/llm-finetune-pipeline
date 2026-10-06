"""Collect legal text + summaries (US bills from BillSum) into data/raw/legal.jsonl."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def collect(limit: int = 2000, max_chars: int = 4000, out: str = "data/raw/legal.jsonl"):
    ds = load_dataset("FiscalNote/billsum", split="train")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if n >= limit:
                break
            rec = {
                "domain": "legal",
                "source": "billsum",
                "text": row["text"][:max_chars],
                "summary": row["summary"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} legal records to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", default="data/raw/legal.jsonl")
    a = p.parse_args()
    collect(limit=a.limit, out=a.out)