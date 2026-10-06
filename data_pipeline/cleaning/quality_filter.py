"""Drop empty, too short, too long, or mostly non-text examples."""
import argparse
import json
from pathlib import Path

SKIP_KEYS = {"domain", "source"}


def is_good(rec: dict, min_chars: int = 30, max_chars: int = 8000) -> bool:
    values = [v for k, v in rec.items() if k not in SKIP_KEYS and isinstance(v, str)]
    if not values or any(not v.strip() for v in values):
        return False
    total = sum(len(v) for v in values)
    if not (min_chars <= total <= max_chars):
        return False
    text = " ".join(values)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio >= 0.5


def main(inp: str, out: str, min_chars: int, max_chars: int):
    with open(inp, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    kept = [r for r in records if is_good(r, min_chars, max_chars)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Kept {len(kept)}/{len(records)} records")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/deduped.jsonl")
    p.add_argument("--out", default="data/processed/filtered.jsonl")
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--max_chars", type=int, default=8000)
    a = p.parse_args()
    main(a.input, a.out, a.min_chars, a.max_chars)