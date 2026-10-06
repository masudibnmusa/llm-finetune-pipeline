"""Exact de-duplication on normalized content."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SKIP_KEYS = {"domain", "source"}


def _key(rec: dict) -> str:
    content = " ".join(
        str(v) for k, v in sorted(rec.items()) if k not in SKIP_KEYS
    )
    content = re.sub(r"\s+", " ", content.lower()).strip()
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def deduplicate(records):
    seen, out = set(), []
    for rec in records:
        k = _key(rec)
        if k not in seen:
            seen.add(k)
            out.append(rec)
    return out


def main(inp: str, out: str):
    with open(inp, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    unique = deduplicate(records)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} -> {len(unique)} records after dedup")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/scrubbed.jsonl")
    p.add_argument("--out", default="data/processed/deduped.jsonl")
    a = p.parse_args()
    main(a.input, a.out)