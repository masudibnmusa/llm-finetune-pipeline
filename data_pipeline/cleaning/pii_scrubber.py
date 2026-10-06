"""Regex-based PII scrubber. For production legal/medical data, also use a NER tool
such as Microsoft Presidio to catch names, addresses, and dates of birth."""
import argparse
import glob
import json
import re
from pathlib import Path

SKIP_KEYS = {"domain", "source"}

PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def scrub_text(text: str) -> str:
    for label, pattern in PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text


def scrub_record(rec: dict) -> dict:
    return {
        k: (scrub_text(v) if isinstance(v, str) and k not in SKIP_KEYS else v)
        for k, v in rec.items()
    }


def main(in_glob: str, out: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fout:
        for path in glob.glob(in_glob):
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    rec = scrub_record(json.loads(line))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Scrubbed {n} records -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/*.jsonl")
    p.add_argument("--out", default="data/processed/scrubbed.jsonl")
    a = p.parse_args()
    main(a.input, a.out)