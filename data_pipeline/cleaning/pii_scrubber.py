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
    # 13-16 digits, optionally separated by single spaces/dashes (no trailing separator)
    ("CREDIT_CARD", re.compile(r"\b\d(?:[ -]?\d){12,15}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _mask_card(match: re.Match) -> str:
    # Luhn check avoids masking things like lists of years or long reference numbers
    digits = re.sub(r"\D", "", match.group(0))
    return "[CREDIT_CARD]" if _luhn_valid(digits) else match.group(0)


def scrub_text(text: str) -> str:
    for label, pattern in PATTERNS:
        repl = _mask_card if label == "CREDIT_CARD" else f"[{label}]"
        text = pattern.sub(repl, text)
    return text


def scrub_record(rec: dict) -> dict:
    return {
        k: (scrub_text(v) if isinstance(v, str) and k not in SKIP_KEYS else v)
        for k, v in rec.items()
    }


def main(in_glob: str, out: str):
    paths = sorted(glob.glob(in_glob))
    if not paths:
        raise SystemExit(f"No input files match '{in_glob}'. Run a collector first.")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fout:
        for path in paths:
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