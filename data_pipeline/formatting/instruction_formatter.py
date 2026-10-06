"""Convert cleaned raw records into instruction/input/output format."""
import argparse
import json
from pathlib import Path

DOMAIN_SPECS = {
    "legal": {
        "instruction": "Summarize the following legal text.",
        "input": "text",
        "output": "summary",
    },
    "medical": {
        "instruction": "Answer the following medical question accurately and concisely.",
        "input": "question",
        "output": "answer",
    },
    "support": {
        "instruction": "Write a helpful, polite support reply to the customer message.",
        "input": "ticket",
        "output": "resolution",
    },
}


def format_record(rec: dict):
    spec = DOMAIN_SPECS.get(rec.get("domain"))
    if not spec:
        return None
    return {
        "domain": rec["domain"],
        "instruction": spec["instruction"],
        "input": rec[spec["input"]].strip(),
        "output": rec[spec["output"]].strip(),
    }


def main(inp: str, out: str):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(inp, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            formatted = format_record(json.loads(line))
            if formatted:
                fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                n += 1
    print(f"Formatted {n} examples -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/filtered.jsonl")
    p.add_argument("--out", default="data/processed/formatted.jsonl")
    a = p.parse_args()
    main(a.input, a.out)