"""Optional: generate synthetic examples with an LLM.
Reads from the TRAIN split only (to avoid leaking into val/test) and writes
data/splits/train_synthetic.jsonl. Always review synthetic data before training."""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

PROMPT = """You are helping build a fine-tuning dataset for the "{domain}" domain.
Here are example records (instruction, input, output):

{examples}

Write {n} NEW, diverse, realistic records in the same style and task.
Do not copy the examples. Do not include real personal data.
Return ONLY a JSON list of objects with keys "instruction", "input", "output"."""


def generate(train_path, out_path, per_call, calls, seed):
    client = anthropic.Anthropic()
    rng = random.Random(seed)
    with open(train_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r.get("domain", "unknown")].append(r)
    if not by_domain:
        raise SystemExit(f"No training rows found in {train_path}")
    domains = sorted(by_domain)

    new_rows = []
    for _ in range(calls):
        # Keep all few-shot examples in one domain so the generated rows match their label
        domain = rng.choice(domains)
        pool = by_domain[domain]
        sample = rng.sample(pool, k=min(3, len(pool)))
        examples = json.dumps(
            [{k: s[k] for k in ("instruction", "input", "output")} for s in sample],
            indent=2,
        )
        msg = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": PROMPT.format(
                domain=domain, examples=examples, n=per_call)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            continue
        try:
            for item in json.loads(match.group(0)):
                if isinstance(item, dict) and all(
                        isinstance(item.get(k), str) for k in ("instruction", "input", "output")):
                    item["domain"] = domain
                    new_rows.append(item)
        except json.JSONDecodeError:
            continue

    with open(out_path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(new_rows)} synthetic examples -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train.jsonl")
    p.add_argument("--out", default="data/splits/train_synthetic.jsonl")
    p.add_argument("--per_call", type=int, default=5)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.train, a.out, a.per_call, a.calls, a.seed)