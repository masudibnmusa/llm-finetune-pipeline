"""Stratified (by domain) train/val/test split."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_records(records, train_frac=0.8, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r.get("domain", "unknown")].append(r)

    train, val, test = [], [], []
    for items in by_domain.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    for subset in (train, val, test):
        rng.shuffle(subset)
    return train, val, test


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(inp, out_dir, train_frac, val_frac, seed):
    with open(inp, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    train, val, test = split_records(records, train_frac, val_frac, seed)
    write_jsonl(f"{out_dir}/train.jsonl", train)
    write_jsonl(f"{out_dir}/val.jsonl", val)
    write_jsonl(f"{out_dir}/test.jsonl", test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/formatted.jsonl")
    p.add_argument("--out_dir", default="data/splits")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.input, a.out_dir, a.train_frac, a.val_frac, a.seed)