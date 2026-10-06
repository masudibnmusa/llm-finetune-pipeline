"""Flag when live queries differ from the training distribution.
Checks: (1) input-length distribution via KS test, (2) out-of-vocabulary rate."""
import argparse
import json
import re

from scipy.stats import ks_2samp


def _words(text: str):
    return re.findall(r"\w+", text.lower())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def detect(train_path, usage_path, p_threshold=0.01, oov_threshold=0.25):
    train = load_jsonl(train_path)
    usage = load_jsonl(usage_path)
    if len(usage) < 30:
        print(f"Only {len(usage)} logged queries; need >= 30 for a meaningful check.")
        return

    train_texts = [t.get("input", "") for t in train]
    live_texts = [u["query"] for u in usage]

    # 1) Length distribution
    stat, p = ks_2samp([len(_words(t)) for t in train_texts],
                       [len(_words(t)) for t in live_texts])

    # 2) OOV rate
    vocab = {w for t in train_texts for w in _words(t)}
    live_words = [w for t in live_texts for w in _words(t)]
    oov = sum(w not in vocab for w in live_words) / max(len(live_words), 1)

    print(f"Length KS statistic={stat:.3f}, p-value={p:.4f}")
    print(f"Out-of-vocabulary rate={oov:.2%}")

    if p < p_threshold or oov > oov_threshold:
        print("DRIFT WARNING: live data differs from training data. "
              "Review logs/usage.jsonl and consider a new fine-tuning round.")
    else:
        print("No significant drift detected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/train.jsonl")
    p.add_argument("--usage", default="logs/usage.jsonl")
    a = p.parse_args()
    detect(a.train, a.usage)