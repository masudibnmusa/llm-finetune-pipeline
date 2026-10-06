"""Side-by-side report: fine-tuned vs base. Reads reports/results.jsonl."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation.metrics.task_specific_metrics import evaluate_predictions  # noqa: E402


def main(results_path, out_path, use_judge, judge_limit):
    with open(results_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r.get("domain", "all")].append(r)
    by_domain["ALL"] = rows

    lines = ["# Fine-tuned vs Base Model\n"]
    for domain, items in by_domain.items():
        refs = [r["reference"] for r in items]
        base = evaluate_predictions([r["base_output"] for r in items], refs)
        ft = evaluate_predictions([r["finetuned_output"] for r in items], refs)

        lines.append(f"\n## {domain} (n={len(items)})\n")
        lines.append("| Metric | Base | Fine-tuned | Delta |")
        lines.append("|---|---|---|---|")
        for m in base:
            lines.append(f"| {m} | {base[m]:.3f} | {ft[m]:.3f} | {ft[m] - base[m]:+.3f} |")

        if use_judge and domain == "ALL":
            from evaluation.metrics.llm_judge import mean_judge_score
            sub = items[:judge_limit]
            tasks = [r["prompt_text"] for r in sub]
            b = mean_judge_score(tasks, [r["reference"] for r in sub], [r["base_output"] for r in sub])
            f_ = mean_judge_score(tasks, [r["reference"] for r in sub], [r["finetuned_output"] for r in sub])
            lines.append(f"| llm_judge (1-5) | {b:.2f} | {f_:.2f} | {f_ - b:+.2f} |")

    report = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="reports/results.jsonl")
    p.add_argument("--out", default="reports/comparison_report.md")
    p.add_argument("--judge", action="store_true", help="Add LLM-as-judge scores")
    p.add_argument("--judge_limit", type=int, default=50)
    a = p.parse_args()
    main(a.results, a.out, a.judge, a.judge_limit)