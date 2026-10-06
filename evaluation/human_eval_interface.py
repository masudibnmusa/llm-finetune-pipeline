"""Blind A/B human evaluation UI (Gradio). Saves votes to reports/human_eval.jsonl."""
import argparse
import json
import random

import gradio as gr


def build_ui(rows, out_path):
    rng = random.Random(0)
    ft_first = [rng.random() < 0.5 for _ in rows]  # True -> A is fine-tuned

    def show(i):
        if i >= len(rows):
            return "All done. Thank you!", "", "", i
        r = rows[i]
        if ft_first[i]:
            a, b = r["finetuned_output"], r["base_output"]
        else:
            a, b = r["base_output"], r["finetuned_output"]
        return r["prompt_text"], a, b, i

    def vote(choice, i):
        if i < len(rows) and choice:
            if choice == "Tie":
                winner = "tie"
            else:
                winner = "finetuned" if (choice == "A") == ft_first[i] else "base"
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"index": i, "winner": winner}) + "\n")
        return show(i + 1)

    with gr.Blocks(title="Human Evaluation") as demo:
        gr.Markdown("## Which response is better? (blind A/B)")
        idx = gr.State(0)
        prompt = gr.Textbox(label="Prompt", lines=6, interactive=False)
        with gr.Row():
            a_box = gr.Textbox(label="Response A", lines=10, interactive=False)
            b_box = gr.Textbox(label="Response B", lines=10, interactive=False)
        choice = gr.Radio(["A", "B", "Tie"], label="Better response")
        submit = gr.Button("Submit")

        demo.load(lambda: show(0), outputs=[prompt, a_box, b_box, idx])
        submit.click(vote, [choice, idx], [prompt, a_box, b_box, idx])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="reports/results.jsonl")
    p.add_argument("--out", default="reports/human_eval.jsonl")
    p.add_argument("--limit", type=int, default=50)
    a = p.parse_args()
    with open(a.results, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][: a.limit]
    build_ui(rows, a.out).launch()