"""Run the test set through the base model and the fine-tuned model.
Saves everything to reports/results.jsonl."""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_pipeline.formatting.chat_template import build_prompt, build_user_message  # noqa: E402


def load_models(base: str, adapter: str, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = None
    if load_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=300, batch_size=4):
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=2048, add_special_tokens=False,
        ).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs += [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outputs


def main(args):
    with open(args.test, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f][: args.limit]

    model, tok = load_models(args.base, args.adapter, not args.no_4bit)
    prompts = [build_prompt(ex, tok) for ex in examples]

    print("Generating with fine-tuned model...")
    ft_out = generate(model, tok, prompts, args.max_new_tokens, args.batch_size)

    print("Generating with base model (adapter disabled)...")
    with model.disable_adapter():
        base_out = generate(model, tok, prompts, args.max_new_tokens, args.batch_size)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex, b, ft in zip(examples, base_out, ft_out):
            f.write(json.dumps({
                "domain": ex.get("domain"),
                "instruction": ex["instruction"],
                "prompt_text": build_user_message(ex),
                "reference": ex["output"],
                "base_output": b,
                "finetuned_output": ft,
            }, ensure_ascii=False) + "\n")
    print(f"Saved results -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", default="data/checkpoints/final")
    p.add_argument("--test", default="data/splits/test.jsonl")
    p.add_argument("--out", default="reports/results.jsonl")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--no_4bit", action="store_true")
    main(p.parse_args())