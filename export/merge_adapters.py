"""Merge LoRA adapter weights into the base model (full precision)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: str, adapter: str, out: str):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default="data/checkpoints/final")
    p.add_argument("--out", default="merged_model")
    a = p.parse_args()
    merge(a.base, a.adapter, a.out)