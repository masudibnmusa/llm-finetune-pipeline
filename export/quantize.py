"""Quantize a merged model to GGUF (llama.cpp), AWQ, or GPTQ.

GGUF needs a local llama.cpp checkout and build:
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
"""
import argparse
import subprocess
from pathlib import Path


def to_gguf(model_dir: str, out_dir: str, llama_cpp: str, quant: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f16 = f"{out_dir}/model-f16.gguf"
    final = f"{out_dir}/model-{quant}.gguf"
    subprocess.run(
        ["python", f"{llama_cpp}/convert_hf_to_gguf.py", model_dir,
         "--outfile", f16, "--outtype", "f16"],
        check=True,
    )
    subprocess.run(
        [f"{llama_cpp}/build/bin/llama-quantize", f16, final, quant],
        check=True,
    )
    print(f"GGUF saved to {final}")


def to_awq(model_dir: str, out_dir: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    model.quantize(tok, quant_config={
        "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
    model.save_quantized(out_dir)
    tok.save_pretrained(out_dir)
    print(f"AWQ model saved to {out_dir}")


def to_gptq(model_dir: str, out_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = GPTQConfig(bits=4, dataset="c4", tokenizer=tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=cfg, device_map="auto")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"GPTQ model saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["gguf", "awq", "gptq"], required=True)
    p.add_argument("--model", default="merged_model")
    p.add_argument("--out", default="quantized")
    p.add_argument("--llama_cpp", default="llama.cpp")
    p.add_argument("--quant", default="Q4_K_M", help="GGUF quant type")
    a = p.parse_args()

    if a.format == "gguf":
        to_gguf(a.model, a.out, a.llama_cpp, a.quant)
    elif a.format == "awq":
        to_awq(a.model, a.out)
    else:
        to_gptq(a.model, a.out)