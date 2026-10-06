"""Start an inference server for the fine-tuned model.

  python serving/inference_server.py --backend vllm --model merged_model
  python serving/inference_server.py --backend ollama --gguf quantized/model-Q4_K_M.gguf
"""
import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def start_vllm(model: str, port: int, name: str, max_len: int):
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--served-model-name", name,
        "--max-model-len", str(max_len),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def start_ollama(gguf: str, name: str):
    modelfile = Path("Modelfile")
    modelfile.write_text(f"FROM {gguf}\nPARAMETER temperature 0.2\n")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(f"Created Ollama model '{name}'. Run it with: ollama run {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vllm", "ollama"], default="vllm")
    p.add_argument("--model", default="merged_model")
    p.add_argument("--gguf", default="quantized/model-Q4_K_M.gguf")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--name", default=os.getenv("SERVED_MODEL_NAME", "domain-llm"))
    a = p.parse_args()

    if a.backend == "vllm":
        start_vllm(a.model, a.port, a.name, a.max_len)
    else:
        start_ollama(a.gguf, a.name)