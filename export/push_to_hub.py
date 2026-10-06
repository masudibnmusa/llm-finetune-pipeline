"""Upload a model folder (merged model or LoRA adapter) to the Hugging Face Hub."""
import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def push(folder: str, repo_id: str, private: bool = True):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id)
    print(f"Uploaded {folder} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="merged_model")
    p.add_argument("--repo_id", required=True, help="e.g. username/my-domain-llm")
    p.add_argument("--public", action="store_true")
    a = p.parse_args()
    push(a.folder, a.repo_id, private=not a.public)