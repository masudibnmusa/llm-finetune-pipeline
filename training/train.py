"""LoRA / QLoRA fine-tuning with TRL's SFTTrainer.
Usage: python training/train.py   (or: accelerate launch training/train.py)"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_pipeline.formatting.chat_template import apply_chat_template  # noqa: E402
from training.callbacks import get_callbacks  # noqa: E402

load_dotenv()


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg_dir: str):
    model_cfg = load_yaml(f"{cfg_dir}/model_config.yaml")
    lora_cfg = load_yaml(f"{cfg_dir}/lora_config.yaml")
    train_cfg = load_yaml(f"{cfg_dir}/training_args.yaml")

    base = model_cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    data_cfg = train_cfg["data"]
    train_files = [data_cfg["train_file"]] + data_cfg.get("extra_train_files", [])
    train_parts = [load_dataset("json", data_files=f, split="train") for f in train_files]
    train_ds = concatenate_datasets(train_parts)
    val_ds = load_dataset("json", data_files=data_cfg["val_file"], split="train")

    fmt = lambda ex: apply_chat_template(ex, tokenizer)  # noqa: E731
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)

    # Model (QLoRA)
    q = model_cfg.get("quantization", {})
    bnb_config = None
    if q.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(**lora_cfg)

    sft_args = SFTConfig(
        dataset_text_field="text",
        max_length=model_cfg.get("max_seq_length", 2048),
        **train_cfg["args"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=get_callbacks(train_cfg.get("early_stopping_patience", 3)),
    )

    trainer.train()

    final_dir = f"{train_cfg['args']['output_dir']}/final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_dir", default="training/config")
    main(p.parse_args().config_dir)