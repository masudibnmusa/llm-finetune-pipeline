"""Optional custom collator: pads batches and masks padding in the labels.
SFTTrainer already handles this by default, so use this only if you move to a
custom Trainer loop. Example: Trainer(..., data_collator=get_collator(tokenizer))"""
import torch


class CausalLMCollator:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def get_collator(tokenizer, max_length: int = 2048):
    return CausalLMCollator(tokenizer, max_length)