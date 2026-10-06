"""Apply a model-specific chat template to instruction/input/output examples."""


def build_user_message(ex: dict) -> str:
    if ex.get("input"):
        return f"{ex['instruction']}\n\n{ex['input']}"
    return ex["instruction"]


def to_messages(ex: dict, include_answer: bool = True) -> list:
    messages = [{"role": "user", "content": build_user_message(ex)}]
    if include_answer:
        messages.append({"role": "assistant", "content": ex["output"]})
    return messages


def apply_chat_template(ex: dict, tokenizer) -> dict:
    """For training: returns {'text': full conversation string}."""
    text = tokenizer.apply_chat_template(
        to_messages(ex, include_answer=True), tokenize=False
    )
    return {"text": text}


def build_prompt(ex: dict, tokenizer) -> str:
    """For inference/eval: user turn only, ready for generation."""
    return tokenizer.apply_chat_template(
        to_messages(ex, include_answer=False),
        tokenize=False,
        add_generation_prompt=True,
    )