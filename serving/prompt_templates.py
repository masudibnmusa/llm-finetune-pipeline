"""Domain instructions used at inference. Keep these identical to training
(see data_pipeline/formatting/instruction_formatter.py) for best results."""

DOMAIN_INSTRUCTIONS = {
    "legal": "Summarize the following legal text.",
    "medical": "Answer the following medical question accurately and concisely.",
    "support": "Write a helpful, polite support reply to the customer message.",
}


def build_user_prompt(domain: str, text: str) -> str:
    if domain not in DOMAIN_INSTRUCTIONS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {list(DOMAIN_INSTRUCTIONS)}")
    return f"{DOMAIN_INSTRUCTIONS[domain]}\n\n{text}"