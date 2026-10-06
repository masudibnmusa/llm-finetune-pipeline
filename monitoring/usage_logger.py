"""Append real-world queries/responses to logs/usage.jsonl."""
import json
import time
from pathlib import Path

LOG_PATH = Path("logs/usage.jsonl")


def log_interaction(domain: str, query: str, response: str, latency_ms: int,
                    feedback: str | None = None):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "domain": domain,
        "query": query,
        "response": response,
        "latency_ms": latency_ms,
        "feedback": feedback,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")