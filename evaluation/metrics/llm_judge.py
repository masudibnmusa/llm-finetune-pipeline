"""LLM-as-judge: scores a response 1-5 against a reference."""
import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")

RUBRIC = """You are a strict evaluator. Score the RESPONSE to the TASK from 1 to 5
considering correctness vs the REFERENCE, completeness, domain terminology, and tone.
5 = excellent, 3 = acceptable, 1 = wrong or unusable.

TASK:
{task}

REFERENCE:
{reference}

RESPONSE:
{response}

Return ONLY JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def judge(task: str, reference: str, response: str) -> dict:
    msg = _get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": RUBRIC.format(
            task=task, reference=reference, response=response)}],
    )
    text = msg.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"score": None, "reason": "parse_error"}


def mean_judge_score(tasks, references, responses) -> float:
    scores = []
    for t, ref, resp in zip(tasks, references, responses):
        s = judge(t, ref, resp).get("score")
        if isinstance(s, (int, float)):
            scores.append(s)
    return sum(scores) / max(len(scores), 1)