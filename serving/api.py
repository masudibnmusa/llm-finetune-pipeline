"""REST API in front of the vLLM OpenAI-compatible server.
Run: uvicorn serving.api:app --port 8080"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from monitoring.usage_logger import log_interaction  # noqa: E402
from serving.prompt_templates import DOMAIN_INSTRUCTIONS, build_user_prompt  # noqa: E402

load_dotenv()
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "domain-llm")

app = FastAPI(title="Domain LLM API")


class ChatRequest(BaseModel):
    domain: str = "support"
    message: str = Field(min_length=1)
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    response: str
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok", "domains": list(DOMAIN_INSTRUCTIONS)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        prompt = build_user_prompt(req.domain, req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    start = time.time()
    try:
        r = requests.post(VLLM_URL, json=payload, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Inference server error: {e}")

    try:
        answer = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected inference server response: {e}")
    latency = int((time.time() - start) * 1000)
    log_interaction(req.domain, req.message, answer, latency)
    return ChatResponse(response=answer, latency_ms=latency)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8080")))
