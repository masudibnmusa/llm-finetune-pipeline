"""REST API in front of the vLLM OpenAI-compatible server.
Run: uvicorn serving.api:app --port 8080"""
import os
import time

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from monitoring.usage_logger import log_interaction
from serving.prompt_templates import DOMAIN_INSTRUCTIONS, build_user_prompt

load_dotenv()
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "domain-llm")

app = FastAPI(title="Domain LLM API")


class ChatRequest(BaseModel):
    domain: str = "support"
    message: str
    max_tokens: int = 512
    temperature: float = 0.2


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

    answer = r.json()["choices"][0]["message"]["content"].strip()
    latency = int((time.time() - start) * 1000)
    log_interaction(req.domain, req.message, answer, latency)
    return ChatResponse(response=answer, latency_ms=latency)