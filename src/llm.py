"""Minimal client for any OpenAI-compatible chat API.

Works with OpenAI, Mistral, Groq, or a local model served by Ollama, by setting
three environment variables:

    LLM_BASE_URL   e.g. https://api.groq.com/openai/v1  or  http://localhost:11434/v1 (Ollama)
    LLM_API_KEY    your key (any text for Ollama)
    LLM_MODEL      e.g. llama-3.3-70b-versatile, mistral-small-latest, qwen2.5-coder:7b
"""
import os

import requests


class LLMError(Exception):
    pass


def chat(messages: list[dict], temperature: float = 0.0, timeout: int = 60) -> str:
    base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "ollama")
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise LLMError("Set LLM_MODEL (and LLM_BASE_URL / LLM_API_KEY) before calling the LLM.")
    r = requests.post(f"{base}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": messages, "temperature": temperature},
                      timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"LLM API returned {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]
