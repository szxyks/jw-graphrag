"""Ollama client — wraps the local Ollama HTTP API.

Provides:
  - chat(messages, model) -> stream of tokens
  - embed(text, model) -> 768-dim vector
  - list_models() -> available models
  - ensure_models() -> pulls required models if missing
"""
import os
import json
import requests
from typing import Generator, List, Dict, Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
DEFAULT_LLM = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b")
DEFAULT_EMBED = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def list_models() -> List[str]:
    """Return list of locally-available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        return [f"(error: {e})"]


def embed(text: str, model: str = None) -> List[float]:
    """Generate embedding for a single text string."""
    model = model or DEFAULT_EMBED
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def embed_batch(texts: List[str], model: str = None) -> List[List[float]]:
    """Generate embeddings for multiple texts (sequential — Ollama is single-thread per request)."""
    return [embed(t, model) for t in texts]


def chat_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Stream chat completion tokens from Ollama.

    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Yields: tokens (strings) as they arrive.
    """
    model = model or DEFAULT_LLM
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    with requests.post(
        f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if obj.get("message", {}).get("content"):
                    yield obj["message"]["content"]
                if obj.get("done"):
                    break
            except json.JSONDecodeError:
                continue


def chat(messages: List[Dict[str, str]], model: str = None) -> str:
    """Non-streaming chat completion."""
    return "".join(chat_stream(messages, model=model))


def is_ready() -> bool:
    """Check if Ollama is up and has at least one model."""
    try:
        models = list_models()
        return len(models) > 0 and not models[0].startswith("(error")
    except Exception:
        return False
