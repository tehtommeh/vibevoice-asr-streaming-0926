"""OpenRouter proxy for the voice-editing pass.

The browser never calls OpenRouter directly: routing through the backend means
the eventual desktop client hits the same endpoint, and the API key can live in
the server's environment instead of in browser storage.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODELS_TTL_SECONDS = 900

DEFAULT_SYSTEM_PROMPT = (
    "You are a text editor driven by voice. You are given a passage and a spoken "
    "instruction describing how to change it.\n\n"
    "Rules:\n"
    "- Return ONLY the edited passage. No preamble, no explanation, no quotes "
    "around it, no code fences.\n"
    "- Apply exactly what the instruction asks, and nothing more. Leave wording "
    "you were not asked to change alone.\n"
    "- Preserve the author's voice, register and formatting conventions unless "
    "the instruction asks you to change them.\n"
    "- The instruction reached you through speech recognition and may contain "
    "mis-heard words. Read it charitably: prefer the interpretation that makes "
    "sense for the passage.\n"
    "- If the instruction names a term that appears garbled in the passage, fix "
    "the passage to use the term as the instruction gives it.\n"
    "- If the instruction is empty or you cannot tell what is being asked, "
    "return the passage unchanged."
)

# Sensible starting points surfaced at the top of the picker.
SUGGESTED_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
]

DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

_models_cache: dict[str, Any] = {"at": 0.0, "data": []}


class LLMError(RuntimeError):
    """Carries an HTTP status so the route can pass it through unchanged."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def server_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return key or None


def _request(url: str, *, payload: Optional[dict] = None, api_key: Optional[str] = None,
             timeout: int = 120) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # OpenRouter attributes traffic with these; they are cosmetic.
        headers["HTTP-Referer"] = "http://localhost:8080"
        headers["X-Title"] = "VibeVoice Voice Editing"

    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error", {}).get("message") or body
        except Exception:
            detail = body
        # 401/402/429 are the user's problem to fix, so keep the real status.
        status = exc.code if exc.code in (401, 402, 404, 429) else 502
        raise LLMError(f"OpenRouter: {detail.strip()[:400]}", status) from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach OpenRouter: {exc.reason}", 504) from exc


def list_models(force: bool = False) -> list[dict]:
    """Model catalogue, cached -- it is a 400+ entry payload that rarely moves."""
    now = time.time()
    if not force and _models_cache["data"] and now - _models_cache["at"] < MODELS_TTL_SECONDS:
        return _models_cache["data"]

    payload = _request(f"{OPENROUTER_BASE}/models", timeout=30)
    models = []
    for item in payload.get("data", []):
        modality = (item.get("architecture") or {}).get("output_modalities") or ["text"]
        if "text" not in modality:
            continue
        pricing = item.get("pricing") or {}
        models.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "context_length": item.get("context_length"),
                "prompt_price": _as_float(pricing.get("prompt")),
                "completion_price": _as_float(pricing.get("completion")),
                "suggested": item.get("id") in SUGGESTED_MODELS,
            }
        )
    models.sort(key=lambda m: (not m["suggested"], m["id"] or ""))
    _models_cache.update(at=now, data=models)
    return models


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def edit_text(
    text: str,
    instruction: str,
    *,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    allow_empty_instruction: bool = False,
) -> dict:
    """Apply a spoken instruction to a passage and return the rewritten text.

    Dictation cleanup has no instruction -- the whole task is in the system
    prompt -- so that case opts out of the empty-instruction guard.
    """
    key = (api_key or "").strip() or server_key()
    if not key:
        raise LLMError(
            "No OpenRouter API key. Add one in the page's settings, or set "
            "OPENROUTER_API_KEY in the backend environment.",
            401,
        )
    if not instruction.strip() and not allow_empty_instruction:
        raise LLMError("No instruction was transcribed -- try recording again.", 400)
    if allow_empty_instruction and not text.strip():
        raise LLMError("Nothing was dictated.", 400)

    model = model or DEFAULT_MODEL
    body = {
        "model": model,
        "temperature": max(0.0, min(2.0, float(temperature))),
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<passage>\n{text}\n</passage>"
                    + (f"\n\n<instruction>\n{instruction}\n</instruction>" if instruction.strip() else "")
                ),
            },
        ],
    }

    started = time.perf_counter()
    payload = _request(f"{OPENROUTER_BASE}/chat/completions", payload=body, api_key=key)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("The model returned no choices.", 502)
    result = (choices[0].get("message") or {}).get("content") or ""

    return {
        "result": _strip_fences(result.strip()),
        "model": payload.get("model", model),
        "usage": payload.get("usage") or {},
        "elapsed_ms": round(elapsed_ms, 1),
        "finish_reason": choices[0].get("finish_reason"),
    }


def _strip_fences(text: str) -> str:
    """Some models wrap the answer in a code fence despite being told not to."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
