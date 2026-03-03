# model_calls.py
"""Model API call functions."""

from typing import Any, Dict, List

import requests

from .config import UPSTREAM_CHAT_URL, ROLES


def resolve_model(role: str) -> str:
    """Resolve role name to model name from config."""
    return str(ROLES.get(role, "")).strip()


def call_model_prompt(
    *,
    role: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    debug_context: str = "",
) -> str:
    """Call model with a simple prompt string."""
    model_name = resolve_model(role)
    if not model_name:
        return f"[router error: no model configured for role '{role}']"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        resp = requests.post(UPSTREAM_CHAT_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json() or {}

        choices = data.get("choices", []) or []
        if not choices:
            return "[router error: no choices from model]"
        msg = choices[0].get("message", {}) or {}
        content = str(msg.get("content", "") or "").strip()
        if not content:
            # Safety: never emit empty assistant content to callers/UI.
            return "Noted."
        return content
    except Exception as e:
        return f"[model '{model_name}' unavailable: {e}]"


def call_model_messages(
    *,
    role: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Call model with a message array."""
    model_name = resolve_model(role)
    if not model_name:
        return f"[router error: no model configured for role '{role}']"

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        resp = requests.post(UPSTREAM_CHAT_URL, json=payload, timeout=240)
        resp.raise_for_status()
        data = resp.json() or {}
        choices = data.get("choices", []) or []
        if not choices:
            return "[router error: no choices from model]"
        msg = choices[0].get("message", {}) or {}
        content = str(msg.get("content", "") or "").strip()
        if not content:
            return "Noted."
        return content
    except Exception as e:
        return f"[model '{model_name}' unavailable: {e}]"
