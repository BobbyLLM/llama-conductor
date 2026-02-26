# model_calls.py
"""Model API call functions."""

from typing import Any, Dict, List

import requests

from .config import LLAMA_SWAP_URL, ROLES, MODEL_DEBUG, MODEL_DEBUG_PAYLOAD
from .privacy_utils import safe_preview, short_hash


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

    if MODEL_DEBUG and debug_context:
        print(f"\n{'='*70}")
        print(f"[MODEL DEBUG] {debug_context}")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Role: {role}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Prompt hash: {short_hash(prompt)}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
        if MODEL_DEBUG_PAYLOAD:
            red = safe_preview(prompt, max_len=1200)
            print(f"\nPrompt preview (PII-redacted):")
            print(f"{'-'*70}")
            print(red)
            print(f"{'-'*70}\n")

    try:
        resp = requests.post(LLAMA_SWAP_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json() or {}
        
        if MODEL_DEBUG and debug_context:
            print(f"\n[MODEL DEBUG] Response metadata:")
            print(f"{'='*70}")
            choices = data.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason", "unknown")
                print(f"Finish reason: {finish_reason}")
            
            usage = data.get("usage", {})
            if usage:
                print(f"Tokens - prompt: {usage.get('prompt_tokens', '?')}, "
                      f"completion: {usage.get('completion_tokens', '?')}, "
                      f"total: {usage.get('total_tokens', '?')}")
            
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                print(f"Response length: {len(content)} chars")
                print(f"Response hash: {short_hash(content)}")
                if MODEL_DEBUG_PAYLOAD:
                    print(f"\nResponse preview (PII-redacted):")
                    print(f"{'-'*70}")
                    print(safe_preview(content, max_len=1200))
                    print(f"{'-'*70}\n")
        
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
        if MODEL_DEBUG and debug_context:
            print(f"\n[MODEL DEBUG] ERROR: {e}\n")
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
        resp = requests.post(LLAMA_SWAP_URL, json=payload, timeout=240)
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
