# model_calls.py
"""Model API call functions."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from pathlib import Path
from datetime import datetime, timezone
from .config import UPSTREAM_CHAT_URL, ROLES, ROUTER_DEBUG, ROUTER_DEBUG_LOG_USER_TEXT
from .privacy_utils import safe_preview, short_hash


def resolve_model(role: str) -> str:
    """Resolve role name to model name from config."""
    return str(ROLES.get(role, "")).strip()


def _extract_finish_reason(data: Dict[str, Any]) -> str:
    choices = data.get("choices", []) or []
    if not choices:
        return "unknown"
    return str(choices[0].get("finish_reason", "unknown") or "unknown")


def _append_raw_model_trace(line: str) -> None:
    try:
        log_path = Path(__file__).resolve().parents[1] / "logs" / "model_raw.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {line}\n")
    except Exception:
        pass


def call_model_prompt(
    *,
    role: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    debug_context: str = "",
) -> ModelCallResult:
    """Call model with a simple prompt string."""
    model_name = resolve_model(role)
    if not model_name:
        return ModelCallResult(f"[router error: no model configured for role '{role}']", "error")

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    if ROUTER_DEBUG and debug_context:
        print(f"\n{'='*70}")
        print(f"[ROUTER DEBUG] {debug_context}")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Role: {role}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Prompt hash: {short_hash(prompt)}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top-p: {top_p}")
        if ROUTER_DEBUG_LOG_USER_TEXT:
            red = safe_preview(prompt, max_len=1200)
            print(f"\nPrompt preview (PII-redacted):")
            print(f"{'-'*70}")
            print(red)
            print(f"{'-'*70}\n")

    try:
        resp = requests.post(UPSTREAM_CHAT_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json() or {}
        usage = data.get("usage", {}) or {}
        finish_reason = _extract_finish_reason(data)
        raw_line = (
            "[model raw] "
            f"requested_max_tokens={max_tokens} "
            f"finish_reason={finish_reason} "
            f"prompt_tokens={usage.get('prompt_tokens', '?')} "
            f"completion_tokens={usage.get('completion_tokens', '?')} "
            f"total_tokens={usage.get('total_tokens', '?')}"
        )
        print(raw_line, flush=True)
        _append_raw_model_trace(raw_line)
        
        if ROUTER_DEBUG and debug_context:
            print(f"\n[ROUTER DEBUG] Response metadata:")
            print(f"{'='*70}")
            print(f"Finish reason: {finish_reason}")
            
            if usage:
                print(f"Tokens - prompt: {usage.get('prompt_tokens', '?')}, "
                      f"completion: {usage.get('completion_tokens', '?')}, "
                      f"total: {usage.get('total_tokens', '?')}")
            
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                print(f"Response length: {len(content)} chars")
                print(f"Response hash: {short_hash(content)}")
                if ROUTER_DEBUG_LOG_USER_TEXT:
                    print(f"\nResponse preview (PII-redacted):")
                    print(f"{'-'*70}")
                    print(safe_preview(content, max_len=1200))
                    print(f"{'-'*70}\n")
        
        choices = data.get("choices", []) or []
        if not choices:
            return ModelCallResult("[router error: no choices from model]", finish_reason)
        msg = choices[0].get("message", {}) or {}
        content = str(msg.get("content", "") or "").strip()
        if not content:
            # Safety: never emit empty assistant content to callers/UI.
            return ModelCallResult("Noted.", finish_reason)
        return ModelCallResult(content, finish_reason)
    except Exception as e:
        if ROUTER_DEBUG and debug_context:
            print(f"\n[ROUTER DEBUG] ERROR: {e}\n")
        return ModelCallResult(f"[model '{model_name}' unavailable: {e}]", "error")


def call_model_messages(
    *,
    role: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> ModelCallResult:
    """Call model with a message array."""
    model_name = resolve_model(role)
    if not model_name:
        return ModelCallResult(f"[router error: no model configured for role '{role}']", "error")

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
            return ModelCallResult("[router error: no choices from model]", _extract_finish_reason(data))
        msg = choices[0].get("message", {}) or {}
        content = str(msg.get("content", "") or "").strip()
        if not content:
            return ModelCallResult("Noted.", _extract_finish_reason(data))
        return ModelCallResult(content, _extract_finish_reason(data))
    except Exception as e:
        return ModelCallResult(f"[model '{model_name}' unavailable: {e}]", "error")
@dataclass
class ModelCallResult:
    text: str
    finish_reason: str = "stop"

    def __bool__(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"ModelCallResult(text={self.text!r}, finish_reason={self.finish_reason!r})"

    def __iter__(self):
        yield self.text
        yield self.finish_reason

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item]

    def __getattr__(self, name: str):
        return getattr(self.text, name)

