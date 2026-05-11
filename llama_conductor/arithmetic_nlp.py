"""Deterministic helpers for plain-NL arithmetic intent handling.

This module is intentionally narrow:
- detect simple arithmetic NL intent
- normalize supported math phrasing into a calc-safe expression
- do not evaluate arithmetic itself
"""

from __future__ import annotations

import re
from typing import Optional


_TRANSCRIPT_USER_BLOCK_RE = re.compile(
    r"(?ims)^\s*##\s*User\s*$\n(?P<body>.*?)(?=^\s*##\s*(?:Assistant|System|User)\s*$|\Z)"
)

_FRAMING_RE = re.compile(
    r"^\s*(?:(?:what(?:'s|\s+is)|how much is|calculate|compute|evaluate|solve|work out)\b(?:\s+the\s+result\s+of)?\s*)+",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[?.!,;:]+\s*$")
_NUMBER_RE = re.compile(r"\d")
_ARITHMETIC_SIGNAL_RE = re.compile(r"[+\-*/%^]")
_RANGE_RE = re.compile(r"(?<!\d)\d+\s*-\s*\d+(?!\d)")
_NUMBER_TOKEN_RE = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_PCT_OF_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s*(?:%|percent)\s+of\s+{_NUMBER_TOKEN_RE}", re.IGNORECASE)
_DIVIDED_BY_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s+(?:divided\s+by|over)\s+{_NUMBER_TOKEN_RE}", re.IGNORECASE)
_TIMES_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s+(?:times|multiplied\s+by)\s+{_NUMBER_TOKEN_RE}", re.IGNORECASE)
_PLUS_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s+plus\s+{_NUMBER_TOKEN_RE}", re.IGNORECASE)
_MINUS_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s+minus\s+{_NUMBER_TOKEN_RE}", re.IGNORECASE)
_OVER_BINARY_RE = re.compile(rf"(?<!\w){_NUMBER_TOKEN_RE}\s+over\s+{_NUMBER_TOKEN_RE}(?!\w)", re.IGNORECASE)
_DIRECT_EXPR_RE = re.compile(
    r"(?i)(?:\d[0-9,]*(?:\.\d+)?\s*(?:[+\-*/%^????]|\b(?:plus|minus|times|multiplied\s+by|divided\s+by|over)\b)\s*)+\d[0-9,]*(?:\.\d+)?"
)
_MATH_WORD_RE = re.compile(
    r"\b(?:plus|minus|times|multiplied\s+by|divided\s+by|percent\s+of)\b",
    re.IGNORECASE,
)


def _scope_text(text: str) -> str:
    """Prefer a transcript user block when one exists."""
    raw = str(text or "")
    if "## User" not in raw:
        return raw
    match = _TRANSCRIPT_USER_BLOCK_RE.search(raw)
    if not match:
        return raw
    body = str(match.group("body") or "").strip()
    return body or raw


def _strip_framing(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    t = _FRAMING_RE.sub("", t).strip()
    return _TRAILING_PUNCT_RE.sub("", t).strip()


def _clean_number(num: str) -> str:
    return str(num or "").replace(",", "").strip()


def _replace_binary_patterns(expr: str) -> str:
    out = str(expr or "")

    def _pct_repl(m: re.Match[str]) -> str:
        left = _clean_number(m.group(1))
        right = _clean_number(m.group(2))
        return f"({left} / 100) * {right}"

    def _div_repl(m: re.Match[str]) -> str:
        left = _clean_number(m.group(1))
        right = _clean_number(m.group(2))
        return f"{left} / {right}"

    def _mul_repl(m: re.Match[str]) -> str:
        left = _clean_number(m.group(1))
        right = _clean_number(m.group(2))
        return f"{left} * {right}"

    def _add_repl(m: re.Match[str]) -> str:
        left = _clean_number(m.group(1))
        right = _clean_number(m.group(2))
        return f"{left} + {right}"

    def _sub_repl(m: re.Match[str]) -> str:
        left = _clean_number(m.group(1))
        right = _clean_number(m.group(2))
        return f"{left} - {right}"

    out = _PCT_OF_RE.sub(_pct_repl, out)
    out = _DIVIDED_BY_RE.sub(_div_repl, out)
    out = _TIMES_RE.sub(_mul_repl, out)
    out = _PLUS_RE.sub(_add_repl, out)
    out = _MINUS_RE.sub(_sub_repl, out)
    out = out.replace("Ãƒâ€”", "*").replace("ÃƒÂ·", "/")
    out = re.sub(r"\bmultiplied\s+by\b", "*", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdivided\s+by\b", "/", out, flags=re.IGNORECASE)
    out = re.sub(r"\bpercent\s+of\b", "*", out, flags=re.IGNORECASE)
    out = re.sub(r"\bover\b", "/", out, flags=re.IGNORECASE)
    out = re.sub(r"\btimes\b", "*", out, flags=re.IGNORECASE)
    out = re.sub(r"\bplus\b", "+", out, flags=re.IGNORECASE)
    out = re.sub(r"\bminus\b", "-", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s*([+\-*/])\s*", r" \1 ", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = _TRAILING_PUNCT_RE.sub("", out).strip()
    return out


def has_arithmetic_intent(text: str) -> bool:
    """Return True for plain-NL arithmetic questions or near-expressions."""
    scoped = _scope_text(text)
    stripped = _strip_framing(scoped)
    if not stripped:
        return False
    if not (_NUMBER_RE.search(stripped) or _ARITHMETIC_SIGNAL_RE.search(stripped)):
        return False
    if _RANGE_RE.search(stripped) and re.search(r"[A-Za-z]", stripped):
        return False
    if _PCT_OF_RE.search(stripped):
        return True
    if _DIRECT_EXPR_RE.search(stripped):
        return True
    if _NUMBER_RE.search(stripped) and _MATH_WORD_RE.search(stripped):
        return True
    if _OVER_BINARY_RE.search(stripped):
        return True
    return False


def normalize_arithmetic_expression(text: str) -> str:
    """Normalize a plain-NL arithmetic question into a calc-safe expression."""
    scoped = _scope_text(text)
    expr = _strip_framing(scoped)
    if not expr:
        raise ValueError("No arithmetic expression found")
    if _RANGE_RE.search(expr) and re.search(r"[A-Za-z]", expr):
        raise ValueError("No arithmetic expression found")
    if not (_NUMBER_RE.search(expr) or _ARITHMETIC_SIGNAL_RE.search(expr)):
        raise ValueError("No arithmetic expression found")
    expr = _replace_binary_patterns(expr)
    if not _NUMBER_RE.search(expr):
        raise ValueError("No arithmetic expression found")
    return expr
