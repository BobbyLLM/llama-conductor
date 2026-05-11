"""Deterministic currency-token extraction helpers.

This module is intentionally narrow:
- extract only $-prefixed currency tokens
- normalize them into a +-delimited arithmetic expression
- do not perform arithmetic itself
"""

from __future__ import annotations

import re
from typing import List


_CURRENCY_TOKEN_RE = re.compile(r"(?<!\w)\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_TRANSCRIPT_USER_BLOCK_RE = re.compile(
    r"(?ims)^\s*##\s*User\s*$\n(?P<body>.*?)(?=^\s*##\s*(?:Assistant|System|User)\s*$|\Z)"
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


def extract_currency_amount_tokens(text: str) -> List[str]:
    """Extract normalized currency tokens from text.

    Returns tokens without the leading dollar sign and without commas.
    """
    scoped = _scope_text(text)
    tokens: List[str] = []
    for match in _CURRENCY_TOKEN_RE.finditer(scoped):
        token = str(match.group(1) or "").replace(",", "").strip()
        if token:
            tokens.append(token)
    return tokens


def normalize_currency_sum_expression(text: str) -> str:
    """Normalize all currency tokens in text into a +-delimited expression."""
    tokens = extract_currency_amount_tokens(text)
    if not tokens:
        raise ValueError("No currency tokens found")
    return " + ".join(tokens)
