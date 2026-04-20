"""Shared helpers for privacy-safe logging and previews."""

from __future__ import annotations

import hashlib
import re


_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}\b",
    re.IGNORECASE,
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\.\- ]+\s(?:st|street|rd|road|ave|avenue|blvd|drive|dr|lane|ln|ct|court)\b",
    re.IGNORECASE,
)


def short_hash(text: str) -> str:
    raw = str(text or "")
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]


def redact_pii(text: str, token: str = "[REDACTED]") -> str:
    s = str(text or "")
    s = _EMAIL_RE.sub(token, s)
    s = _PHONE_RE.sub(token, s)
    s = _ADDRESS_RE.sub(token, s)
    return s


def contains_likely_pii(text: str) -> bool:
    s = str(text or "")
    if not s:
        return False
    low = s.lower()
    if any(k in low for k in ("date of birth", "dob", "my address", "my phone number", "my email")):
        return True
    return bool(_EMAIL_RE.search(s) or _PHONE_RE.search(s) or _ADDRESS_RE.search(s))


def safe_preview(text: str, max_len: int = 220) -> str:
    s = redact_pii(str(text or ""), token="[REDACTED]").replace("\n", " ").strip()
    if len(s) > max_len:
        return s[: max(0, max_len - 3)].rstrip() + "..."
    return s
