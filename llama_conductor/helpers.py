# helpers.py
"""Helper functions for message parsing and manipulation."""

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Message Parsing
# ---------------------------------------------------------------------------

_IMAGE_BLOCK_TYPES = {"image_url", "input_image"}


def _extract_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("text", "input_text"):
            t = b.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts).strip()


def _has_image_blocks(blocks: List[Dict[str, Any]]) -> bool:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") in _IMAGE_BLOCK_TYPES:
            return True
    return False


def has_images_in_messages(messages: List[Dict[str, Any]]) -> bool:
    """Check if any message in the list contains images."""
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, list) and _has_image_blocks(c):
            return True
    return False


def has_mentats_in_recent_history(messages: List[Dict[str, Any]], last_n: int = 5) -> bool:
    """Check if any of the last N messages contains Mentats output markers."""
    recent = messages[-last_n:] if len(messages) > last_n else messages
    for m in recent:
        c = m.get("content", "")
        if isinstance(c, str) and ("[ZARDOZ HATH SPOKEN]" in c or "Sources: Vault" in c):
            return True
    return False


def normalize_history(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert possibly-multimodal content into text-only messages for non-vision pipelines."""
    out: List[Dict[str, Any]] = []
    for m in raw_messages or []:
        role = m.get("role", "")
        c = m.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = _extract_text_from_blocks(c)
            if _has_image_blocks(c):
                text = (text + "\n[image]") if text else "[image]"
        else:
            text = ""
        out.append({"role": role, "content": text})
    return out


def last_user_message(messages: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Extract the last user message text and its index."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            c = messages[i].get("content", "")
            if isinstance(c, str):
                return c, i
            if isinstance(c, list):
                return _extract_text_from_blocks(c), i
            return "", i
    return "", -1


# ---------------------------------------------------------------------------
# Command/Selector Parsing
# ---------------------------------------------------------------------------

def is_command(s: str) -> bool:
    """Check if string is a session command (>>). Does NOT match ?? (Vodka recall)."""
    t = (s or "").lstrip()
    return t.startswith(">>") or t.startswith("»") or t.startswith("Â»")


def strip_cmd_prefix(s: str) -> str:
    """Strip command prefix (>>, », or mojibake variants)."""
    s = (s or "").lstrip()
    
    # Support single-char command sigil as equivalent to >>
    # NOTE: Some clients deliver the '»' sigil as mojibake 'Â»'.
    if s.startswith("Â»"):
        return s[2:].strip()
    if s.startswith("»"):
        return s[1:].strip()
    if s.startswith(">>"):
        return s[2:].strip()
    return s.strip()


def split_selector(user_text: str) -> Tuple[str, str]:
    """Return (selector, text). selector is one of: '', 'mentats','fun','vision','ocr'."""
    t = (user_text or "").lstrip()

    # Support command-style aliases for vision/OCR:
    # - >>vision / >>vl / >>v
    # - >>ocr / >>read
    # Also supports mojibake command sigils handled by strip_cmd_prefix.
    if is_command(t):
        cmd = strip_cmd_prefix(t)
        m_cmd = re.match(r"^([A-Za-z_]+)\b(.*)$", cmd, flags=re.I | re.S)
        if m_cmd:
            head = (m_cmd.group(1) or "").strip().lower()
            rest = (m_cmd.group(2) or "").lstrip()
            if head in ("vision", "vl", "v"):
                return "vision", rest
            if head in ("ocr", "read"):
                return "ocr", rest
        return "", user_text

    if not t.startswith("##"):
        return "", user_text

    # allow: ##m, ##mentats, ##fun, ##vision, ##ocr
    m = re.match(r"^##\s*([A-Za-z_]+)\b(.*)$", t, flags=re.I | re.S)
    if not m:
        return "", user_text

    sel = (m.group(1) or "").strip().lower()
    rest = (m.group(2) or "").lstrip()

    if sel in ("m", "mentats"):
        return "mentats", rest
    if sel in ("fun", "f"):
        return "fun", rest
    if sel in ("vision",):
        return "vision", rest
    if sel in ("ocr",):
        return "ocr", rest

    return "", user_text


def parse_args(cmd: str) -> List[str]:
    """Parse space-separated arguments from command string."""
    return [p for p in (cmd or "").split() if p]
