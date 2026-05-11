# quotes.py
"""Quote loading and tone inference.

Step-1 register metadata:
- Parse section headers in quotes.md
- Auto-classify each section into macro register tags
- Expose read-only section register map for later selector filtering
"""

import os
import random
from typing import Dict, List, Optional, Set

from .config import QUOTES_MD_PATH
from .model_calls import call_model_prompt


def load_quotes_md(path: str) -> Dict[str, List[str]]:
    """Parse quotes.md into {tag: [quotes...]}. Supports headings like: ## futurama snark sarcastic"""
    if not os.path.isfile(path):
        return {}

    tag_to_quotes: Dict[str, List[str]] = {}
    current_tags: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith("##"):
                tags = line.strip().lstrip("#").strip()
                current_tags = [t.strip().lower() for t in tags.split() if t.strip()]
                continue

            q = line.strip()
            if not q:
                continue

            # ignore markdown bullet markers
            if q.startswith("-"):
                q = q.lstrip("-").strip()
            if not q:
                continue

            for t in current_tags or ["default"]:
                tag_to_quotes.setdefault(t, []).append(q)

    return tag_to_quotes


_QUOTES_CACHE: Optional[Dict[str, List[str]]] = None
_QUOTE_SECTION_REGISTERS_CACHE: Optional[Dict[str, List[str]]] = None
_QUOTE_SECTION_QUOTES_CACHE: Optional[Dict[str, List[str]]] = None


REGISTER_KEYWORDS = {
    "casual": ["snark", "sarcastic", "deadpan", "banter", "one-liners", "quips", "dry"],
    "personal": ["warm", "supportive", "compassionate", "hopeful", "resilient"],
    "working": ["meta", "identity", "performance", "existential"],
    "exclude_distress": ["threat", "warning"],
}
_REGISTER_FAIL_OPEN = ["casual", "working", "personal"]


def _classify_section_registers(header_text: str) -> List[str]:
    """Classify a quotes.md section header into register tags (auto, fail-open)."""
    low = str(header_text or "").strip().lower()
    if not low:
        return list(_REGISTER_FAIL_OPEN)
    matched: Set[str] = set()
    for reg, keys in REGISTER_KEYWORDS.items():
        for kw in keys:
            if kw in low:
                matched.add(reg)
                break
    if not matched:
        return list(_REGISTER_FAIL_OPEN)
    # Keep stable order for deterministic downstream behavior.
    ordered = [k for k in ("casual", "working", "personal", "exclude_distress") if k in matched]
    return ordered or list(_REGISTER_FAIL_OPEN)


def load_quote_section_register_map(path: str) -> Dict[str, List[str]]:
    """Build {section_header_text_lower: [register_tags...]} from quotes.md."""
    if not os.path.isfile(path):
        return {}
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip().startswith("##"):
                continue
            header = line.strip().lstrip("#").strip().lower()
            if not header:
                continue
            out[header] = _classify_section_registers(header)
    return out


def load_quote_section_quotes_map(path: str) -> Dict[str, List[str]]:
    """Build {section_header_text_lower: [quote lines...]} from quotes.md."""
    if not os.path.isfile(path):
        return {}
    out: Dict[str, List[str]] = {}
    current_header = ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue
            if s.startswith("##"):
                current_header = s.lstrip("#").strip().lower()
                out.setdefault(current_header, [])
                continue
            if s.startswith("#"):
                continue
            q = s.lstrip("-").strip() if s.startswith("-") else s
            if not q or not current_header:
                continue
            out.setdefault(current_header, []).append(q)
    return out


def quotes_by_tag() -> Dict[str, List[str]]:
    """Get cached quotes dictionary."""
    global _QUOTES_CACHE, _QUOTE_SECTION_REGISTERS_CACHE, _QUOTE_SECTION_QUOTES_CACHE
    if _QUOTES_CACHE is None:
        _QUOTES_CACHE = load_quotes_md(QUOTES_MD_PATH)
        _QUOTE_SECTION_REGISTERS_CACHE = load_quote_section_register_map(QUOTES_MD_PATH)
        _QUOTE_SECTION_QUOTES_CACHE = load_quote_section_quotes_map(QUOTES_MD_PATH)
        # Debug: log if quotes failed to load
        if not _QUOTES_CACHE:
            print(f"[router WARNING] quotes.md not found or empty at {QUOTES_MD_PATH}")
    return _QUOTES_CACHE


def quote_section_register_map() -> Dict[str, List[str]]:
    """Get cached section->register tags map built from quotes.md headers."""
    global _QUOTE_SECTION_REGISTERS_CACHE
    if _QUOTE_SECTION_REGISTERS_CACHE is None:
        _QUOTE_SECTION_REGISTERS_CACHE = load_quote_section_register_map(QUOTES_MD_PATH)
    return _QUOTE_SECTION_REGISTERS_CACHE


def quote_section_quotes_map() -> Dict[str, List[str]]:
    """Get cached section->quotes map built from quotes.md headers."""
    global _QUOTE_SECTION_QUOTES_CACHE
    if _QUOTE_SECTION_QUOTES_CACHE is None:
        _QUOTE_SECTION_QUOTES_CACHE = load_quote_section_quotes_map(QUOTES_MD_PATH)
    return _QUOTE_SECTION_QUOTES_CACHE


def pick_quote_for_tone(tone: str) -> str:
    """Select a random quote matching the given tone tag."""
    tone = (tone or "").strip().lower()
    qb = quotes_by_tag()

    pool = qb.get(tone) or qb.get("default") or []
    if not pool:
        return ""

    return random.choice(pool)


def infer_tone(user_text: str, answer_text: str) -> str:
    """Ask the model for a single tone tag that exists in quotes.md."""
    tags = sorted(set(quotes_by_tag().keys()))
    if not tags:
        return "default"

    prompt = (
        "You are selecting a single tone tag for a pop-culture quote seed.\n"
        "Return EXACTLY ONE tag from the allowed list. No extra text.\n\n"
        f"ALLOWED_TAGS: {', '.join(tags[:120])}\n\n"
        f"USER: {user_text.strip()}\n\n"
        f"ANSWER: {answer_text.strip()}\n\n"
        "TAG:"
    )

    raw = call_model_prompt(role="thinker", prompt=prompt, max_tokens=10, temperature=0.1, top_p=0.9)
    tag = (raw or "").strip().lower().split()[0] if raw else ""
    return tag if tag in set(tags) else "default"
