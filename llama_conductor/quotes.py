# quotes.py
"""Quote loading and tone inference."""

import os
import random
from typing import Dict, List, Optional

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


def quotes_by_tag() -> Dict[str, List[str]]:
    """Get cached quotes dictionary."""
    global _QUOTES_CACHE
    if _QUOTES_CACHE is None:
        _QUOTES_CACHE = load_quotes_md(QUOTES_MD_PATH)
        # Debug: log if quotes failed to load
        if not _QUOTES_CACHE:
            print(f"[router WARNING] quotes.md not found or empty at {QUOTES_MD_PATH}")
    return _QUOTES_CACHE


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
