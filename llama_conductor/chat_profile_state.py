from __future__ import annotations

import re
from typing import Any, Callable, List

_SENSITIVE_CONFIRM_INTENT_RE = re.compile(
    r"\b(sext|sexy|dirty|horny|erotic|nude|blowjob|handjob|cum|tits?|boobs?|dick|pussy|slut|whore|fuck|fucking|go fuck|eat shit)\b",
    re.IGNORECASE,
)

_NICKNAME_BLOCK_PATTERNS = [
    re.compile(
        r"\b(?:don['â€™]?t|do not|stop)\s+call(?:ing)?\s+me\s+([a-z0-9][a-z0-9 '\\-]{1,60}?)(?=(?:\s*(?:[.!?,;:]|$)))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+am\s+not\s+([a-z0-9][a-z0-9 '\\-]{1,60}?)(?=(?:\s+(?:you|ya|u)\b|\s*(?:[.!?,;:]|$)))",
        re.IGNORECASE,
    ),
]


def clean_nickname_candidate(raw: str) -> str:
    s = (raw or "").strip().strip("'\"`")
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(
        r"\b(?:you|ya|u|dumb|stupid|fucking|fuckin|fuck|cunt|bitch|retard|asshole|muppet|idiot)\b.*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = s.strip("-_.,;:!? ")
    if len(s) < 2:
        return ""
    if len(s) > 48:
        s = s[:48].strip()
    return s


def extract_blocked_nicknames(user_text: str) -> List[str]:
    found: List[str] = []
    t = user_text or ""
    for rx in _NICKNAME_BLOCK_PATTERNS:
        for m in rx.finditer(t):
            nick = clean_nickname_candidate(m.group(1))
            if nick and nick not in found:
                found.append(nick)
    return found


def update_blocked_nicknames(state: Any, user_text: str) -> None:
    for nick in extract_blocked_nicknames(user_text):
        state.profile_blocked_nicknames.add(nick)


def requires_sensitive_confirm(
    *,
    state: Any,
    user_text: str,
    classify_sensitive_context_fn: Callable[[str], bool],
) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    if getattr(getattr(state, "interaction_profile", None), "sensitive_override", False):
        return False
    if not classify_sensitive_context_fn(t):
        return False
    return bool(_SENSITIVE_CONFIRM_INTENT_RE.search(t))
