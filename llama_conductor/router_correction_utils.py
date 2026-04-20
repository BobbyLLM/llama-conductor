from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_CORRECTION_INTENT_RE = re.compile(
    r"\b(i mean|i meant|no i meant|sorry i meant|rather than)\b",
    re.IGNORECASE,
)
# NOTE(debt): lexical fallback only.
# Primary correction routing should use `evaluate_structural_correction_intent`
# so we rely on deterministic turn/state shape, not token whack-a-mole.
_EXPLICIT_REENGAGE_RE = re.compile(
    r"\b(back to|same decision|that decision|re-?engage|option\s*[123])\b",
    re.IGNORECASE,
)
_CORRECTION_NUM_UNIT_RE = re.compile(
    r"\b(?P<new>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b"
    r"[^\n]{0,48}?\bnot\b[^\n]{0,48}?"
    r"(?P<old>\d+(?:\.\d+)?)\s*(?P=unit)\b",
    re.IGNORECASE,
)
_CORRECTION_NEW_ONLY_RE = re.compile(
    r"\b(i mean|i meant|no i meant|sorry i meant|whoops|whoops!|whoops!!|rather)\b"
    r"[^\n]{0,64}?"
    r"(?P<new>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b",
    re.IGNORECASE,
)
_NUM_UNIT_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b",
    re.IGNORECASE,
)

_CORRECTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "just",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
    "you",
    "your",
}
_CORRECTION_STRUCTURAL_REPLACEMENT_RE = re.compile(
    r"\b("
    r"i meant(?: to say)?|no i meant|sorry i meant|"
    r"not\s+.+\s+but\s+.+|rather than|instead of|"
    r"actually|correction|to clarify"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']{1,}", re.IGNORECASE)


def is_correction_intent_query(text: str) -> bool:
    return bool(_CORRECTION_INTENT_RE.search(str(text or "")))


def _token_set(text: str) -> set[str]:
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(str(text or "").lower()):
        tok = str(m.group(0) or "").strip("'-")
        if not tok or tok in _CORRECTION_STOPWORDS:
            continue
        out.add(tok)
    return out


_POSITIVE_AGREEMENT_RE = re.compile(
    r"\b(?:good|great|solid|fair|helpful|pretty good|nice|excellent|perfect)\s+"
    r"(?:advice|point|answer|take|call|idea|suggestion|tip)\b"
    r"|\bthat(?:'s| is)\s+(?:good|great|helpful|solid|fair|pretty good|right|true)\b"
    r"|\bnot\s+(?:bad|wrong)\b"
    r"|\bthanks?\b|\bthank\s+you\b|\bcheers\b",
    re.IGNORECASE,
)

_CONTRASTIVE_OPERATOR_RE = re.compile(
    r"\bbut\b|\bactually\b|\bhowever\b|\binstead\b"
    r"|\bnot\s+(?:right|correct|quite)\b"
    r"|\bwrong\b|\bi\s+meant\b|\brather\b",
    re.IGNORECASE,
)


def evaluate_structural_correction_intent(
    *,
    user_text: str,
    prior_assistant_text: str,
    recent_user_nouns: Optional[List[str]] = None,
    recent_thread_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministic correction-intent check using recent state only.

    Shape required:
    1) prior claim exists
    2) user references prior claim surface
    3) user provides replacement payload (new content/contrast)
    """
    user = str(user_text or "").strip()
    prior = str(prior_assistant_text or "").strip()
    user_toks = _token_set(user)
    prior_toks = _token_set(prior)
    topic_surface = {
        str(x or "").strip().lower()
        for x in (list(recent_user_nouns or []) + list(recent_thread_tokens or []))
        if str(x or "").strip()
    }

    prior_claim_exists = bool(prior and len(prior_toks) >= 4)
    overlap = user_toks & prior_toks
    topic_overlap = user_toks & topic_surface if topic_surface else set()
    references_prior = bool(overlap) and bool(topic_overlap or len(overlap) >= 2)

    novel = user_toks - prior_toks
    has_novel_payload = len(novel) >= 2
    has_numeric_delta = bool(re.search(r"\d", user)) and not bool(re.search(r"\d", prior))
    has_restatement_operator = bool(_CORRECTION_STRUCTURAL_REPLACEMENT_RE.search(user))
    has_replacement_payload = bool(has_novel_payload or has_numeric_delta or has_restatement_operator)

    is_structural = bool(prior_claim_exists and references_prior and has_replacement_payload)

    # Positive agreement without contrastive language is not a correction.
    # "That's pretty good advice!" → not structural.
    # "That's pretty good advice, but make it 10km" → still structural.
    if is_structural and bool(_POSITIVE_AGREEMENT_RE.search(user)):
        if not bool(_CONTRASTIVE_OPERATOR_RE.search(user)):
            is_structural = False

    return {
        "is_structural": is_structural,
        "prior_claim_exists": prior_claim_exists,
        "references_prior": references_prior,
        "has_replacement_payload": has_replacement_payload,
        "overlap_count": len(overlap),
        "topic_overlap_count": len(topic_overlap),
        "novel_count": len(novel),
    }


def is_explicit_reengage_query(text: str) -> bool:
    return bool(_EXPLICIT_REENGAGE_RE.search(str(text or "")))


def extract_numeric_correction(correction_text: str) -> tuple[str, str, str]:
    m = _CORRECTION_NUM_UNIT_RE.search(str(correction_text or ""))
    if m:
        return (
            str(m.group("new") or "").strip(),
            str(m.group("unit") or "").strip(),
            str(m.group("old") or "").strip(),
        )
    m2 = _CORRECTION_NEW_ONLY_RE.search(str(correction_text or ""))
    if m2:
        return (
            str(m2.group("new") or "").strip(),
            str(m2.group("unit") or "").strip(),
            "",
        )
    return ("", "", "")


def unit_canonical(unit: str) -> str:
    u = str(unit or "").strip().lower()
    if u.startswith("kilo") or u == "km":
        return "km"
    if u.startswith("met"):
        return "m"
    return u


def units_match(u1: str, u2: str) -> bool:
    c1 = unit_canonical(u1)
    c2 = unit_canonical(u2)
    return bool(c1 and c2 and c1 == c2)


def extract_first_num_unit(text: str) -> tuple[str, str]:
    m = _NUM_UNIT_RE.search(str(text or ""))
    if not m:
        return ("", "")
    return (str(m.group("num") or "").strip(), str(m.group("unit") or "").strip())


def to_km(value: str, unit: str) -> Optional[float]:
    try:
        v = float(str(value or "").strip())
    except Exception:
        return None
    cu = unit_canonical(unit)
    if cu == "km":
        return v
    if cu == "m":
        return v / 1000.0
    return None


def resolve_old_from_prior_answer(*, new: str, unit: str, old: str, prior_answer: str) -> str:
    if old:
        return str(old).strip()
    if not (new and unit and prior_answer):
        return ""
    p_num, p_unit = extract_first_num_unit(prior_answer)
    if p_num and p_unit and units_match(p_unit, unit):
        return p_num
    return ""


def strip_got_it_prefix(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    return re.sub(r"^\s*got it\s*[:.,-]?\s*", "", t, flags=re.IGNORECASE)


def fallback_contextual_correction(
    *,
    user_text: str,
    query_family: str,
    old_val: str,
    new_val: str,
    unit: str,
    extra_note: str = "",
) -> str:
    q = " ".join(str(user_text or "").lower().split())
    unit_txt = str(unit or "").strip()
    prefix = f"Got it. {new_val}{unit_txt} instead of {old_val}{unit_txt}. " if old_val else f"Got it. {new_val}{unit_txt}. "
    if "bike" in q or "walk" in q or "drive" in q:
        if query_family == "constraint_decision":
            body = (
                "That changes the trade-off. For this distance, walking is practical and driving only helps if you need speed or are carrying load."
            )
        else:
            body = "Updated distance noted. Recalculate with the new value before deciding."
    else:
        body = "Update applied to the prior calculation."
    if extra_note:
        body = f"{body} {extra_note}".strip()
    return (prefix + body).strip()


def last_assistant_text(history: List[Dict[str, Any]]) -> str:
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            return str(history[i].get("content", "") or "").strip()
    return ""


def last_user_text(history: List[Dict[str, Any]]) -> str:
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            return str(history[i].get("content", "") or "").strip()
    return ""


def last_user_text_before(history: List[Dict[str, Any]], current_user_text: str) -> str:
    cur = str(current_user_text or "").strip()
    seen_current = False
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") != "user":
            continue
        t = str(history[i].get("content", "") or "").strip()
        if not seen_current and cur and t == cur:
            seen_current = True
            continue
        return t
    return ""


def last_non_correction_user_text(history: List[Dict[str, Any]], current_user_text: str = "") -> str:
    cur = str(current_user_text or "").strip()
    seen_current = False
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") != "user":
            continue
        t = str(history[i].get("content", "") or "").strip()
        if not seen_current and cur and t == cur:
            seen_current = True
            continue
        if not is_correction_intent_query(t):
            return t
    return ""
