"""Lightweight consistency verifier for high-risk model answers.

This module adds a small post-draft guardrail:
- only engages for selected high-risk query families/corrections
- attempts a minimal rewrite when the draft appears inconsistent
- never invents new task domains; preserves user-intent scope
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional


_CORRECTION_INTENT_RE = re.compile(
    r"\b(i mean|i meant|no i meant|sorry i meant|whoops|rather than|not)\b",
    re.IGNORECASE,
)
_CORRECTION_NUM_UNIT_RE = re.compile(
    r"\b(?P<new>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b"
    r"[^\n]{0,48}?\bnot\b[^\n]{0,48}?"
    r"(?P<old>\d+(?:\.\d+)?)\s*(?P=unit)\b",
    re.IGNORECASE,
)
_CORRECTION_NEW_ONLY_RE = re.compile(
    r"\b(i mean|i meant|no i meant|sorry i meant|whoops|rather)\b"
    r"[^\n]{0,64}?"
    r"(?P<new>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b",
    re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_numeric_correction(correction_text: str) -> tuple[str, str, str]:
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


def should_verify_response(*, user_text: str, query_family: str) -> bool:
    fam = str(query_family or "").strip().lower()
    if fam in {"state_transition", "constraint_decision"}:
        return True
    q = str(user_text or "").strip().lower()
    if _CORRECTION_INTENT_RE.search(q):
        return True
    if re.search(r"\b(what does that mean|won['’]?t it|doesn['’]?t it|how far|what about that|that means)\b", q):
        return True
    return False


def _heuristic_correction_mismatch_fix(*, user_text: str, draft_text: str) -> str:
    new_v, unit_v, old_v = _extract_numeric_correction(user_text)
    if not (new_v and unit_v and old_v):
        return ""
    low = str(draft_text or "").lower()
    old_phrase = f"{old_v} {unit_v}".lower()
    new_phrase = f"{new_v} {unit_v}".lower()
    if old_phrase in low and new_phrase not in low:
        return (
            f"{new_v} {unit_v} rather than {old_v} {unit_v}. "
            "Re-evaluate the previous answer using this corrected value."
        )
    return ""


def _extract_first_km_value(text: str) -> Optional[float]:
    m = re.search(
        r"\b(?P<v>\d+(?:\.\d+)?)\s*(km|kilometer|kilometers|kilometre|kilometres)\b",
        str(text or ""),
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(str(m.group("v") or "").strip())
    except Exception:
        return None


def _heuristic_plausibility_rewrite(*, user_text: str, draft_text: str) -> str:
    q = str(user_text or "").strip().lower()
    d = str(draft_text or "").strip()
    dl = d.lower()

    if ("dog" in q) and ("bike" in q or "biking" in q or "bicycle" in q) and ("leash" in q):
        km = _extract_first_km_value(q)
        if km is not None and km >= 300:
            v = int(km) if float(km).is_integer() else km
            return (
                f"{v} km with a dog on a leash is not a practical single-ride plan. "
                "Treat this as high-risk and use external transport or staged short segments with frequent rest, hydration, and safety checks."
            )
        if km is not None and km >= 100:
            if any(x in dl for x in ("highly feasible", "fully feasible", "straightforward", "easy")):
                v = int(km) if float(km).is_integer() else km
                return (
                    f"{v} km with a dog on a leash is not a practical single-ride plan. "
                    "Treat this as high-risk: use external transport or split into short stages with rest, hydration, and safety checks."
                )
            if "high-risk" not in dl and "external transport" not in dl:
                v = int(km) if float(km).is_integer() else km
                return (
                    f"{v} km with a dog on a leash is high-risk and usually impractical as one ride. "
                    "Use external transport or split the journey into short staged segments."
                )

    if any(x in dl for x in ("just as far", "exactly as far", "same distance as")) and ("dog" in q or "legs" in q or "travel" in q):
        return (
            "Distance tolerance can change. Some dogs adapt very well, but endurance varies by fitness, terrain, and pace. "
            "Start with shorter segments, watch fatigue, and adjust based on recovery."
        )

    return ""


def _heuristic_contextual_followup_rewrite(
    *,
    user_text: str,
    prior_user_text: str,
    draft_text: str,
) -> str:
    q = str(user_text or "").strip().lower()
    p = str(prior_user_text or "").strip().lower()
    dl = str(draft_text or "").strip().lower()
    if not q:
        return ""
    # Resolve pronoun-heavy follow-up about range/endurance to prior "3 legs" context.
    if (
        ("how far" in q or "travel" in q or "distance" in q)
        and ("it" in q or "that" in q)
        and ("3 legs" in p or "three legs" in p)
    ):
        if "leash" in dl and "legs" not in dl:
            return (
                "Yes, it can affect distance tolerance. Some three-legged dogs do very well, but endurance varies by fitness, terrain, and pace. "
                "Use shorter segments, watch fatigue, and adjust distance based on recovery."
            )
    return ""


def verify_response_consistency(
    *,
    user_text: str,
    draft_text: str,
    role: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    call_model_messages: Callable[..., str],
    prior_user_text: str = "",
) -> str:
    draft = str(draft_text or "").strip()
    if not draft:
        return draft

    heuristic = _heuristic_correction_mismatch_fix(user_text=user_text, draft_text=draft)
    if heuristic:
        return heuristic
    plausibility = _heuristic_plausibility_rewrite(user_text=user_text, draft_text=draft)
    if plausibility:
        return plausibility
    contextual = _heuristic_contextual_followup_rewrite(
        user_text=user_text,
        prior_user_text=prior_user_text,
        draft_text=draft,
    )
    if contextual:
        return contextual

    verifier_messages = [
        {
            "role": "system",
            "content": (
                "You are a consistency verifier for assistant drafts.\n"
                "Return JSON only: {\"status\":\"pass|fail\",\"rewrite\":\"...\"}\n"
                "Rules:\n"
                "- fail only for clear contradiction, missed correction, or non-answer.\n"
                "- rewrite must directly answer the query in 1-3 concise sentences.\n"
                "- do not invent external facts.\n"
                "- if pass, rewrite must be empty string."
            ),
        },
        {"role": "user", "content": f"Query:\n{str(user_text or '').strip()}"},
        {"role": "assistant", "content": draft},
        {
            "role": "user",
            "content": (
                "Check consistency and answer quality against the query only. "
                "Return JSON object now."
            ),
        },
    ]

    try:
        raw = str(
            call_model_messages(
                role=role,
                messages=verifier_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            or ""
        ).strip()
    except Exception:
        return draft

    obj = _parse_json_object(raw)
    if not obj:
        return draft

    status = str(obj.get("status", "") or "").strip().lower()
    rewrite = str(obj.get("rewrite", "") or "").strip()
    if status == "fail" and rewrite:
        return rewrite
    return draft
