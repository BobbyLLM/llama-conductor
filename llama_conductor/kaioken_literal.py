"""KAIOKEN literal follow-up parsing and routing helpers.

This module is intentionally pure helper logic so router orchestration can stay
small and keep explicit priority ordering in one place.
"""

from __future__ import annotations

import re
from typing import Callable

from .session_state import SessionState

_LITERAL_ANCHOR_STOPWORDS = {
    "this", "that", "it", "thing", "stuff", "one", "mine", "myself", "yourself",
    "my", "the", "what", "about", "and", "umm", "um", "uh", "hmm", "tips", "advice",
}

_ANCHOR_QUERY_STOPWORDS = _LITERAL_ANCHOR_STOPWORDS | {
    "any", "got", "have", "has", "had", "give", "tell", "show", "with", "for", "from",
    "into", "onto", "your", "you", "me", "i", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "can", "could", "should", "would", "will", "to", "of", "in", "on",
    "hot", "takes", "take", "thoughts", "thought", "pro", "advice", "tips", "tip", "help",
}

_LITERAL_DIRECT_HEAD_RE = re.compile(
    r"^\s*(?:and\s+)?(?:(?:what|how)\s+about\s+)?(?:my|the)\s+([a-z0-9][a-z0-9\-']{1,})\s*$",
    re.IGNORECASE,
)


def _extract_concrete_nouns(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", str(text or "").lower())
    out: list[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in _LITERAL_ANCHOR_STOPWORDS:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out


def _first_clause_for_anchor_parse(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    head = re.split(r"[!?]", t, maxsplit=1)[0]
    return str(head or "").strip()


def _resolve_recent_anchor(state: SessionState, query_text: str = "") -> str:
    try:
        recent = list(getattr(state, "kaioken_recent_user_nouns", []) or [])
        if recent:
            q_terms = {
                t for t in _extract_concrete_nouns(query_text)
                if t and t not in _ANCHOR_QUERY_STOPWORDS
            }
            best_term = ""
            best_score = -1.0
            n = len(recent)
            freq: dict[str, int] = {}
            last_idx: dict[str, int] = {}
            for idx, tok in enumerate(recent):
                if not tok or tok in _LITERAL_ANCHOR_STOPWORDS:
                    continue
                freq[tok] = int(freq.get(tok, 0)) + 1
                last_idx[tok] = idx
            for tok, f in freq.items():
                idx = int(last_idx.get(tok, 0))
                recency = (idx + 1) / max(1, n)
                score = (2.0 * float(f)) + recency
                if q_terms:
                    # Prefer overlaps/morphological-near matches when query carries topic signal.
                    if tok in q_terms:
                        score += 3.0
                    elif any(tok.startswith(q) or q.startswith(tok) for q in q_terms):
                        score += 1.5
                if score > best_score:
                    best_score = score
                    best_term = tok
            if best_term:
                return best_term
    except Exception:
        pass
    prev = str(getattr(state, "last_user_text", "") or "").strip()
    prev_nouns = _extract_concrete_nouns(prev)
    return prev_nouns[-1] if prev_nouns else ""


def _extract_literal_followup_anchor(user_text: str, *, state: SessionState | None = None) -> str:
    s = str(user_text or "").strip()
    if not s:
        return ""
    s_l = s.lower()
    s_head = _first_clause_for_anchor_parse(s).lower()
    if state is not None:
        try:
            cur_turn = int(getattr(state, "kaioken_turn_counter", 0) or 0)
            exp_turn = int(getattr(state, "kaioken_literal_anchor_expire_turn", -1) or -1)
            active_anchor = str(getattr(state, "kaioken_literal_anchor_active", "") or "").strip().lower()
            if (
                active_anchor
                and cur_turn <= exp_turn
                and re.search(
                    r"\b(what should i do about it|what do i do about it|how do i handle it|what about it|about it)\b",
                    s_l,
                    flags=re.IGNORECASE,
                )
            ):
                return active_anchor
        except Exception:
            pass
    m_hot = re.search(
        r"\b(?:hot\s+takes?|thoughts?|tips?|advice)\b.*?\b(?:about|on)\s+([a-z0-9][a-z0-9\-']{1,})\b",
        s_head or s_l,
        flags=re.IGNORECASE,
    )
    if m_hot:
        cand = str(m_hot.group(1) or "").strip().lower()
        if cand and cand not in _LITERAL_ANCHOR_STOPWORDS:
            return cand
        if state is not None:
            return _resolve_recent_anchor(state, s_l)
        return ""
    # Explicit short literal follow-up only; avoid matching arbitrary
    # possessives in normal sentences (e.g., "what's the point of ...").
    m_direct = _LITERAL_DIRECT_HEAD_RE.search(s_head or s_l)
    if m_direct:
        cand = str(m_direct.group(1) or "").strip().lower()
        if cand and cand not in _LITERAL_ANCHOR_STOPWORDS:
            return cand
        if state is not None:
            return _resolve_recent_anchor(state, s_l)
        return ""
    if re.search(r"\b(define that|literal|keep it literal|talk around)\b", s_l):
        if state is not None:
            return _resolve_recent_anchor(state, s_l)
        return ""
    if re.search(r"\b(hot\s+takes?|pro\s+tips?|tips?|advice|thoughts?)\b", s_l):
        if state is not None:
            return _resolve_recent_anchor(state, s_l)
        return ""
    return ""


def _literal_anchor_is_primary_subject(user_text: str, anchor: str) -> bool:
    s = str(user_text or "").strip().lower()
    s_head = _first_clause_for_anchor_parse(s).lower()
    a = str(anchor or "").strip().lower()
    if not s or not a:
        return False
    if _LITERAL_DIRECT_HEAD_RE.search(s_head or s):
        m = _LITERAL_DIRECT_HEAD_RE.search(s_head or s)
        got = str((m.group(1) if m else "") or "").strip().lower()
        if got == a:
            return True
    if re.search(
        rf"^\s*(?:and\s+)?(?:(?:what|how)\s+about\s+)?(?:my|the)\s+{re.escape(a)}\s*$",
        s_head or s,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(rf"\b{re.escape(a)}\b", s_head or s, flags=re.IGNORECASE) and re.search(
        r"\b(define that|literal|keep it literal|talk around)\b", s_head or s, flags=re.IGNORECASE
    ):
        return True
    return False


def _is_literal_followup_turn(
    *,
    state: SessionState,
    user_text: str,
    is_kaioken_guard_candidate_fn: Callable[..., bool],
) -> bool:
    anchor = _extract_literal_followup_anchor(user_text, state=state)
    prev = str(getattr(state, "last_user_text", "") or "").strip()
    if not prev:
        return False
    prev_l = prev.lower()
    cur_l = str(user_text or "").lower()
    # Challenge-phrasing lineage:
    if re.search(r"\b(define that|literal|keep it literal|talk around)\b", cur_l):
        return bool(anchor)
    if not anchor:
        return False
    if not _literal_anchor_is_primary_subject(user_text, anchor):
        return False
    # Path 1: short-form literal follow-up should route to literal lane by
    # anchor lineage, independent of classifier confidence.
    short_form = bool(
        re.search(
            rf"^\s*(?:and\s+)?(?:(?:what|how)\s+about\s+)?(?:my|the)\s+{re.escape(anchor)}\s*$",
            _first_clause_for_anchor_parse(cur_l) or cur_l,
            flags=re.IGNORECASE,
        )
    )
    if short_form:
        try:
            known = set(getattr(state, "kaioken_distress_topics", set()) or set())
            recent = {str(x).strip().lower() for x in list(getattr(state, "kaioken_recent_user_nouns", []) or [])}
            prev_terms = {str(x).strip().lower() for x in _extract_concrete_nouns(prev)}
            if anchor in known or anchor in recent or anchor in prev_terms:
                return True
        except Exception:
            pass
    # Explicit literal framing should always activate literal lane when anchor
    # parsing succeeds, even if classifier confidence is not high.
    if re.search(r"\b(metaphorical|actual|literal)\b", cur_l, flags=re.IGNORECASE):
        return True
    if re.search(rf"\b{re.escape(anchor)}\b", prev_l) and re.search(rf"\b{re.escape(anchor)}\b", cur_l):
        return True
    # Must be in the distress/VUH lineage we already guard.
    if not is_kaioken_guard_candidate_fn(state=state, user_text=user_text):
        return False
    return True
