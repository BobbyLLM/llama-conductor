"""KAIOKEN output-guard utility helpers.

This module keeps pure, reusable guard mechanics separate from router
orchestration so priority decisions remain explicit in one place.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Tuple

from .session_state import SessionState


def _split_fun_prefix(text: str) -> tuple[str, str]:
    raw = str(text or "")
    if not raw.strip():
        return "", ""
    lines = raw.splitlines()
    if not lines:
        return "", raw
    first = (lines[0] or "").strip()
    if first.startswith("[FUN]") or first.startswith("[FUN REWRITE]"):
        idx = 1
        while idx < len(lines) and not (lines[idx] or "").strip():
            idx += 1
        prefix = "\n".join(lines[:idx]).rstrip()
        body = "\n".join(lines[idx:]).strip()
        return prefix, body
    return "", raw.strip()


def _normalize_for_repeat_check(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    lines: list[str] = []
    for ln in raw.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if re.match(r"^\s*(confidence|source|profile)\s*:", s, flags=re.I):
            continue
        lines.append(s)
    out = " ".join(lines).strip().lower()
    out = re.sub(r"\s+", " ", out)
    return out


def _sentence_count_for_guard(text: str) -> int:
    t = str(text or "").strip()
    if not t:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", t)
    return len([p for p in parts if str(p).strip()])


def _repeat_like(a: str, b: str) -> bool:
    a_n = _normalize_for_repeat_check(a)
    b_n = _normalize_for_repeat_check(b)
    if not a_n or not b_n:
        return False
    if len(a_n) < 40 or len(b_n) < 40:
        return a_n == b_n or SequenceMatcher(None, a_n, b_n).ratio() >= 0.95
    return SequenceMatcher(None, a_n, b_n).ratio() >= 0.90


def _recent_repeat_within_window(state: SessionState, text: str, *, window: int = 4) -> bool:
    cur = _normalize_for_repeat_check(text)
    if not cur:
        return False
    try:
        recent = list(getattr(state, "kaioken_recent_assistant_bodies", []) or [])
        for prev in recent[-max(1, int(window)):]:
            if _repeat_like(str(prev or ""), cur):
                return True
    except Exception:
        return False
    return False


def _remember_recent_assistant_body(state: SessionState, text: str, *, max_items: int = 8) -> None:
    try:
        cur = _normalize_for_repeat_check(text)
        if not cur:
            return
        recent = list(getattr(state, "kaioken_recent_assistant_bodies", []) or [])
        recent.append(cur)
        setattr(state, "kaioken_recent_assistant_bodies", recent[-max_items:])
    except Exception:
        pass


def _choose_short_fallback(
    state: SessionState,
    short_fallbacks: Tuple[str, ...],
    *,
    prior_text: str = "",
) -> str:
    key = f"{getattr(state, 'kaioken_turn_counter', 0)}|{str(getattr(state, 'session_id', '') or '')}"
    idx = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % len(short_fallbacks)
    cand = short_fallbacks[idx]
    if _repeat_like(cand, prior_text):
        cand = short_fallbacks[(idx + 1) % len(short_fallbacks)]
    try:
        last_short = str(getattr(state, "kaioken_last_short_fallback", "") or "").strip().lower()
        base = cand.strip()
        turn = int(getattr(state, "kaioken_turn_counter", 0) or 0)
        last_short_turn = int(getattr(state, "kaioken_last_short_turn", -9999) or -9999)
        distress_lane = bool(getattr(state, "kaioken_short_fallback_distress_lane", False))
        consecutive_turn = bool(last_short and (turn - last_short_turn) <= 2)
        if consecutive_turn:
            if distress_lane:
                invite_opts = ("I'm listening.", "Go on, if you want.")
                key = f"{getattr(state, 'session_id', '')}|{turn}|invite"
                iidx = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % len(invite_opts)
                cand = f"{base} {invite_opts[iidx]}"
            elif last_short == base.lower():
                cand = f"{base} Again."
        setattr(state, "kaioken_last_short_fallback", base)
        setattr(state, "kaioken_last_short_turn", turn)
    except Exception:
        pass
    return cand

