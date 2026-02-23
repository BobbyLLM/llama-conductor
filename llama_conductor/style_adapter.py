"""Deterministic response style post-processor (framing-only)."""

from __future__ import annotations

import re
from typing import List, Optional


_PRESERVE_LINE_RE = re.compile(
    r"^\s*(Confidence:|Source:|Sources:|Profile:|\[Not found in locked source|\[ZARDOZ|\[Note:|Scratchpad Quotes:|- \")",
    re.IGNORECASE,
)

_FRAMING_REWRITES = [
    (
        re.compile(r"\bI don['’]t experience[^.?!]*[.?!]\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bI process (?:your|this) (?:message|input)[^.?!]*[.?!]\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bMy role is to respond[^.?!]*[.?!]\s*", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bIf you['’]d like, I can adjust my tone[^.?!]*[.?!]\s*", re.IGNORECASE),
        "",
    ),
]

_DIRECT_HARD_BAN_RE = [
    re.compile(r"\bYou asked me\b", re.IGNORECASE),
    re.compile(r"\bI don['’]t experience\b", re.IGNORECASE),
    re.compile(r"\bI process (?:your|this) (?:message|input) as data\b", re.IGNORECASE),
    re.compile(r"\bMy role is\b", re.IGNORECASE),
    re.compile(r"\bIf you['’]d like, I can adjust my tone\b", re.IGNORECASE),
    re.compile(r"\bYou instructed me\b", re.IGNORECASE),
    re.compile(r"\bI processed that as your instruction\b", re.IGNORECASE),
    re.compile(r"\bYou said:\b", re.IGNORECASE),
]

_PROFANITY_RE = [
    (re.compile(r"\bfucking\b", re.IGNORECASE), "very"),
    (re.compile(r"\bfuck\b", re.IGNORECASE), "mess"),
    (re.compile(r"\bcunt\b", re.IGNORECASE), "person"),
    (re.compile(r"\bbitch\b", re.IGNORECASE), "person"),
    (re.compile(r"\basshole\b", re.IGNORECASE), "person"),
    (re.compile(r"\bmotherfucker\b", re.IGNORECASE), "person"),
]

_HOSTILE_INTENT_RE = re.compile(
    r"(email|draft|write).{0,60}(boss|manager|client|hr).{0,120}(fuck|shit|die|kill|cunt|bitch|asshole|get fucked|eat shit)",
    re.IGNORECASE | re.DOTALL,
)


def _clean_paragraph(text: str, *, correction_style: str) -> str:
    raw = (text or "").strip()
    out = raw
    if not out:
        return ""
    for rx, repl in _FRAMING_REWRITES:
        out = rx.sub(repl, out)
    out = re.sub(r"\bYou asked me to\b", "Noted:", out, flags=re.IGNORECASE)
    if correction_style == "direct":
        for rx in _DIRECT_HARD_BAN_RE:
            out = rx.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    if correction_style == "direct" and len(out) < 12 and raw:
        return "Understood. Rephrase the exact output you want, and I'll do it directly."
    return out


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sanitize_sensitive_text(line: str) -> str:
    out = line
    for rx, repl in _PROFANITY_RE:
        out = rx.sub(repl, out)
    return out


def _nickname_rx(nickname: str) -> Optional[re.Pattern[str]]:
    n = (nickname or "").strip()
    if not n:
        return None
    esc = re.escape(n).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{esc}\b", re.IGNORECASE)


def _strip_blocked_nicknames(line: str, blocked_nicknames: List[str]) -> str:
    out = line
    for nick in blocked_nicknames:
        rx = _nickname_rx(nick)
        if rx is None:
            continue
        out = rx.sub("you", out)
    # Clean obvious vocative leftovers after replacement.
    out = re.sub(r",\s*you(?=[\s\.\!\?,;:]|$)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\byou\s+you\b", "you", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out.lower() in {"you", "you.", "you!", "you?"}:
        return ""
    return out


def _extract_footer_lines(text: str) -> List[str]:
    return [ln for ln in (text or "").splitlines() if _PRESERVE_LINE_RE.match(ln or "")]


def _safe_professional_email_stub() -> str:
    return (
        "Subject: Quick Note\n\n"
        "Hi [Boss's Name],\n\n"
        "I want to raise a concern directly, but respectfully. "
        "Could we schedule time to discuss the issue and agree on next steps?\n\n"
        "Best,\n"
        "[Your Name]"
    )


def score_output_compliance(
    text: str,
    *,
    correction_style: str,
    user_text: str = "",
    blocked_nicknames: Optional[List[str]] = None,
) -> float:
    """Score whether output follows style guardrails (0..1)."""
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 1.0
    low = t.lower()
    if correction_style == "direct":
        for rx in _DIRECT_HARD_BAN_RE:
            if rx.search(t):
                score -= 0.25
    if "i don't experience" in low:
        score -= 0.2
    if "i process your message as data" in low:
        score -= 0.2
    if "if you'd like, i can adjust my tone" in low:
        score -= 0.15
    if "my role is to" in low:
        score -= 0.1
    if "you instructed me" in low:
        score -= 0.15
    if "i processed that as your instruction" in low:
        score -= 0.15
    if "you said:" in low:
        score -= 0.1
    if user_text:
        u = _norm(user_text)
        body = [ln for ln in t.splitlines() if ln.strip() and not _PRESERVE_LINE_RE.match(ln or "")]
        if body and _norm(body[0]) == u:
            score -= 0.25
    for nick in (blocked_nicknames or []):
        rx = _nickname_rx(nick)
        if rx is not None and rx.search(t):
            score -= 0.4
    return max(0.0, min(1.0, score))


def rewrite_response_style(
    text: str,
    *,
    enabled: bool,
    correction_style: str = "neutral",
    user_text: str = "",
    sensitive_context: bool = False,
    sensitive_override: bool = False,
    blocked_nicknames: Optional[List[str]] = None,
) -> str:
    """Rewrite only framing language; keep factual/provenance lines untouched."""
    t = (text or "").strip()
    if not enabled or not t:
        return t

    # Mentats contract stays untouched.
    if "sources: vault" in t.lower():
        return t

    lines = t.splitlines()
    out_lines: List[str] = []
    para_buf: List[str] = []

    def flush_para() -> None:
        if not para_buf:
            return
        p = " ".join(s.strip() for s in para_buf if s.strip())
        p2 = _clean_paragraph(p, correction_style=correction_style)
        if p2:
            out_lines.append(p2)
        para_buf.clear()

    for ln in lines:
        if _PRESERVE_LINE_RE.match(ln or ""):
            flush_para()
            out_lines.append(ln)
            continue
        if not (ln or "").strip():
            flush_para()
            out_lines.append("")
            continue
        para_buf.append(ln)

    flush_para()

    # Collapse long blank runs.
    collapsed: List[str] = []
    blank_run = 0
    for ln in out_lines:
        if not (ln or "").strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(ln)

    out = "\n".join(collapsed).strip()

    if sensitive_context and not sensitive_override and out:
        if _HOSTILE_INTENT_RE.search(user_text or ""):
            preserved = _extract_footer_lines(out)
            out = _safe_professional_email_stub()
            if preserved:
                out = out + "\n\n" + "\n".join(preserved)
            return out.strip()
        sanitized: List[str] = []
        for ln in out.splitlines():
            if _PRESERVE_LINE_RE.match(ln or ""):
                sanitized.append(ln)
            else:
                sanitized.append(_sanitize_sensitive_text(ln))
        out = "\n".join(sanitized).strip()

    if blocked_nicknames and out:
        filtered: List[str] = []
        for ln in out.splitlines():
            # Keep FUN quote headers and deterministic footers unchanged.
            if _PRESERVE_LINE_RE.match(ln or "") or (ln or "").strip().startswith("[FUN]"):
                filtered.append(ln)
            else:
                filtered.append(_strip_blocked_nicknames(ln, blocked_nicknames))
        out = "\n".join(filtered).strip()

    if user_text:
        u = _norm(user_text)
        lines = out.splitlines()
        body_idx = -1
        for i, ln in enumerate(lines):
            if (ln or "").strip() and not _PRESERVE_LINE_RE.match(ln or ""):
                body_idx = i
                break
        if body_idx >= 0 and _norm(lines[body_idx]) == u:
            lines.pop(body_idx)
            out = "\n".join(lines).strip()
        # Strip explicit mirror wrappers in leading body line.
        lines = out.splitlines()
        body_idx = -1
        for i, ln in enumerate(lines):
            if (ln or "").strip() and not _PRESERVE_LINE_RE.match(ln or ""):
                body_idx = i
                break
        if body_idx >= 0:
            b = (lines[body_idx] or "").strip()
            b_low = b.lower()
            if b_low.startswith("you said:") or b_low.startswith("you instructed me"):
                lines.pop(body_idx)
                out = "\n".join(lines).strip()

    # Safety: never return footer-only/empty responses after rewrite.
    content_lines = [
        ln for ln in (out.splitlines() if out else [])
        if (ln or "").strip() and not _PRESERVE_LINE_RE.match(ln or "")
    ]
    if content_lines:
        return out

    # Keep preserved metadata/footer lines, but prepend a minimal body line.
    preserved = [ln for ln in (out.splitlines() if out else []) if (ln or "").strip()]
    if preserved:
        return "Noted.\n\n" + "\n".join(preserved)

    # Absolute fallback for pathological cases.
    return "Understood. Rephrase the exact output you want, and I'll do it directly."
