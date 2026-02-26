from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List


_INLINE_CONFIDENCE_LINE_RE = re.compile(
    r"^\s*(?:high|medium|low|top|unverified)\s+confidence(?:\s*[:\-]?\s*(?:contextual|model|docs|scratchpad|vault|user|mixed)|\.\s*(?:contextual|model|docs|scratchpad|vault|user|mixed))?[\.\!\?]?\s*$",
    re.IGNORECASE,
)
_INLINE_SOURCE_ONLY_RE = re.compile(
    r"^\s*(?:contextual|model|docs|scratchpad|vault|user|mixed)\.?\s*$",
    re.IGNORECASE,
)
_INLINE_CONF_SOURCE_PREFIX_RE = re.compile(r"^\s*(confidence|source|sources)\s*[:\-]", re.IGNORECASE)
_ARGUMENTATIVE_PROMPT_RE = re.compile(
    r"\b(argue|argument|debate|logical gap|non sequitur|reply|response|rebut|opponent|premise|conclusion)\b",
    re.IGNORECASE,
)
_ARGUMENT_MARKER_RE = re.compile(
    r"\b(because|therefore|however|but|non sequitur|category error|premise|conclusion|begging the question|assumes)\b",
    re.IGNORECASE,
)
_ACK_REFRAME_LOOP_RE = re.compile(
    r"("
    r"you['â€™]re right"
    r"|you have made it clear"
    r"|let['â€™]s move past repetition"
    r"|missed (?:the )?emotional tone"
    r"|overly mechanical"
    r"|i should (?:be )?more attuned"
    r"|my (?:last )?reply was stiff"
    r")",
    re.IGNORECASE,
)


def strip_in_body_confidence_source_claims(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    out: List[str] = []
    drop_next_source = False
    for ln in t.splitlines():
        s = (ln or "").strip()
        if not s:
            out.append("")
            drop_next_source = False
            continue
        if _INLINE_CONF_SOURCE_PREFIX_RE.match(s) or _INLINE_CONFIDENCE_LINE_RE.match(s):
            drop_next_source = True
            continue
        if drop_next_source and _INLINE_SOURCE_ONLY_RE.match(s):
            drop_next_source = False
            continue
        drop_next_source = False
        out.append(ln)
    collapsed: List[str] = []
    blank = 0
    for ln in out:
        if not (ln or "").strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        collapsed.append(ln)
    return "\n".join(collapsed).strip()


def is_argumentative_prompt(user_text: str) -> bool:
    return bool(_ARGUMENTATIVE_PROMPT_RE.search(str(user_text or "")))


def extract_body_for_mode_checks(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    lines: List[str] = []
    skip_quotes_block = False
    for ln in t.splitlines():
        s = (ln or "").strip()
        if not s:
            lines.append("")
            skip_quotes_block = False
            continue
        if s.startswith("[FUN]") or s.startswith("[FUN REWRITE]"):
            continue
        if s.startswith("Scratchpad Quotes:"):
            skip_quotes_block = True
            continue
        if skip_quotes_block and s.startswith('- "'):
            continue
        if s.lower().startswith("confidence:") or s.lower().startswith("source:") or s.lower().startswith("profile:"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def is_argumentatively_complete(text: str) -> bool:
    body = extract_body_for_mode_checks(text)
    if not body:
        return False
    if len(body) < 180:
        return False
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", body))
    if sentence_count < 2:
        return False
    if not _ARGUMENT_MARKER_RE.search(body):
        return False
    return True


def fallback_with_mode_header(current_text: str, fallback_body: str) -> str:
    cur = (current_text or "").strip()
    fb = (fallback_body or "").strip()
    if not fb:
        return cur
    lines = cur.splitlines()
    if not lines:
        return fb
    header = (lines[0] or "").strip()
    if header.startswith("[FUN]") or header.startswith("[FUN REWRITE]"):
        return f"{header}\n\n{fb}"
    return fb


def strip_footer_lines_for_scan(text: str) -> str:
    keep: List[str] = []
    for ln in (text or "").splitlines():
        low = (ln or "").strip().lower()
        if low.startswith("confidence:"):
            continue
        if low.startswith("source:") or low.startswith("sources:"):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def is_ack_reframe_only(text: str) -> bool:
    body = strip_footer_lines_for_scan(text)
    if not body:
        return False
    if len(body) > 420:
        return False
    if not _ACK_REFRAME_LOOP_RE.search(body):
        return False
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("[FUN]")]
    if not lines:
        return False
    return len(lines) <= 5


def normalize_signature_text(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 220:
        s = s[:220].strip()
    return s


def _fun_body_lines(text: str) -> List[str]:
    lines: List[str] = []
    for ln in (text or "").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        low = s.lower()
        if s.startswith("[FUN]"):
            continue
        if low.startswith("confidence:") or low.startswith("source:") or low.startswith("sources:"):
            continue
        lines.append(s)
    return lines


def enforce_fun_antiparrot(text: str, user_text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    body_lines = _fun_body_lines(t)
    if not body_lines:
        return t

    body = " ".join(body_lines)
    bnorm = normalize_signature_text(body)
    unorm = normalize_signature_text(user_text)
    if not bnorm or not unorm:
        return t

    sim = SequenceMatcher(None, bnorm, unorm).ratio()
    body_low = body.lower()
    user_low = (user_text or "").strip().lower()
    likely_echo = (
        sim >= 0.82
        or body_low.startswith(user_low[: min(len(user_low), 80)])
        or (len(body_lines) == 1 and sim >= 0.72)
    )
    if not likely_echo:
        return t

    quote_line = ""
    for ln in t.splitlines():
        if (ln or "").strip().startswith("[FUN]"):
            quote_line = (ln or "").strip()
            break

    replacement = (
        "Heard you. I won't parrot you back.\n\n"
        "Pick one: banter, direct answer, or rewrite request."
    )
    return f"{quote_line}\n\n{replacement}".strip() if quote_line else replacement


def strip_irrelevant_proofread_tail(text: str, user_text: str) -> str:
    t = str(text or "").strip()
    u = " ".join(str(user_text or "").lower().split())
    if not t:
        return t
    asks_proofread = any(
        k in u for k in (
            "typo", "typos", "grammar", "proofread", "spelling", "factually incorrect", "sentiment analysis"
        )
    )
    if asks_proofread:
        return t
    lines: List[str] = []
    for ln in t.splitlines():
        low = " ".join((ln or "").lower().split())
        if not low:
            lines.append("")
            continue
        if "no typos were identified" in low:
            continue
        if "sentence structure is clear and grammatically sound" in low:
            continue
        lines.append(ln)
    out = "\n".join(lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out or t


def serious_max_tokens_for_query(user_text: str, base: int) -> int:
    u = " ".join(str(user_text or "").lower().split())
    long_proofread = (
        len(u) >= 700
        and any(k in u for k in ("sentiment analysis", "factually incorrect", "correct any typos", "proofread", "grammar"))
    )
    if long_proofread:
        return max(base, 640)
    return base


def normalize_agreement_ack_tense(text: str, user_text: str) -> str:
    t = str(text or "")
    u = " ".join(str(user_text or "").lower().split())
    if "?" in u:
        return t
    if not any(k in u for k in ("fair point", "i can see your perspective")):
        return t
    t = re.sub(r"\bYour statement was factually correct\b", "Your statement is factually correct", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwas valid and well-phrased\b", "is valid and well-phrased", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwas a core design principle\b", "is a core design principle", t, flags=re.IGNORECASE)
    return t

