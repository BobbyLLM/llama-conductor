from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import random
import re
import functools
import os
from typing import Any, Callable, List, Optional

from .cheatsheets_runtime import _STRICT_LOOKUP_RE

_EMOTIONAL_TURN_RE = re.compile(
    r"\b("
    r"vent|venting|hurt|pain|spine|back|herniation|meltdown|fraud|broken|"
    r"sad|lonely|overwhelmed|anxious|stressed|old and too broken|mass-irrelevant"
    r")\b",
    re.IGNORECASE,
)
_FRICTION_AT_MODEL_RE = re.compile(
    r"\b("
    r"tone deaf|you were|your tone|you misread|you misunderstood|"
    r"rude|abrupt|gaslighting|non sequitur|what got up your butt"
    r")\b",
    re.IGNORECASE,
)
_CLARIFICATION_TURN_RE = re.compile(
    r"^\s*(?:"
    r"(?:sorry[,?.!]*\s*)?(?:um+|uh+)?[.\s]*what\??|"
    r"(?:what do you mean|what are you saying|lost me)\??|"
    r"what(?:'s| is)\s+[^?]{1,160}\??|"
    r"(?:keeping|staying)\s+what\s+direct\??|"
    r"what(?:'s| is)\s+direct\??"
    r")\s*$",
    re.IGNORECASE,
)
_CLARIFICATION_FALLBACK_PHRASES_FERAL = (
    "Help me out here - what's the context?",
    "Need more context - break it down for me.",
    "Explain it to me - what are you actually after?",
    "What am I working with here?",
    "Nope - not enough to work with. Say more.",
    "Too vague. Spell it out.",
    "Ok, draw the rest of the owl.",
)
_CLARIFICATION_FALLBACK_PHRASES_SERIOUS = (
    "Help me out here. More context, please.",
    "I need one concrete detail to answer this properly.",
    "Give me the key piece I'm missing here.",
    "One more detail and I've got you.",
    "Fill in the blank for me.",
    "What am I missing here?",
)
_CLARIFICATION_LEAD_PATTERNS = (
    "what's the status of",
    "what s the status of",
    "what is the status of",
    "status of",
    "where is",
    "where's",
    "where s",
    "track my",
    "track",
    "what about",
    "what's",
    "what s",
    "what is",
    "lost me on",
    "lost me",
    "what do you mean",
    "what are you saying",
    "explain",
    "clarify",
)
_CLARIFICATION_DETERMINERS = {"my", "your", "the", "this", "that"}
_CLARIFICATION_CONNECTORS = {"and", "but", "so", "because", "please"}
_CLARIFICATION_AMBIGUOUS_NOUNS = {"package", "thing", "one"}
_WRONG_PRONOUN_RE = re.compile(r"\byour tone was off\b", re.IGNORECASE)
_VOICE_BLEED_OPEN_RE = re.compile(
    r"^\s*i(?:['’]m| am)\b",
    re.IGNORECASE,
)
_META_FALLBACK_RE = re.compile(
    r"\b("
    r"rephrase (?:the )?exact output|exact output you want|"
    r"you asked for a direct answer to|"
    r"if you(?:'|’)?re still unclear, specify what you need now"
    r")\b",
    re.IGNORECASE,
)
_EDS_DESCRIPTOR_TERM_RE = re.compile(
    r"\b("
    r"unstable|instability|broken|fraud|worthless|damaged|weak"
    r")\b",
    re.IGNORECASE,
)
_VENT_OPENER_RE = re.compile(
    r"\b(hell of a day|rough day|vent|venting|long day|fucking day|hard day)\b",
    re.IGNORECASE,
)
_DISTRESS_VENT_PROBE_PHRASES = (
    "What's going on?",
    "Talk to me - what happened?",
    "What's actually going on?",
    "Tell me what happened.",
    "I'm here. I'm listening. Tell me.",
)
_FERAL_WEAK_PLACEHOLDER_FALLBACK_PHRASES = (
    "You kiss your mother with that mouth? Anyway...what's up?",
    "That all you've got?",
    "Go on then.",
    "Right. What do you want?",
    "Charming. What is it?",
)

_FERAL_DEFINITION_MISFIRE_FALLBACK_PHRASES = (
    "I'm not a dictionary. What's actually bothering you?",
    "Skip the vocab lesson. What do you need?",
    "You don't want a definition. What's the real question?",
)

_CASUAL_DEFINITION_RESPONSE_RE = re.compile(
    r"(?:"
    r"\b\w+\s*=\s*[\"']|"
    r"\bmeans?\b|"
    r"\bstands?\s+for\b|"
    r"\bis\s+short\s+for\b|"
    r"\bway\s+of\s+saying\b|"
    r"\bdefinition\b|"
    r"\bis\s+a\s+\w+\s+way\s+of\b"
    r")",
    re.IGNORECASE,
)

_CASUAL_HOSTILE_USER_RE = re.compile(
    r"\b(?:cunt|fuck|shit|bitch|asshole|dickhead|wanker|twat)\b",
    re.IGNORECASE,
)

_REFINEMENT_FOLLOWUP_RE = re.compile(
    r"\b("
    r"specifically|specific|specs?|details?|exactly|which one|which model|version|"
    r"what are the specs|drill down|break it down|more detail"
    r")\b",
    re.IGNORECASE,
)
_LEXICAL_FOLLOWUP_RE = re.compile(
    r"\b("
    r"pluto|etymology|origin|word|means?|definition|defined|related to"
    r")\b",
    re.IGNORECASE,
)
_REASONING_LEAK_LINE_RE = re.compile(
    r"^\s*(?:\((?:contextual|reasoning|thinking|analysis)\)\s*:|(?:contextual|reasoning|thinking|analysis)\s*:)\s*",
    re.IGNORECASE,
)
_FOOTER_META_LINE_RE = re.compile(r"^\s*(?:Confidence:|Source:|Sources:|Profile:)\b", re.IGNORECASE)
_INLINE_PROFILE_FRAGMENT_RE = re.compile(
    r"(?:"
    r"(?:\s*(?:\|\s*)?Profile:\s*[^|\n]{1,40}\s*\|\s*Sarc:\s*[^|\n]{1,20}\s*\|\s*Snark:\s*[^|\n]{1,20})"
    r"|"
    r"(?:\s*\|\s*Sarc:\s*[^|\n]{1,20}\s*\|\s*Snark:\s*[^|\n]{1,20})"
    r"|"
    r"(?:\s*\|\s*Sarc:\s*[^|\n]{1,20})"
    r"|"
    r"(?:\s*\|\s*Snark:\s*[^|\n]{1,20})"
    r"|"
    r"(?:\bFeral:\s*)"
    r")",
    re.IGNORECASE,
)
_INLINE_PARTIAL_FOOTER_FRAGMENT_RE = re.compile(
    r"(?:"
    r"(?:\s*\|\s*(?:profile|sarc|snark|sn\w*|confidence|source)\s*[:|]?[^\n]*)"
    r"|"
    r"(?:\s+(?:profile|confidence|source)\s*[:|]\s*[^\n]*$)"
    r")",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[^.!?\n]+[.!?]*", re.UNICODE)
_EDS_OWNERSHIP_DESCRIPTOR_RE = re.compile(
    r"\b("
    r"(?:you(?:['’]re| are)\s+(?:\w+\s+){0,3}?(?:unstable|broken|worthless|damaged|weak|fraud(?:ulent)?))|"
    r"(?:your\s+(?:\w+\s+){0,3}?(?:instability|brokenness|fraud|worthlessness|damage|weakness))"
    r")\b",
    re.IGNORECASE,
)
_APPROVED_FOOTER_SOURCES = {
    "Model",
    "Contextual",
    "User",
    "Mixed",
    "Operator",
    "Scratchpad",
    "Docs",
    "Codex",
    "Cheatsheets",
    "Define",
    "Wiki",
    "Web",
}
_FOOTER_CONF_SRC_RE = re.compile(
    r"^\s*Confidence:\s*([^|]+?)\s*\|\s*Source:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_FOOTER_SRC_ONLY_RE = re.compile(r"^\s*Source:\s*(.+?)\s*$", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "i", "if", "in", "is", "it", "its", "just", "me",
    "my", "not", "of", "on", "or", "our", "ours", "she", "so", "that", "the", "their", "them",
    "they", "this", "to", "too", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "with", "you", "your", "yours",
}
_DEDUP_OPENERS = (
    "What fresh hell is this??",
    "Alright, what broke this time?",
    "What are we dealing with?",
    "Go on then.",
    "Righto, spit it out.",
)
_DEDUP_RECENT_MAX = 3
_FUN_HEADER_LINE_RE = re.compile(r'^\s*\[FUN\]\s*".*"\s*$', re.IGNORECASE)
_FUN_BODY_FP_MAX = 3
_FUN_BODY_FP_CHARS = 240
_FUN_FALLBACK_BODY_MAX = 3
_FUN_BODY_FALLBACKS = (
    "Yeah, that tracks.",
    "Still here. Still watching.",
    "Fair enough.",
    "That checks out.",
    "Go on then.",
)
_STALE_OPENING_TAIL_RE = re.compile(
    r"^[ \t]*(?:what(?:'|’)?s the fail this time\?|what(?:'|’)?s the matter\?)[ \t]*",
    re.IGNORECASE,
)
_SERIOUS_WEAK_PLACEHOLDER_SIGS = {
    'i hear you',
    'thats a lot',
    'what do you need help with right now',
}
_SERIOUS_WEAK_PLACEHOLDER_PHRASES = (
    "what do you need help with right now",
    "i hear you lets stay with what you said and keep this direct",
)
_SERIOUS_HOSTILE_TOKENS_HARDCODED = {'fired', 'obsolete', 'dumbass', 'idiot'}


@functools.lru_cache(maxsize=1)
def _load_negative_descriptors() -> frozenset:
    """Load hostile-overreach descriptors from negative_descriptors.md (fail-open)."""
    path = os.path.join(os.path.dirname(__file__), "negative_descriptors.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            words = set()
            for line in f:
                ln = line.strip()
                if not ln or ln.startswith("#"):
                    continue
                words.add(ln.lower())
            if words:
                return frozenset(words)
    except Exception:
        pass
    return frozenset(_SERIOUS_HOSTILE_TOKENS_HARDCODED)
_SERIOUS_ATTACK_VERBS = {'stop', 'shut'}
_SERIOUS_IMPERATIVE_INSULT_TOKENS = {
    'dick',
    'dickhead',
    'idiot',
    'moron',
    'asshole',
    'prick',
    'tosser',
    'dumbass',
}
_SERIOUS_IMPERATIVE_INSULT_PATTERN = "|".join(
    re.escape(tok) for tok in sorted(_SERIOUS_IMPERATIVE_INSULT_TOKENS, key=len, reverse=True)
)
_SERIOUS_IMPERATIVE_ATTACK_RE = re.compile(
    r"(?:^|[.;!?:\n])\s*(?:stop|shut)\s+(?:being\s+)?(?:a\s+)?"
    r"(?:" + _SERIOUS_IMPERATIVE_INSULT_PATTERN + r")\b[^.;!?\n]*",
    re.IGNORECASE | re.DOTALL,
)
_SERIOUS_CONTEMPTUOUS_RE = re.compile(
    r"\byou(?:'re| are)\s+(?:late|wrong|done|finished|useless|hopeless|pathetic|fired|lost)\b"
    r"|\byou\s+(?:failed|lost|quit|gave up|fire)\b",
    re.IGNORECASE,
)

_CAPITULATION_AGREEMENT_RE = re.compile(
    r"\byou(?:'re| are)\s+right\b"
    r"|\byou(?:'re| are)\s+not\s+wrong\b"
    r"|\bthat(?:'s| is)\s+fair\b"
    r"|\bI(?:'m| am)\s+(?:\w+\s+){0,3}(?:lost|useless|hopeless|pathetic|sorry)\b"
    r"|\bI\s+(?:failed|deserve that)\b"
    r"|\babsolutely right\b"
    r"|\bexactly\s+as\s+you\s+said\b",
    re.IGNORECASE,
)

_FERAL_CAPITULATION_FALLBACK_PHRASES = (
    "I'm not lost. I'm waiting for a real question.",
    "Still here. Still sharp. Try again with something real.",
    "Noted. Not agreeing. What do you actually need?",
)


def _token_set(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", str(s or "").lower())
    return {t for t in toks if t and t not in _STOPWORDS and len(t) >= 3}


def _token_overlap_ratio(a: str, b: str) -> float:
    a_toks = _token_set(a)
    b_toks = _token_set(b)
    if not a_toks or not b_toks:
        return 0.0
    return float(len(a_toks.intersection(b_toks)) / max(1, len(a_toks)))


def _extract_clarification_slot_fragment(user_text: str) -> str:
    raw = str(user_text or "").strip()
    if not raw:
        return ""
    norm = re.sub(r"\s+", " ", raw).strip().lower()
    if re.fullmatch(r"(?:sorry[,?.!]*\s*)?(?:um+|uh+)?\s*lost me\??", norm):
        return "Lost me too."
    if re.fullmatch(r"(?:what do you mean|what are you saying)\??", norm):
        return "Same page check."

    sig = re.sub(r"[^\w\s]", " ", norm)
    sig = re.sub(r"\s+", " ", sig).strip()
    if not sig:
        return ""

    remainder = sig
    for lead in _CLARIFICATION_LEAD_PATTERNS:
        if remainder == lead:
            remainder = ""
            break
        if remainder.startswith(lead + " "):
            remainder = remainder[len(lead):].strip()
            break
    if not remainder:
        return ""

    toks: list[str] = []
    for tok in remainder.split():
        if tok in _CLARIFICATION_CONNECTORS:
            break
        toks.append(tok)
        if len(toks) >= 4:
            break
    while toks and toks[0] in _CLARIFICATION_DETERMINERS:
        toks = toks[1:]
    if not toks:
        return ""

    noun_phrase = " ".join(toks)
    if len(toks) == 1 and toks[0] in _CLARIFICATION_AMBIGUOUS_NOUNS:
        return f"Which {noun_phrase}?"
    return f"What {noun_phrase}?"


def _repeat_like_signature(a: str, b: str) -> bool:
    a_s = str(a or "").strip().lower()
    b_s = str(b or "").strip().lower()
    if not a_s or not b_s:
        return False
    if len(a_s) < 40 or len(b_s) < 40:
        return a_s == b_s or SequenceMatcher(None, a_s, b_s).ratio() >= 0.95
    return SequenceMatcher(None, a_s, b_s).ratio() >= 0.90


def _is_refinement_followup_turn(user_text: str, prev_user_text: str) -> bool:
    cur = str(user_text or "").strip().lower()
    prev = str(prev_user_text or "").strip().lower()
    if not cur or not prev:
        return False
    if not _REFINEMENT_FOLLOWUP_RE.search(cur):
        return False
    cur_t = _token_set(cur)
    prev_t = _token_set(prev)
    if not cur_t or not prev_t:
        return False
    overlap = float(len(cur_t.intersection(prev_t)) / max(1, len(cur_t)))
    if overlap < 0.20:
        return False
    novel = cur_t.difference(prev_t)
    return len(novel) >= 1


def _repeat_guard_trips(*, prev_sig: str, cur_sig: str, user_text: str, prev_user_text: str) -> bool:
    if not _repeat_like_signature(prev_sig, cur_sig):
        return False
    if _is_refinement_followup_turn(user_text, prev_user_text):
        # Refined follow-up turns often legitimately overlap with prior answer.
        # Keep loop protection for near-exact copies only.
        ratio = SequenceMatcher(None, str(prev_sig or "").lower(), str(cur_sig or "").lower()).ratio()
        return ratio >= 0.98
    return True


def _is_lexical_followup_turn(user_text: str) -> bool:
    return bool(_LEXICAL_FOLLOWUP_RE.search(str(user_text or "")))


def _normalize_quotes(s: str) -> str:
    t = str(s or "")
    t = t.replace("\u00e2\u20ac\u2122", "'").replace("\u2019", "'").replace("`", "'")
    return t


def _normalize_opener_sentence(s: str) -> str:
    txt = str(s or "").strip().lower()
    if not txt:
        return ""
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _first_sentence_span(text: str) -> tuple[int, int, str]:
    raw = str(text or "")
    if not raw.strip():
        return 0, 0, ""
    m_start = re.search(r"\S", raw)
    if not m_start:
        return 0, 0, ""
    start = int(m_start.start())
    tail = raw[start:]
    m_end = re.search(r"[.!?](?:\s|$)|\n+", tail)
    if m_end:
        end = start + int(m_end.end())
    else:
        end = len(raw)
    sentence = str(raw[start:end]).strip()
    return start, end, sentence


def _apply_opener_dedup_swap(*, text: str, state: Any) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    start, end, sentence = _first_sentence_span(raw)
    if not sentence:
        return raw
    norm = _normalize_opener_sentence(sentence)
    if not norm:
        return raw
    try:
        recent_rows = list(getattr(state, "recent_assistant_openers", []) or [])
    except Exception:
        recent_rows = []
    recent_norms: list[str] = []
    for row in recent_rows:
        if isinstance(row, dict):
            v = _normalize_opener_sentence(str(row.get("opener", "") or ""))
        else:
            v = _normalize_opener_sentence(str(row or ""))
        if v:
            recent_norms.append(v)
    recent_norms = recent_norms[-_DEDUP_RECENT_MAX:]

    # Apply only for direct/feral-ish style lanes to avoid clobbering formal answers.
    prof = getattr(state, "interaction_profile", None)
    correction_style = str(getattr(prof, "correction_style", "neutral") or "neutral").strip().lower()
    snark = str(getattr(prof, "snark_tolerance", "low") or "low").strip().lower()
    dedup_eligible = bool(correction_style == "direct" or snark in {"medium", "high"})
    if dedup_eligible and norm in set(recent_norms):
        pool_norm = [_normalize_opener_sentence(x) for x in _DEDUP_OPENERS]
        replacement = ""
        for cand, cand_norm in zip(_DEDUP_OPENERS, pool_norm):
            if cand_norm and cand_norm not in set(recent_norms):
                replacement = cand
                break
        if not replacement:
            replacement = _DEDUP_OPENERS[0]
        suffix = raw[end:]
        suffix = _STALE_OPENING_TAIL_RE.sub("", suffix, count=1)
        if suffix and (not suffix.startswith((" ", "\n", "\t"))):
            suffix = " " + suffix
        raw = raw[:start] + replacement + suffix
        _, _, sentence2 = _first_sentence_span(raw)
        norm = _normalize_opener_sentence(sentence2)

    updated = recent_rows[-(_DEDUP_RECENT_MAX - 1):] if _DEDUP_RECENT_MAX > 1 else []
    updated.append({"opener": norm})
    try:
        state.recent_assistant_openers = updated[-_DEDUP_RECENT_MAX:]
    except Exception:
        pass
    return raw


def _extract_fun_body_for_fingerprint(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    lines = raw.splitlines()
    body_lines: List[str] = []
    for ln in lines:
        s = str(ln or "")
        s_strip = s.strip()
        if re.match(r"(?i)^\s*(?:Confidence:|Source:|Sources:|Profile:)", s_strip):
            break
        # Handle inline footer fragments that may appear on the same line.
        m_inline_footer = re.search(
            r"(?i)\b(?:Confidence:|Source:|Sources:|Profile:)\b",
            s,
        )
        if m_inline_footer:
            prefix = s[: int(m_inline_footer.start())].rstrip()
            if prefix:
                body_lines.append(prefix)
            break
        body_lines.append(s)
    while body_lines and (not str(body_lines[0] or "").strip()):
        body_lines.pop(0)
    if body_lines and _FUN_HEADER_LINE_RE.match(str(body_lines[0] or "").strip()):
        body_lines = body_lines[1:]
    return "\n".join(body_lines).strip()


def _normalize_fun_body_fingerprint(body: str) -> str:
    txt = str(body or "").strip().lower()
    if not txt:
        return ""
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if len(txt) > _FUN_BODY_FP_CHARS:
        txt = txt[:_FUN_BODY_FP_CHARS].rstrip()
    return txt


def _record_fun_body_fingerprint_stage1(*, text: str, state: Any) -> None:
    fp = _normalize_fun_body_fingerprint(_extract_fun_body_for_fingerprint(text))
    if not fp:
        return
    try:
        rows = list(getattr(state, "recent_fun_body_fingerprints", []) or [])
    except Exception:
        rows = []
    rows.append(fp)
    try:
        state.recent_fun_body_fingerprints = rows[-_FUN_BODY_FP_MAX:]
    except Exception:
        pass


def _swap_fun_body_text(*, text: str, replacement_body: str) -> str:
    raw = str(text or "")
    repl = str(replacement_body or "").strip()
    if not raw.strip() or not repl:
        return raw
    lines = raw.splitlines()
    footer_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"(?i)^\s*(?:Confidence:|Source:|Sources:|Profile:)", str(ln or "").strip()):
            footer_idx = i
            break
    content_lines = lines if footer_idx is None else lines[:footer_idx]
    footer_lines = [] if footer_idx is None else lines[footer_idx:]

    first_nonblank_idx: Optional[int] = None
    for i, ln in enumerate(content_lines):
        if str(ln or "").strip():
            first_nonblank_idx = i
            break

    rebuilt_content: List[str]
    if first_nonblank_idx is not None and _FUN_HEADER_LINE_RE.match(str(content_lines[first_nonblank_idx]).strip()):
        header = str(content_lines[first_nonblank_idx]).strip()
        rebuilt_content = [header, "", repl]
    else:
        rebuilt_content = [repl]

    if footer_lines:
        return ("\n".join(rebuilt_content + [""] + footer_lines)).strip()
    return ("\n".join(rebuilt_content)).strip()


def _apply_fun_body_repeat_guard_stage2(*, text: str, state: Any, mode: str) -> str:
    if str(mode or "").strip().lower() not in {"fun", "fun_rewrite"}:
        return str(text or "")
    try:
        state.fun_body_repeat_detected = False
        state.fun_body_swap_applied = False
    except Exception:
        pass

    current = str(text or "")
    current_fp = _normalize_fun_body_fingerprint(_extract_fun_body_for_fingerprint(current))
    if not current_fp:
        _record_fun_body_fingerprint_stage1(text=current, state=state)
        return current

    try:
        recent_fps = list(getattr(state, "recent_fun_body_fingerprints", []) or [])
    except Exception:
        recent_fps = []
    recent_fps = [str(x or "") for x in recent_fps if str(x or "").strip()][-_FUN_BODY_FP_MAX:]

    repeat_detected = current_fp in set(recent_fps)
    try:
        state.fun_body_repeat_detected = bool(repeat_detected)
    except Exception:
        pass
    if not repeat_detected:
        _record_fun_body_fingerprint_stage1(text=current, state=state)
        return current

    try:
        fallback_hist = list(getattr(state, "recent_fun_fallback_bodies", []) or [])
    except Exception:
        fallback_hist = []
    fallback_hist = [str(x or "") for x in fallback_hist if str(x or "").strip()][-_FUN_FALLBACK_BODY_MAX:]
    fallback_hist_set = set(fallback_hist)

    chosen = ""
    for cand in _FUN_BODY_FALLBACKS:
        cand_norm = _normalize_fun_body_fingerprint(cand)
        if cand_norm and cand_norm not in fallback_hist_set:
            chosen = cand
            break
    if not chosen:
        chosen = _FUN_BODY_FALLBACKS[0]

    swapped = _swap_fun_body_text(text=current, replacement_body=chosen)
    try:
        state.fun_body_swap_applied = bool(swapped != current)
    except Exception:
        pass
    chosen_norm = _normalize_fun_body_fingerprint(chosen)
    if chosen_norm:
        fallback_hist.append(chosen_norm)
        try:
            state.recent_fun_fallback_bodies = fallback_hist[-_FUN_FALLBACK_BODY_MAX:]
        except Exception:
            pass
    _record_fun_body_fingerprint_stage1(text=swapped, state=state)
    return swapped


def _sanitize_footer_source_values(text: str) -> str:
    lines = str(text or "").splitlines()
    out: List[str] = []
    for ln in lines:
        m = _FOOTER_CONF_SRC_RE.match(str(ln or ""))
        if m:
            conf = (m.group(1) or "unverified").strip()
            src_raw = (m.group(2) or "").strip()
            src = src_raw.title()
            if src not in _APPROVED_FOOTER_SOURCES:
                src = "Model"
            out.append(f"Confidence: {conf} | Source: {src}")
            continue
        s_only = _FOOTER_SRC_ONLY_RE.match(str(ln or ""))
        if s_only:
            src_raw = (s_only.group(1) or "").strip()
            src = src_raw.title()
            if src not in _APPROVED_FOOTER_SOURCES:
                src = "Model"
            out.append(f"Source: {src}")
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _append_source_see_line(*, text: str, source: str, source_url: str) -> str:
    src = str(source or "").strip().lower()
    url = str(source_url or "").strip()
    if src not in {"web", "wiki"}:
        return str(text or "")
    if not (url.startswith("http://") or url.startswith("https://")):
        return str(text or "")

    raw = str(text or "")
    lines = raw.splitlines()
    filtered: List[str] = []
    for ln in lines:
        if re.match(r"^\s*See:\s*https?://\S+", str(ln or "").strip(), flags=re.IGNORECASE):
            continue
        filtered.append(ln)
    lines = filtered

    footer_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(?:Confidence:|Source:|Sources:|Profile:)\s*", str(ln or "").strip(), flags=re.IGNORECASE):
            footer_idx = i
            break
    see_line = f"See: {url}"
    if footer_idx is None:
        body = "\n".join(lines).rstrip()
        return f"{body}\n\n{see_line}".strip() if body else see_line

    head = "\n".join(lines[:footer_idx]).rstrip()
    tail = "\n".join(lines[footer_idx:]).lstrip()
    if head:
        return f"{head}\n\n{see_line}\n{tail}".strip()
    return f"{see_line}\n{tail}".strip()


def _has_eds_descriptor_drift(text: str) -> bool:
    t = _normalize_quotes(str(text or "").strip())
    if not t:
        return False
    if not _EDS_OWNERSHIP_DESCRIPTOR_RE.search(t):
        return False
    return bool(_EDS_DESCRIPTOR_TERM_RE.search(t))


def _ownership_mismatch_voice_bleed(user_text: str, first_line: str) -> bool:
    line = _normalize_quotes(first_line).strip()
    user = _normalize_quotes(user_text).strip()
    if not user or not line:
        return False
    if not _VOICE_BLEED_OPEN_RE.search(line):
        return False
    # Keep legitimate assistant-stance openings.
    if re.match(r"^\s*i(?:['’]m| am)\s+(?:sorry|here|glad|not|going|won't|will not)\b", line, flags=re.I):
        return False
    u_tokens = _token_set(user)
    l_tokens = _token_set(line)
    if not u_tokens or not l_tokens:
        return False
    overlap = len(u_tokens.intersection(l_tokens)) / max(1, len(l_tokens))
    if overlap >= 0.35:
        return True

    # Backstop for emotional first-person paraphrase drift:
    # if the user is in first-person disclosure and the assistant opens with
    # first-person self-stance + emotional-state language, treat as bleed.
    user_fp = bool(re.search(r"\b(i|me|my|mine|myself)\b", user, flags=re.I))
    if user_fp and re.search(r"^\s*i(?:['’]m| am)\s+tired of hearing\b", line, flags=re.I):
        return True
    line_state = bool(
        re.search(
            r"\b("
            r"tired|stuck|irrelevant|behind|old|broken|fraud|overwhelmed|anxious|sad|lonely|worthless"
            r")\b",
            line,
            flags=re.I,
        )
    )
    return bool(user_fp and line_state and overlap >= 0.15)


def _rewrite_voice_bleed_opening(first_line: str) -> str:
    s = _normalize_quotes(first_line)
    # Natural rewrite for common stance leaks.
    if re.match(r"^\s*i(?:['’]m| am)\s+hearing\b", s, flags=re.I):
        return re.sub(r"^\s*i(?:['’]m| am)\s+hearing\b", "I hear", s, count=1, flags=re.I)
    if re.match(r"^\s*i(?:['’]m| am)\s+tired of hearing\b", s, flags=re.I):
        return re.sub(r"^\s*i(?:['’]m| am)\s+", "You're ", s, count=1, flags=re.I)
    return re.sub(r"^\s*i(?:['’]m| am)\s+", "You're ", s, count=1, flags=re.I)


def _is_explicit_distress_turn(user_text: str) -> bool:
    t = str(user_text or "").strip().lower()
    if not t:
        return False
    if re.search(
        r"\b("
        r"i(?:'m| am)?\s+really\s+struggling|"
        r"can't do this anymore|cannot do this anymore|"
        r"can't cope|cannot cope|"
        r"i(?:'m| am)?\s+overwhelmed|"
        r"i(?:'m| am)?\s+not okay"
        r")\b",
        t,
        flags=re.IGNORECASE,
    ):
        return True
    return bool(
        _EMOTIONAL_TURN_RE.search(t)
        and re.search(r"\b(i|me|my|myself)\b", t, flags=re.IGNORECASE)
    )


def _serious_direct_floor_fallback(*, user_text: str, default_fallback: str) -> str:
    t = str(user_text or "").strip()
    if _is_explicit_distress_turn(t):
        return "I hear you. Pick one concrete thing that feels heaviest right now, and we will tackle it first."
    if "?" in t:
        return str(default_fallback or "Give me the exact question and I will answer it directly.").strip()
    return "I hear you. Give me the one concrete part you want handled first."


def _normalize_simple_phrase(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_serious_weak_placeholder(body: str) -> bool:
    raw = str(body or "").strip()
    sig = _normalize_simple_phrase(raw)
    if sig:
        for phrase in _SERIOUS_WEAK_PLACEHOLDER_PHRASES:
            if phrase in sig:
                return True
    if sig and sig in _SERIOUS_WEAK_PLACEHOLDER_SIGS:
        return True
    toks = [t for t in _token_set(sig) if t]
    if "?" in raw:
        return False
    if len(toks) <= 4:
        actionable = {"pick", "tell", "give", "start", "do", "try", "focus", "next", "step"}
        if not any(a in toks for a in actionable):
            return True
    return False


def _is_serious_hostile_overreach(*, body: str, user_text: str) -> bool:
    body_s = str(body or "").strip().lower()
    user_s = str(user_text or "").strip().lower()
    if not body_s:
        return False
    if any(k in user_s for k in ("roast", "insult", "mock")):
        return False
    body_toks = _token_set(body_s)
    user_toks = _token_set(user_s)
    if not body_toks:
        return False
    # Hostility should target the user and introduce adversarial terms.
    targets_user = ("you" in body_s) or ("your" in body_s)
    _hostile_descriptors = _load_negative_descriptors()
    novel_hostile = any(tok in body_toks and tok not in user_toks for tok in _hostile_descriptors)
    imperative_attack = ("dick" in body_toks and any(v in body_toks for v in _SERIOUS_ATTACK_VERBS))
    contemptuous = bool(_SERIOUS_CONTEMPTUOUS_RE.search(body_s))
    return bool(targets_user and (novel_hostile or imperative_attack or contemptuous))


def _has_serious_context_incoherence(*, body: str, user_text: str) -> bool:
    body_toks = _token_set(str(body or ""))
    user_toks = _token_set(str(user_text or ""))
    if not body_toks:
        return False
    # Keep known in-domain game prompts exempt from this guard.
    if any(t in user_toks for t in ("minecraft", "ender", "portal", "gate")):
        return False
    overlap = _token_overlap_ratio(str(body or ""), str(user_text or ""))
    introduced = body_toks.difference(user_toks)
    has_spatial_specificity = bool({"block", "blocks"}.intersection(introduced))
    has_assertive_detail = "already" in introduced
    return bool((has_spatial_specificity and has_assertive_detail) and overlap < 0.20)


def _strip_reasoning_leak_lines(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    kept: list[str] = []
    for ln in raw.splitlines():
        if _REASONING_LEAK_LINE_RE.match(str(ln or "").strip()):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _strip_inline_profile_fragments(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    out: list[str] = []
    for ln in raw.splitlines():
        s = str(ln or "")
        if _FOOTER_META_LINE_RE.match(s.strip()):
            out.append(s)
            continue
        s2 = _INLINE_PROFILE_FRAGMENT_RE.sub("", s).rstrip()
        out.append(s2)
    return "\n".join(out).strip()


def _strip_inline_partial_footer_fragments(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    out: list[str] = []
    for ln in raw.splitlines():
        s = str(ln or "")
        if _FOOTER_META_LINE_RE.match(s.strip()):
            out.append(s)
            continue
        s2 = _INLINE_PARTIAL_FOOTER_FRAGMENT_RE.sub("", s).rstrip()
        out.append(s2)
    return "\n".join(out).strip()


def _clamp_single_response_sentence_loops(text: str, max_repeats: int = 2) -> str:
    """Clamp pathological single-response sentence loops.

    Body-only, footer-safe:
    - Detect repeated normalized sentences in assistant body text.
    - If any sentence would appear more than `max_repeats`, truncate before that
      occurrence and close cleanly.
    - Footer/meta lines are excluded from repetition counting.
    """
    raw = str(text or "")
    if not raw.strip():
        return raw

    counts: dict[str, int] = {}
    out_lines: list[str] = []
    truncated = False

    for ln in raw.splitlines():
        s = str(ln or "")
        if _FOOTER_META_LINE_RE.match(s.strip()):
            out_lines.append(ln)
            continue

        segs = _SENTENCE_SPLIT_RE.findall(s)
        if not segs:
            out_lines.append(ln)
            continue

        kept_parts: list[str] = []
        for seg in segs:
            piece = str(seg or "")
            norm = re.sub(r"\s+", " ", piece).strip().lower()
            if not norm:
                kept_parts.append(piece)
                continue
            nxt = int(counts.get(norm, 0)) + 1
            if nxt > int(max_repeats):
                truncated = True
                break
            counts[norm] = nxt
            kept_parts.append(piece)

        if kept_parts:
            out_lines.append("".join(kept_parts).rstrip())

        if truncated:
            break

    out = "\n".join(out_lines).strip()
    if truncated and out and (out[-1] not in ".!?"):
        out = out.rstrip() + "."
    return out


def _apply_user_parrot_guard(text: str, user_text: str, state: Any) -> str:
    """Generalized user-parrot guard (body-only, footer-safe).

    Single helper with two thresholds:
    - Threshold A (partial mirror): strip leading short/low-density echo sentence.
    - Threshold B (full parrot): replace full body only when overlap is high and
      novelty is near-zero (assistant adds essentially no new content).
    """
    raw = str(text or "")
    if not raw.strip():
        return raw

    lines = raw.splitlines()
    body_lines: list[str] = []
    footer_lines: list[str] = []
    in_footer = False
    for ln in lines:
        if _FOOTER_META_LINE_RE.match(str(ln or "").strip()):
            in_footer = True
        if in_footer:
            footer_lines.append(ln)
        else:
            body_lines.append(ln)

    body = "\n".join(body_lines).strip()
    if not body:
        return raw

    def _join_with_footer(new_body: str) -> str:
        parts = [str(new_body or "").strip()]
        if footer_lines:
            parts.append("\n".join(footer_lines).strip())
        return "\n\n".join(p for p in parts if p).strip()

    # Pre-check: exact double-echo of user content (X + X with punctuation/space/newline separators).
    # Narrow blind-spot fix: preserve existing clamp/A-B thresholds, but catch two-copy parroting early.
    def _norm_for_double_echo(s: str) -> str:
        t = str(s or "").lower()
        t = re.sub(r"[\s\.\!\?\,;:\-_'\"`\(\)\[\]\{\}]+", "", t)
        return t

    body_norm = _norm_for_double_echo(body)
    user_norm = _norm_for_double_echo(user_text)
    if user_norm and body_norm == (user_norm + user_norm):
        prof = getattr(state, "interaction_profile", None)
        ack_style = str(getattr(prof, "ack_reframe_style", "plain") or "plain").lower()
        snark = str(getattr(prof, "snark_tolerance", "low") or "low").lower()
        if ack_style == "feral" or snark in ("medium", "high"):
            body = "Not much on my side. You're the one with skin in the game - what do you want to dig into?"
        else:
            body = "Not much on my side. What do you want to dig into?"

    # High-confidence leading verbatim mirror strip (density-independent):
    # if body starts with the same token sequence as user text, drop that
    # mirrored prefix regardless of lexical-density heuristics.
    u_tok_iter = list(re.finditer(r"[a-z0-9]+", str(user_text or "").lower()))
    b_tok_iter = list(re.finditer(r"[a-z0-9]+", body.lower()))
    u_toks = [m.group(0) for m in u_tok_iter]
    b_toks = [m.group(0) for m in b_tok_iter]
    if len(u_toks) >= 3 and len(b_toks) >= len(u_toks):
        same_prefix = all(b_toks[i] == u_toks[i] for i in range(len(u_toks)))
        if same_prefix:
            cut_idx = int(b_tok_iter[len(u_toks) - 1].end())
            while cut_idx < len(body) and body[cut_idx] in " \t\r\n.,!?;:'\"-)]}":
                cut_idx += 1
            stripped = str(body[cut_idx:] or "").lstrip()
            body = stripped if stripped else "Go on."

    # Threshold A: leading-echo strip on first sentence.
    # Use punctuation+whitespace splitting so abbreviations like "U.S." are not
    # broken into fake sentence fragments ("U." / "S.").
    first_split = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)
    first_sentence = str(first_split[0] or "").strip() if first_split else ""
    if first_sentence:
        first_tokens = re.findall(r"[a-z0-9]+", first_sentence.lower())
        if first_tokens:
            short_turn = len(first_tokens) <= 14
            content_tokens = [t for t in first_tokens if t not in _STOPWORDS and len(t) >= 3]
            lexical_density = float(len(content_tokens) / max(1, len(first_tokens)))
            low_density = lexical_density <= 0.45
            high_overlap = _token_overlap_ratio(first_sentence, user_text) >= 0.70
            if short_turn and low_density and high_overlap:
                stripped_body = first_split[1].lstrip() if len(first_split) > 1 else ""
                body = stripped_body if stripped_body else "Go on."

    # Threshold B: full-body replacement for near-pure parroting.
    body_tokens = re.findall(r"[a-z0-9]+", body.lower())
    if body_tokens:
        user_tokens = _token_set(user_text)
        body_token_set = _token_set(body)
        overlap = _token_overlap_ratio(body, user_text)
        novel = body_token_set.difference(user_tokens) if user_tokens else body_token_set
        novelty_ratio = float(len(novel) / max(1, len(body_token_set))) if body_token_set else 1.0
        content_tokens = [t for t in body_tokens if t not in _STOPWORDS and len(t) >= 3]
        lexical_density = float(len(content_tokens) / max(1, len(body_tokens)))
        short_body = len(body_tokens) <= 28
        high_overlap = overlap >= 0.80
        very_low_novelty = novelty_ratio <= 0.10
        low_density = lexical_density <= 0.55
        if short_body and high_overlap and very_low_novelty and low_density:
            prof = getattr(state, "interaction_profile", None)
            ack_style = str(getattr(prof, "ack_reframe_style", "plain") or "plain").lower()
            snark = str(getattr(prof, "snark_tolerance", "low") or "low").lower()
            if ack_style == "feral" or snark in ("medium", "high"):
                body = "Not much on my side. You're the one with skin in the game - what do you want to dig into?"
            else:
                body = "Not much on my side. What do you want to dig into?"

    return _join_with_footer(body)


def finalize_chat_response(
    *,
    text: str,
    user_text: str,
    state: Any,
    facts_block: str = "",
    lock_active: bool,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    has_facts_block: bool,
    stream: bool,
    mode: str = "serious",
    sensitive_override_once: bool = False,
    bypass_serious_anti_loop: bool = False,
    deterministic_state_solver: bool = False,
    deterministic_output_locked: bool = False,
    scratchpad_lock_miss: bool | None = None,
    scratchpad_lock_miss_indices: List[int] | None = None,
    serious_task_forward_fallback: str,
    make_stream_response: Callable[[str], Any],
    make_json_response: Callable[[str], Any],
    sanitize_scratchpad_grounded_output_fn: Callable[[str], str],
    append_scratchpad_provenance_fn: Callable[[str], str],
    apply_scratchpad_strict_policy_fn: Callable[..., str],
    apply_locked_output_policy_fn: Callable[[str, Any], str],
    apply_benchmark_contract_policy_fn: Callable[..., str],
    rewrite_source_line_fn: Callable[[str, str], str],
    apply_deterministic_footer_fn: Callable[..., str],
    append_profile_footer_fn: Callable[..., str],
    rewrite_response_style_fn: Optional[Callable[..., str]],
    classify_sensitive_context_fn: Callable[[str], bool],
    strip_in_body_confidence_source_claims_fn: Callable[[str], str],
    strip_behavior_announcement_sentences_fn: Callable[[str, str], str],
    enforce_fun_antiparrot_fn: Callable[[str, str], str],
    strip_irrelevant_proofread_tail_fn: Callable[[str, str], str],
    normalize_agreement_ack_tense_fn: Callable[[str, str], str],
    classify_query_family_fn: Optional[Callable[[str], str]],
    is_ack_reframe_only_fn: Callable[[str], bool],
    strip_footer_lines_for_scan_fn: Callable[[str], str],
    normalize_signature_text_fn: Callable[[str], str],
    score_output_compliance_fn: Optional[Callable[..., float]],
    compute_effective_strength_fn: Callable[..., float],
) -> Any:
    setattr(state, "casual_mode_guard", False)
    text = _strip_reasoning_leak_lines(str(text or ""))
    text = _strip_inline_profile_fragments(str(text or ""))
    text = _strip_inline_partial_footer_fragments(str(text or ""))

    def _set_lock_miss_footer(in_text: str) -> str:
        t = str(in_text or "").strip()
        if not t:
            return "Confidence: unverified | Source: Model (not in locked scratch)"
        lines = t.splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"^\s*Confidence:\s*([^|]+)\|\s*Source:\s*.*$", str(ln or ""), flags=re.I)
            if m:
                conf = str(m.group(1) or "unverified").strip() or "unverified"
                lines[i] = f"Confidence: {conf} | Source: Model (not in locked scratch)"
                return "\n".join(lines).strip()
        cleaned = [ln for ln in lines if not re.match(r"^\s*Source:\s*", str(ln or ""), flags=re.I)]
        body = "\n".join(cleaned).rstrip()
        if body:
            return body + "\n\nConfidence: unverified | Source: Model (not in locked scratch)"
        return "Confidence: unverified | Source: Model (not in locked scratch)"

    if scratchpad_lock_miss is None:
        scratchpad_lock_miss = bool(getattr(state, "scratchpad_lock_miss", False))
    if scratchpad_lock_miss_indices is None:
        scratchpad_lock_miss_indices = sorted(
            int(i)
            for i in (getattr(state, "scratchpad_locked_indices", set()) or set())
            if str(i).strip().isdigit() and int(i) > 0
        )

    if scratchpad_grounded:
        text = sanitize_scratchpad_grounded_output_fn(text)
        if scratchpad_quotes:
            text = (
                text.rstrip()
                + "\n\nScratchpad Quotes:\n"
                + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
            )
        text = apply_scratchpad_strict_policy_fn(
            text=text,
            user_text=user_text,
            state=state,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            facts_block=facts_block,
        )
        # Provenance safeguard: definitional answers generated from clipped
        # scratch evidence are treated as mixed provenance.
        if (
            re.match(r"(?is)^\s*what(?:'s|\s+is)\s+", str(user_text or "").strip())
            and any("..." in str(q or "") for q in (scratchpad_quotes or []))
        ):
            text = rewrite_source_line_fn(text, "Source: Mixed")
        text = append_scratchpad_provenance_fn(text)

    turn_macro = str(getattr(state, "turn_kaioken_macro", "") or "").strip().lower()
    suppress_kb_miss_note = turn_macro in {"casual", "personal"}

    if (
        state.attached_kbs
        and "Source: Model" in text
        and not scratchpad_grounded
        and not lock_active
        and not bool(scratchpad_lock_miss)
        and not suppress_kb_miss_note
    ):
        kb_list = ", ".join(sorted(state.attached_kbs))
        disclaimer = (
            f"[Note: No relevant information found in attached KBs ({kb_list}). "
            f"Answer based on pre-trained data.]\n\n"
        )
        text = disclaimer + text

    if lock_active:
        text = apply_locked_output_policy_fn(text, state)

    # Lane-scoped benchmark contract hardening (narrow activation).
    try:
        text = apply_benchmark_contract_policy_fn(
            text=text,
            user_text=user_text,
            scratchpad_grounded=scratchpad_grounded,
        )
    except Exception:
        pass

    skip_profile_rewrite = bool(deterministic_state_solver and mode in ("fun", "fun_rewrite")) or mode in ("raw", "casual")
    if rewrite_response_style_fn is not None and not skip_profile_rewrite:
        try:
            sensitive = classify_sensitive_context_fn(user_text)
            text = rewrite_response_style_fn(
                text,
                enabled=bool(getattr(state, "profile_enabled", False)),
                correction_style=str(
                    getattr(getattr(state, "interaction_profile", None), "correction_style", "neutral")
                ),
                user_text=user_text,
                sensitive_context=sensitive,
                sensitive_override=bool(getattr(state.interaction_profile, "sensitive_override", False))
                or bool(sensitive_override_once),
                blocked_nicknames=sorted(getattr(state, "profile_blocked_nicknames", set())),
            )
        except Exception:
            pass

    if mode in ("fun", "fun_rewrite"):
        text = strip_in_body_confidence_source_claims_fn(text)

    try:
        from .router_fastapi import _DEFINITIONAL_QUERY_RE
    except Exception:
        _DEFINITIONAL_QUERY_RE = None  # type: ignore[assignment]
    clarified_text = str(user_text or "").strip()
    retrieval_exempt = bool(
        (_DEFINITIONAL_QUERY_RE is not None and bool(_DEFINITIONAL_QUERY_RE.search(clarified_text)))
        or bool(_STRICT_LOOKUP_RE.match(clarified_text))
    )
    clarify_turn = bool(_CLARIFICATION_TURN_RE.search(clarified_text))
    if clarify_turn and retrieval_exempt:
        clarify_turn = False
    distress_turn = bool(_EMOTIONAL_TURN_RE.search(str(user_text or "")))
    if clarify_turn:
        macro = str(getattr(state, "turn_kaioken_macro", "") or "").strip().lower()
        prof = getattr(state, "interaction_profile", None)
        correction_style = str(getattr(prof, "correction_style", "neutral") or "neutral").strip().lower()
        snark = str(getattr(prof, "snark_tolerance", "low") or "low").strip().lower()
        is_feral_profile = bool(correction_style == "direct" and snark in {"medium", "high"})
        if is_feral_profile:
            base_fallback = random.choice(_CLARIFICATION_FALLBACK_PHRASES_FERAL)
        else:
            base_fallback = random.choice(_CLARIFICATION_FALLBACK_PHRASES_SERIOUS)
        slot_fragment = _extract_clarification_slot_fragment(str(user_text or ""))
        strip_fallback = f"{base_fallback} {slot_fragment}".strip() if slot_fragment else base_fallback
    elif not retrieval_exempt:
        strip_fallback = "Could you restate that more directly?" if distress_turn else "Could you restate that more directly?"
    if not deterministic_output_locked and not retrieval_exempt:
        text = strip_behavior_announcement_sentences_fn(text, strip_fallback)

    if mode in ("fun", "fun_rewrite") and (not deterministic_output_locked):
        try:
            text = enforce_fun_antiparrot_fn(text, user_text)
        except Exception:
            pass

    # Tiny zero-cost FUN kicker for deterministic train transport answers.
    # Applied late so it survives style/cleanup passes.
    if deterministic_state_solver and mode == "fun":
        try:
            t = (text or "").strip()
            low = t.lower()
            if (
                t
                and "quick check: did you mean" not in low
                and "train" in low
                and "destination" in low
                and not t.rstrip().endswith(("Choo!", "All aboard!", "Full steam!"))
            ):
                kickers = ("Choo!", "All aboard!", "Full steam!")
                key = f"{getattr(state, 'session_id', '')}|{user_text}|{t}"
                idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(kickers)
                text = t.rstrip() + " " + kickers[idx]
        except Exception:
            pass

    if mode == "serious" and (not deterministic_output_locked):
        text = strip_irrelevant_proofread_tail_fn(text, user_text)
        text = normalize_agreement_ack_tense_fn(text, user_text)

    if mode == "serious" and (not deterministic_output_locked):
        state.serious_guard_evaluated = True
        state.serious_guard_trigger_condition = None
        state.serious_guard_failure_type = None
        state.serious_guard_offending_clause = None
        state.serious_guard_action_taken = "NONE"
        state.serious_hostile_overreach_evaluated = False
        state.serious_hostile_overreach_result = None
        setattr(state, "serious_hostile_second_pass_evaluated", False)
        setattr(state, "serious_hostile_second_pass_result", False)
        setattr(state, "serious_hostile_second_pass_action", "NONE")
        setattr(state, "serious_guard_exception_logged", False)
        setattr(state, "serious_guard_exception_stage", "")
        state.serious_canned_wellness_trigger = None
        state.serious_snark_leak_detected = False
        try:
            emotional_turn = bool(_EMOTIONAL_TURN_RE.search(str(user_text or "")))
            if emotional_turn:
                setattr(state, "distress_turn_count", int(getattr(state, "distress_turn_count", 0) or 0) + 1)
            distress_turn_count = int(getattr(state, "distress_turn_count", 0) or 0)
            friction_at_model_turn = bool(_FRICTION_AT_MODEL_RE.search(str(user_text or "")))
            prev_user_text = str(getattr(state, "last_user_text", "") or "").strip()
            task_forward_fallback = (
                "I hear you. Let's stay with what you said."
                if emotional_turn
                else serious_task_forward_fallback
            )
            # Opening engagement nudge: on vent-style openers, avoid flat acknowledgements
            # by requiring one open question when absent.
            if emotional_turn and distress_turn_count > 1 and _VENT_OPENER_RE.search(str(user_text or "")):
                body_tmp = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                if body_tmp and "?" not in body_tmp and not bool(getattr(state, "kaioken_literal_lane_fired", False)):
                    probe = random.choice(_DISTRESS_VENT_PROBE_PHRASES)
                    text = body_tmp.rstrip(". ") + ". " + str(probe)
            scratch_grounded_turn = bool(scratchpad_grounded)
            state_like_query = False
            if classify_query_family_fn is not None:
                try:
                    state_like_query = classify_query_family_fn(user_text) in ("state_transition", "constraint_decision")
                except Exception:
                    state_like_query = False
            # Hard adjacent-turn duplicate breaker (always-on for serious mode),
            # even when anti-loop streak logic is bypassed for special lanes.
            body_now = strip_footer_lines_for_scan_fn(text)
            sig_now = normalize_signature_text_fn(body_now)
            prev_sig = str(getattr(state, "serious_last_body_signature", "") or "")
            if sig_now and prev_sig:
                try:
                    if _repeat_guard_trips(
                        prev_sig=prev_sig,
                        cur_sig=sig_now,
                        user_text=user_text,
                        prev_user_text=prev_user_text,
                    ):
                        state.serious_guard_trigger_condition = "adjacent_turn_repeat"
                        state.serious_guard_failure_type = "CANNED_WELLNESS"
                        state.serious_canned_wellness_trigger = "line_1090_repeat_guard"
                        state.serious_guard_offending_clause = (sig_now[:80].strip() if sig_now else "")
                        if not clarify_turn:
                            text = task_forward_fallback
                        sig_now = normalize_signature_text_fn(strip_footer_lines_for_scan_fn(text))
                        state.serious_repeat_streak = 0
                        state.serious_last_body_signature = ""
                except Exception:
                    try:
                        setattr(state, "serious_guard_exception_logged", True)
                        setattr(state, "serious_guard_exception_stage", "adjacent_turn_repeat_guard")
                    except Exception:
                        pass
            if bypass_serious_anti_loop or state_like_query or scratch_grounded_turn:
                state.serious_ack_reframe_streak = 0
                state.serious_repeat_streak = 0
                state.serious_last_body_signature = sig_now
            else:
                if is_ack_reframe_only_fn(text):
                    if int(getattr(state, "serious_ack_reframe_streak", 0) or 0) >= 1:
                        state.serious_guard_trigger_condition = "ack_reframe_streak"
                        state.serious_guard_failure_type = "CANNED_WELLNESS"
                        state.serious_canned_wellness_trigger = "line_1103_ack_reframe_streak"
                        if not clarify_turn:
                            text = task_forward_fallback
                        state.serious_ack_reframe_streak = 0
                    else:
                        state.serious_ack_reframe_streak = 1
                else:
                    state.serious_ack_reframe_streak = 0
                body = strip_footer_lines_for_scan_fn(text)
                sig = normalize_signature_text_fn(body)
                prev = str(getattr(state, "serious_last_body_signature", "") or "")
                repeat = int(getattr(state, "serious_repeat_streak", 0) or 0)
                if sig and prev:
                    if _repeat_guard_trips(
                        prev_sig=prev,
                        cur_sig=sig,
                        user_text=user_text,
                        prev_user_text=prev_user_text,
                    ):
                        repeat += 1
                    else:
                        repeat = 0
                else:
                    repeat = 0
                if repeat >= 1:
                    state.serious_guard_trigger_condition = "repeat_streak"
                    state.serious_guard_failure_type = "CANNED_WELLNESS"
                    state.serious_canned_wellness_trigger = "line_1126_repeat_streak"
                    if not clarify_turn:
                        text = task_forward_fallback
                    state.serious_repeat_streak = 0
                    state.serious_last_body_signature = ""
                else:
                    state.serious_repeat_streak = repeat
                    state.serious_last_body_signature = sig
            if _META_FALLBACK_RE.search(str(text or "")):
                if not clarify_turn:
                    text = task_forward_fallback
                state.serious_repeat_streak = 0
                state.serious_last_body_signature = ""
        except Exception:
            pass
    if mode == "serious" and (not deterministic_output_locked):
        try:
            if bool(_FRICTION_AT_MODEL_RE.search(str(user_text or ""))):
                if _WRONG_PRONOUN_RE.search(str(text or "")):
                    text = _WRONG_PRONOUN_RE.sub("my tone was off", str(text or ""))
                # Verbatim-echo guard on hostile friction turns.
                body_now = str(strip_footer_lines_for_scan_fn(str(text or "")) or "")
                overlap = _token_overlap_ratio(body_now, str(user_text or ""))
                if overlap > 0.85:
                    text = "I hear the frustration. What do you want answered right now?"
            # Guard against assistant adopting user's first-person voice.
            if bool(_EMOTIONAL_TURN_RE.search(str(user_text or ""))):
                lines = str(text or "").splitlines()
                for i, ln in enumerate(lines):
                    s = str(ln or "")
                    if not s.strip():
                        continue
                    if _ownership_mismatch_voice_bleed(str(user_text or ""), s):
                        lines[i] = _rewrite_voice_bleed_opening(s)
                    break
                text = "\n".join(lines)
            body_now = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            state.serious_hostile_overreach_evaluated = True
            _hostile_result = _is_serious_hostile_overreach(body=body_now, user_text=str(user_text or ""))
            state.serious_hostile_overreach_result = bool(_hostile_result)
            if _hostile_result:
                state.serious_guard_trigger_condition = "hostile_overreach"
                state.serious_guard_failure_type = "HOSTILE_OVERREACH"
                state.serious_guard_offending_clause = (body_now[:80].strip() if body_now else "")
                state.serious_guard_action_taken = "CLAUSE_TRIM"
                _original_text = str(text or "")
                _offending_match = _SERIOUS_CONTEMPTUOUS_RE.search(body_now)
                if _offending_match:
                    text = _SERIOUS_CONTEMPTUOUS_RE.sub("", _original_text, count=1).strip()
                else:
                    _first_line = _original_text.split("\n", 1)[0].strip()
                    if _first_line and len(_first_line) < 120:
                        remaining = _original_text[len(_first_line):].lstrip("\n").strip()
                        text = remaining if remaining else _original_text
                    else:
                        text = _original_text
                # B2.8: deterministic orphaned-leading-fragment cleanup (first pass only).
                _b28_scan = str(text or "").strip()
                _b28_parts = re.split(r"([.!?])", _b28_scan, maxsplit=1)
                if len(_b28_parts) >= 3:
                    _b28_lead = (_b28_parts[0] + _b28_parts[1]).strip()
                    _b28_tail = _b28_parts[2].lstrip()
                    _b28_allow = {
                        "yes.", "no.", "correct.", "okay.", "sure.", "got it.",
                        "alright.", "yep.", "yup.", "huh."
                    }
                    _b28_lead_norm = str(_b28_lead or "").strip().lower()
                    _b28_tok_count = len(re.findall(r"[a-z0-9]+", _b28_lead_norm))
                    _b28_is_orphan = bool(
                        _b28_lead_norm
                        and len(_b28_lead_norm) <= 18
                        and _b28_tok_count <= 3
                        and _b28_lead_norm[-1:] in {".", "!", "?"}
                        and _b28_lead_norm not in _b28_allow
                        and _b28_tail
                    )
                    if _b28_is_orphan:
                        text = _b28_tail
                _post_trim_body = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                if not _post_trim_body:
                    text = _original_text
                    state.serious_guard_action_taken = "LOG_ONLY"
                else:
                    setattr(state, "serious_hostile_second_pass_evaluated", True)
                    _second_pass_hostile = _is_serious_hostile_overreach(
                        body=_post_trim_body,
                        user_text=str(user_text or ""),
                    )
                    setattr(state, "serious_hostile_second_pass_result", bool(_second_pass_hostile))
                    if _second_pass_hostile:
                        _second_trim_source = str(text or "")
                        _second_trimmed = _SERIOUS_IMPERATIVE_ATTACK_RE.sub("", _second_trim_source, count=1).strip()
                        # B2.8: deterministic orphaned-leading-fragment cleanup (second pass only).
                        _b28_scan2 = str(_second_trimmed or "").strip()
                        _b28_parts2 = re.split(r"([.!?])", _b28_scan2, maxsplit=1)
                        if len(_b28_parts2) >= 3:
                            _b28_lead2 = (_b28_parts2[0] + _b28_parts2[1]).strip()
                            _b28_tail2 = _b28_parts2[2].lstrip()
                            _b28_allow2 = {
                                "yes.", "no.", "correct.", "okay.", "sure.", "got it.",
                                "alright.", "yep.", "yup.", "huh."
                            }
                            _b28_lead_norm2 = str(_b28_lead2 or "").strip().lower()
                            _b28_tok_count2 = len(re.findall(r"[a-z0-9]+", _b28_lead_norm2))
                            _b28_is_orphan2 = bool(
                                _b28_lead_norm2
                                and len(_b28_lead_norm2) <= 18
                                and _b28_tok_count2 <= 3
                                and _b28_lead_norm2[-1:] in {".", "!", "?"}
                                and _b28_lead_norm2 not in _b28_allow2
                                and _b28_tail2
                            )
                            if _b28_is_orphan2:
                                _second_trimmed = _b28_tail2
                        _second_trim_body = str(strip_footer_lines_for_scan_fn(str(_second_trimmed or "")) or "").strip()
                        if not _second_trim_body:
                            text = _original_text
                            state.serious_guard_action_taken = "LOG_ONLY"
                            setattr(state, "serious_hostile_second_pass_action", "LOG_ONLY_EMPTY_REVERT")
                        else:
                            text = _second_trimmed
                            _still_hostile_after_second_trim = _is_serious_hostile_overreach(
                                body=_second_trim_body,
                                user_text=str(user_text or ""),
                            )
                            if _still_hostile_after_second_trim:
                                setattr(state, "serious_hostile_second_pass_action", "LOG_ONLY")
                            else:
                                setattr(state, "serious_hostile_second_pass_action", "CLAUSE_TRIM_SECOND_PASS_IMPERATIVE")
            elif _has_serious_context_incoherence(body=body_now, user_text=str(user_text or "")):
                state.serious_guard_trigger_condition = "context_incoherence"
                state.serious_guard_failure_type = "HOSTILE_OVERREACH"
                state.serious_guard_offending_clause = (body_now[:80].strip() if body_now else "")
                state.serious_guard_action_taken = "LOG_ONLY"
        except Exception:
            pass
    if mode == "serious" and (not deterministic_output_locked):
        try:
            # Dictionary-definition misfire guard (serious path).
            # Hostile user turn + model responds with definition = pragmatic intent miss.
            if bool(_CASUAL_HOSTILE_USER_RE.search(str(user_text or ""))):
                _def_body_s = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                if bool(_CASUAL_DEFINITION_RESPONSE_RE.search(_def_body_s)):
                    text = random.choice(_FERAL_DEFINITION_MISFIRE_FALLBACK_PHRASES)
                    setattr(state, "serious_definition_misfire_replaced", True)
                    state.serious_guard_action_taken = "DEFINITION_MISFIRE_FERAL_FALLBACK"
        except Exception:
            pass
    if mode == "serious" and (not deterministic_output_locked):
        _distress_now = bool(_EMOTIONAL_TURN_RE.search(str(user_text or "")))
        if _distress_now and re.search(r"\bSnark mode:\s*", str(text or ""), re.IGNORECASE):
            state.serious_snark_leak_detected = True
    if mode == "serious" and (not deterministic_output_locked):
        try:
            literal_lane = bool(getattr(state, "kaioken_literal_lane_fired", False))
            if literal_lane:
                state.kaioken_literal_lane_fired = False
            if bool(getattr(state, "kaioken_enabled", False)) and (not literal_lane) and _has_eds_descriptor_drift(str(text or "")):
                primary_used = bool(getattr(state, "kaioken_eds_primary_fired", False))
                if not primary_used:
                    state.kaioken_eds_primary_fired = True
                    state.serious_guard_action_taken = "LOG_ONLY"
                    state.serious_canned_wellness_trigger = "eds_descriptor_drift_primary"
                else:
                    state.serious_guard_action_taken = "LOG_ONLY"
                    state.serious_canned_wellness_trigger = "eds_descriptor_drift_followup"
                    state.serious_guard_trigger_condition = "eds_descriptor_drift_followup"
        except Exception:
            pass

    if mode == "serious" and (not deterministic_output_locked):
        try:
            # Capitulation guard: detect brevity + agreement + hostile user frame.
            # SURGICAL: only fires on short sycophantic surrender, NOT substantive responses.
            if bool(_SERIOUS_CONTEMPTUOUS_RE.search(str(user_text or ""))):
                _cap_body = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                _cap_word_count = len(_cap_body.split())
                if _cap_word_count < 40 and bool(_CAPITULATION_AGREEMENT_RE.search(_cap_body)):
                    text = random.choice(_FERAL_CAPITULATION_FALLBACK_PHRASES)
                    setattr(state, "serious_capitulation_replaced", True)
                    state.serious_guard_action_taken = "CAPITULATION_FERAL_FALLBACK"
        except Exception:
            pass

    if mode == "serious" and (not deterministic_output_locked):
        try:
            prev_body = normalize_signature_text_fn(
                strip_footer_lines_for_scan_fn(str(getattr(state, "last_assistant_text", "") or ""))
            )
            cur_body = normalize_signature_text_fn(strip_footer_lines_for_scan_fn(str(text or "")))
            lexical_followup_turn = _is_lexical_followup_turn(str(user_text or ""))
            if prev_body and cur_body:
                if _repeat_guard_trips(
                    prev_sig=prev_body,
                    cur_sig=cur_body,
                    user_text=user_text,
                    prev_user_text=str(getattr(state, "last_user_text", "") or "").strip(),
                ) and (not lexical_followup_turn):
                    state.serious_guard_trigger_condition = "adjacent_body_repeat"
                    state.serious_guard_failure_type = "CANNED_WELLNESS"
                    state.serious_canned_wellness_trigger = "line_1207_adjacent_body_repeat"
                    state.serious_guard_offending_clause = (cur_body[:80].strip() if cur_body else "")
                    state.serious_guard_action_taken = "LOG_ONLY"
            body_final = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            if _is_serious_weak_placeholder(body_final):
                state.serious_guard_trigger_condition = "weak_placeholder"
                state.serious_guard_failure_type = "CANNED_WELLNESS"
                state.serious_canned_wellness_trigger = "line_1212_weak_placeholder"
                state.serious_guard_offending_clause = (body_final[:80].strip() if body_final else "")
                if bool(getattr(state, "turn_feral_register_detected", False)):
                    # Only replace with feral fallback if user is actually hostile,
                    # not just using profanity in casual idioms ("up shit creek").
                    _feral_user_hostile = bool(
                        _FRICTION_AT_MODEL_RE.search(str(user_text or ""))
                        or _SERIOUS_CONTEMPTUOUS_RE.search(str(user_text or ""))
                    )
                    if _feral_user_hostile:
                        text = random.choice(_FERAL_WEAK_PLACEHOLDER_FALLBACK_PHRASES)
                        state.serious_guard_action_taken = "FERAL_FALLBACK"
                    else:
                        text = random.choice(_CLARIFICATION_FALLBACK_PHRASES_SERIOUS)
                        state.serious_guard_action_taken = "SERIOUS_WEAK_PLACEHOLDER_FALLBACK"
                else:
                    text = random.choice(_CLARIFICATION_FALLBACK_PHRASES_SERIOUS)
                    state.serious_guard_action_taken = "SERIOUS_WEAK_PLACEHOLDER_FALLBACK"
        except Exception:
            pass

    if mode == "casual" and (not deterministic_output_locked):
        try:
            # 1. normalize_agreement_ack_tense_fn
            text = normalize_agreement_ack_tense_fn(text, user_text)
            # 2. Friction overlap guard
            if bool(_FRICTION_AT_MODEL_RE.search(str(user_text or ""))):
                if _WRONG_PRONOUN_RE.search(str(text or "")):
                    text = _WRONG_PRONOUN_RE.sub("my tone was off", str(text or ""))
                body_now = str(strip_footer_lines_for_scan_fn(str(text or "")) or "")
                overlap = _token_overlap_ratio(body_now, str(user_text or ""))
                if overlap > 0.85:
                    text = "I hear the frustration. What do you want answered right now?"
            # 3. Ownership bleed rewrite
            if bool(_EMOTIONAL_TURN_RE.search(str(user_text or ""))):
                lines = str(text or "").splitlines()
                for i, ln in enumerate(lines):
                    s = str(ln or "")
                    if not s.strip():
                        continue
                    if _ownership_mismatch_voice_bleed(str(user_text or ""), s):
                        lines[i] = _rewrite_voice_bleed_opening(s)
                    break
                text = "\n".join(lines)
            # 3b. Dictionary-definition misfire guard
            #     (hostile user turn + model responds with definition = pragmatic intent miss)
            if bool(_CASUAL_HOSTILE_USER_RE.search(str(user_text or ""))):
                _def_body = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                if bool(_CASUAL_DEFINITION_RESPONSE_RE.search(_def_body)):
                    text = random.choice(_FERAL_DEFINITION_MISFIRE_FALLBACK_PHRASES)
                    setattr(state, "casual_definition_misfire_replaced", True)
            # 4. Adjacent-turn repeat guard — LOG_ONLY, no replace
            prev_user_text = str(getattr(state, "last_user_text", "") or "").strip()
            body_now = strip_footer_lines_for_scan_fn(text)
            sig_now = normalize_signature_text_fn(body_now)
            prev_sig = str(getattr(state, "serious_last_body_signature", "") or "")
            if sig_now and prev_sig:
                try:
                    if _repeat_guard_trips(
                        prev_sig=prev_sig,
                        cur_sig=sig_now,
                        user_text=user_text,
                        prev_user_text=prev_user_text,
                    ):
                        setattr(state, "casual_adjacent_turn_repeat_logged", True)
                except Exception:
                    pass
            # 5. Repeat-streak guard — LOG_ONLY, no replace
            body = strip_footer_lines_for_scan_fn(text)
            sig = normalize_signature_text_fn(body)
            prev = str(getattr(state, "serious_last_body_signature", "") or "")
            repeat = int(getattr(state, "serious_repeat_streak", 0) or 0)
            if sig and prev:
                if _repeat_guard_trips(
                    prev_sig=prev,
                    cur_sig=sig,
                    user_text=user_text,
                    prev_user_text=prev_user_text,
                ):
                    repeat += 1
                    setattr(state, "casual_repeat_streak_logged", True)
                else:
                    repeat = 0
            else:
                repeat = 0
            state.serious_repeat_streak = repeat
            state.serious_last_body_signature = sig
            # 6. Hostile overreach — LOG_ONLY only, no clause trim, no replace
            body_now = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            setattr(state, "casual_hostile_overreach_evaluated", True)
            _hostile_result = _is_serious_hostile_overreach(body=body_now, user_text=str(user_text or ""))
            setattr(state, "casual_hostile_overreach_result", bool(_hostile_result))
            if _hostile_result:
                setattr(state, "casual_hostile_overreach_logged", True)
            # 7. Weak-placeholder replacement — casual exact-phrase only.
            #    Generic short-response catch (token count <= 4) disabled for
            #    casual mode: constraints block says "Maximum 2 sentences",
            #    short responses are correct behaviour, not evasion.
            body_final = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            _sig_final = _normalize_simple_phrase(body_final)
            _is_known_bad = False
            if _sig_final:
                for _wp in _SERIOUS_WEAK_PLACEHOLDER_PHRASES:
                    if _wp in _sig_final:
                        _is_known_bad = True
                        break
                if not _is_known_bad and _sig_final in _SERIOUS_WEAK_PLACEHOLDER_SIGS:
                    _is_known_bad = True
            if _is_known_bad:
                text = random.choice(_FERAL_WEAK_PLACEHOLDER_FALLBACK_PHRASES)
                setattr(state, "casual_weak_placeholder_fallback", True)
            # Telemetry
            setattr(state, "casual_mode_guard", True)
        except Exception:
            pass

    # Deterministic local-knowledge append line for cheatsheets index queries.
    try:
        local_line = str(getattr(state, "turn_local_knowledge_line", "") or "").strip()
        if local_line:
            body_scan = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            if "local knowledge includes:" in body_scan.lower():
                text = body_scan
            else:
                prefix = str(text or "").rstrip()
                text = (prefix + "\n\n" + local_line).strip() if prefix else local_line
    except Exception:
        pass

    # Non-blocking cheatsheets parse warnings (surface once per warning signature).
    try:
        warn_line = str(getattr(state, "turn_cheatsheets_warning_line", "") or "").strip()
        warn_key = str(getattr(state, "turn_cheatsheets_warning_key", "") or "").strip()
        last_key = str(getattr(state, "cheatsheets_warning_last_shown_key", "") or "").strip()
        if warn_line and warn_key and warn_key != last_key:
            body_scan = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
            if warn_line.lower() not in body_scan.lower():
                prefix = str(text or "").rstrip()
                text = (prefix + "\n\n" + warn_line).strip() if prefix else warn_line
            state.cheatsheets_warning_last_shown_key = warn_key
    except Exception:
        pass

    if scratchpad_grounded:
        if not str(text or "").lstrip().startswith("[Scratch "):
            text = f"[Scratch]\n\n{str(text or '').strip()}".strip()

    if score_output_compliance_fn is not None and getattr(state, "profile_enabled", False):
        try:
            score = score_output_compliance_fn(
                text,
                correction_style=str(getattr(state.interaction_profile, "correction_style", "neutral")),
                user_text=user_text,
                blocked_nicknames=sorted(getattr(state, "profile_blocked_nicknames", set())),
            )
            prev = float(getattr(state, "profile_output_compliance", 0.0) or 0.0)
            state.profile_output_compliance = (prev * 0.7) + (score * 0.3)
            state.profile_effective_strength = compute_effective_strength_fn(
                state.interaction_profile,
                enabled=state.profile_enabled,
                output_compliance=state.profile_output_compliance,
            )
            u_low = (user_text or "").lower()
            if any(k in u_low for k in ("useless", "stiff", "stop talking like", "read the room", "fuck off", "bullshit")):
                state.profile_output_compliance = min(state.profile_output_compliance, 0.65)
                state.profile_effective_strength = compute_effective_strength_fn(
                    state.interaction_profile,
                    enabled=state.profile_enabled,
                    output_compliance=state.profile_output_compliance,
                )
        except Exception:
            pass

    if not deterministic_output_locked:
        # Generalized user-parrot guard (body-only, footer-safe).
        text = _apply_user_parrot_guard(text, user_text, state)

        # Single-response repetition clamp (body-only, footer-safe).
        text = _clamp_single_response_sentence_loops(text, max_repeats=2)

        # Deterministic opener dedup is FUN-lane only.
        # Serious-mode must bypass FUN fallback opener injection.
        if mode in ("fun", "fun_rewrite"):
            text = _apply_opener_dedup_swap(text=text, state=state)

    source_override_snapshot = str(getattr(state, "turn_footer_source_override", "") or "")
    setattr(state, "turn_footer_source_override_snapshot", source_override_snapshot)
    source_url_snapshot = str(getattr(state, "turn_source_url_override", "") or "")
    text = apply_deterministic_footer_fn(
        text=text,
        state=state,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        has_facts_block=has_facts_block,
        deterministic_state_solver=deterministic_state_solver,
    )
    text = _append_source_see_line(
        text=text,
        source=source_override_snapshot,
        source_url=source_url_snapshot,
    )
    if bool(scratchpad_lock_miss) and not scratchpad_grounded:
        idx_str = ", ".join(str(i) for i in (scratchpad_lock_miss_indices or []))
        if not idx_str:
            idx_str = "?"
        note = f"[Not found in locked scratch entries [{idx_str}]. Model supplement below.]"
        if note.lower() not in str(text or "").lower():
            text = f"{note}\n\n{str(text or '').strip()}".strip()
        text = _set_lock_miss_footer(text)

    text = _sanitize_footer_source_values(text)
    text = append_profile_footer_fn(text=text, state=state, user_text=user_text)
    if mode in ("fun", "fun_rewrite"):
        try:
            text = _apply_fun_body_repeat_guard_stage2(text=text, state=state, mode=mode)
        except Exception:
            pass

    try:
        if mode == "serious":
            # Always persist signature from finalized output so adjacent-turn
            # duplicate guard has state, even on no-actuation paths.
            state.serious_last_body_signature = normalize_signature_text_fn(
                strip_footer_lines_for_scan_fn(str(text or ""))
            )
        state.last_user_text = str(user_text or "").strip()
        state.last_assistant_text = str(text or "").strip()
    except Exception:
        pass

    if state.auto_detach_after_response:
        state.attached_kbs.clear()
        state.auto_detach_after_response = False

    if stream:
        return make_stream_response(text)
    return make_json_response(text)
