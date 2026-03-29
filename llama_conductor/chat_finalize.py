from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import re
from typing import Any, Callable, List, Optional

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


def _token_set(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", str(s or "").lower())
    return {t for t in toks if t and t not in _STOPWORDS and len(t) >= 3}


def _token_overlap_ratio(a: str, b: str) -> float:
    a_toks = _token_set(a)
    b_toks = _token_set(b)
    if not a_toks or not b_toks:
        return 0.0
    return float(len(a_toks.intersection(b_toks)) / max(1, len(a_toks)))


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
        if re.match(r"^\s*See:\s*https?://\S+\s*$", str(ln or "").strip(), flags=re.IGNORECASE):
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

    skip_profile_rewrite = bool(deterministic_state_solver and mode in ("fun", "fun_rewrite"))
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

    clarify_turn = bool(_CLARIFICATION_TURN_RE.search(str(user_text or "").strip()))
    distress_turn = bool(_EMOTIONAL_TURN_RE.search(str(user_text or "")))
    strip_fallback = (
        "Sorry, I lost the thread there - tell me again?"
        if clarify_turn
        else ("I hear you." if distress_turn else "I hear you.")
    )
    if not deterministic_output_locked:
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
        try:
            emotional_turn = bool(_EMOTIONAL_TURN_RE.search(str(user_text or "")))
            friction_at_model_turn = bool(_FRICTION_AT_MODEL_RE.search(str(user_text or "")))
            prev_user_text = str(getattr(state, "last_user_text", "") or "").strip()
            task_forward_fallback = (
                "I hear you. Let's stay with what you said."
                if emotional_turn
                else serious_task_forward_fallback
            )
            # Opening engagement nudge: on vent-style openers, avoid flat acknowledgements
            # by requiring one open question when absent.
            if emotional_turn and _VENT_OPENER_RE.search(str(user_text or "")):
                body_tmp = str(strip_footer_lines_for_scan_fn(str(text or "")) or "").strip()
                if body_tmp and "?" not in body_tmp and not bool(getattr(state, "kaioken_literal_lane_fired", False)):
                    text = body_tmp.rstrip(". ") + ". Want to talk about what happened?"
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
                        text = task_forward_fallback
                        sig_now = normalize_signature_text_fn(strip_footer_lines_for_scan_fn(text))
                        state.serious_repeat_streak = 0
                        state.serious_last_body_signature = ""
                except Exception:
                    pass
            if bypass_serious_anti_loop or state_like_query or scratch_grounded_turn:
                state.serious_ack_reframe_streak = 0
                state.serious_repeat_streak = 0
                state.serious_last_body_signature = sig_now
            else:
                if is_ack_reframe_only_fn(text):
                    if int(getattr(state, "serious_ack_reframe_streak", 0) or 0) >= 1:
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
                    text = task_forward_fallback
                    state.serious_repeat_streak = 0
                    state.serious_last_body_signature = ""
                else:
                    state.serious_repeat_streak = repeat
                    state.serious_last_body_signature = sig
            if _META_FALLBACK_RE.search(str(text or "")):
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
        except Exception:
            pass
    if mode == "serious" and (not deterministic_output_locked):
        try:
            literal_lane = bool(getattr(state, "kaioken_literal_lane_fired", False))
            if literal_lane:
                state.kaioken_literal_lane_fired = False
            if bool(getattr(state, "kaioken_enabled", False)) and (not literal_lane) and _has_eds_descriptor_drift(str(text or "")):
                primary_used = bool(getattr(state, "kaioken_eds_primary_fired", False))
                if not primary_used:
                    text = "I hear you."
                    state.kaioken_eds_primary_fired = True
                else:
                    text = "That's a lot."
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
                    text = "I hear you."
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

    source_override_snapshot = str(getattr(state, "turn_footer_source_override", "") or "")
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
