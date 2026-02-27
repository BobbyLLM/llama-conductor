from __future__ import annotations

import re
from typing import Any, Callable, Dict, List


_QUOTE_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_QUOTE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "should", "would", "from", "about", "tell", "me", "more", "all", "any",
}
_PROFILE_FOOTER_FRAGMENT_RE = re.compile(
    r"Profile:\s*[^|\n]{1,40}\|\s*Sarc:\s*[^|\n]{1,20}\|\s*Snark:\s*[^|\n]{1,20}",
    re.IGNORECASE,
)


def quote_query_tokens(text: str) -> set[str]:
    toks = {m.group(0).lower() for m in _QUOTE_TOKEN_RE.finditer(text or "")}
    return {t for t in toks if len(t) > 2 and t not in _QUOTE_STOPWORDS}


def scratchpad_quote_lines(
    sp_block: str,
    *,
    query: str = "",
    max_quotes: int = 2,
    max_len: int = 160,
) -> List[str]:
    out: List[str] = []
    q_tokens = quote_query_tokens(query)
    for ln in (sp_block or "").splitlines():
        s = (ln or "").strip()
        if not s.startswith("- [scratchpad "):
            continue
        if "] " not in s:
            continue
        snippet = s.split("] ", 1)[1].strip()
        if not snippet:
            continue
        if q_tokens:
            s_tokens = quote_query_tokens(snippet)
            if not (q_tokens & s_tokens):
                continue
        if len(snippet) > max_len:
            snippet = snippet[: max_len - 1].rstrip() + "..."
        out.append(snippet)
        if len(out) >= max_quotes:
            break
    return out


def sanitize_scratchpad_grounded_output(text: str) -> str:
    if not text:
        return ""
    cleaned: List[str] = []
    for ln in text.splitlines():
        low = ln.lower()
        if "source: docs" in low:
            continue
        if "source: model" in low:
            continue
        if "source: facts" in low:
            ln = ln.replace("Source: Facts", "Source: Scratchpad")
            ln = ln.replace("source: facts", "source: scratchpad")
        if "source: contextual" in low:
            ln = ln.replace("Source: Contextual", "Source: Scratchpad")
            ln = ln.replace("source: contextual", "source: scratchpad")
        if "source: mixed" in low:
            ln = ln.replace("Source: Mixed", "Source: Scratchpad")
            ln = ln.replace("source: mixed", "source: scratchpad")
        cleaned.append(ln)

    out: List[str] = []
    blank_run = 0
    for ln in cleaned:
        if not ln.strip():
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        out.append(ln)
    return "\n".join(out).strip()


def append_scratchpad_provenance(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Source: Scratchpad"
    for ln in t.splitlines():
        if "source:" in ln.lower():
            return t
    return t + "\n\nSource: Scratchpad"


def rewrite_source_line(text: str, source_line: str) -> str:
    t = (text or "").strip()
    if not t:
        return source_line
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        if "source:" in ln.lower():
            lines[i] = source_line
            return "\n".join(lines).strip()
    return t.rstrip() + "\n\n" + source_line


def lock_constraints_block(locked_file: str) -> str:
    lf = (locked_file or "SUMM_locked.md").strip()
    return (
        "Grounding mode: LOCKED_SUMM_FILE.\n"
        f"- Locked file: {lf}\n"
        "- Prefer facts from the locked file for answers.\n"
        f"- If the answer is not present in {lf}, you may answer from pre-trained knowledge.\n"
        f"- In that fallback case, include this exact note line first: "
        f"[Not found in locked source {lf}. Answer based on pre-trained data.]\n"
        "- In that fallback case, final source line must be exactly: Source: Model (not in locked file)"
    )


def apply_locked_output_policy(text: str, state: Any) -> str:
    if not state.locked_summ_path:
        return text
    locked_file = state.locked_summ_file or "SUMM_locked.md"
    t = (text or "").strip()
    if not t:
        return rewrite_source_line("", f"Source: Locked file ({locked_file})")

    low = t.lower()
    if "source: model" in low:
        note = f"[Not found in locked source {locked_file}. Answer based on pre-trained data.]"
        if note.lower() not in low:
            t = note + "\n\n" + t
        return rewrite_source_line(t, "Source: Model (not in locked file)")

    return rewrite_source_line(t, f"Source: Locked file ({locked_file})")


def apply_deterministic_footer(
    *,
    text: str,
    state: Any,
    lock_active: bool,
    scratchpad_grounded: bool,
    has_facts_block: bool,
    deterministic_state_solver: bool = False,
    normalize_sources_footer_fn: Callable[..., str] | None,
) -> str:
    if not normalize_sources_footer_fn:
        return text
    try:
        if deterministic_state_solver:
            t = (text or "").strip()
            if t:
                # Preserve deterministic provenance on mode-wrapped answers.
                if "Source: Contextual" not in t:
                    t = t + "\nSource: Contextual"
            else:
                t = "Source: Contextual"
            text = t
        return normalize_sources_footer_fn(
            text=text,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            has_facts_block=has_facts_block,
            rag_hits=int(getattr(state, "rag_last_hits", 0) or 0),
            locked_fact_lines=int(getattr(state, "locked_last_fact_lines", 0) or 0),
        )
    except Exception:
        return text


def strip_profile_footer_lines(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _PROFILE_FOOTER_FRAGMENT_RE.sub("", t)
    out: List[str] = []
    for ln in t.splitlines():
        s = (ln or "").strip()
        if not s:
            out.append("")
            continue
        if s.lower().startswith("profile:"):
            continue
        s = re.sub(r"\s{2,}", " ", s).rstrip(" |")
        if s:
            out.append(s)

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


def append_profile_footer(
    *,
    text: str,
    state: Any,
    user_text: str,
    cfg_get_fn: Callable[[str, Any], Any],
    effective_profile_fn: Callable[..., Any],
) -> str:
    if not bool(cfg_get_fn("footer.profile.enabled", True)):
        return text
    if not bool(getattr(state, "profile_enabled", False)):
        return text

    try:
        prof = effective_profile_fn(state.interaction_profile, user_text)
        style = str(getattr(prof, "correction_style", "neutral"))
        sarc = str(getattr(prof, "sarcasm_level", "off"))
        snark = str(getattr(prof, "snark_tolerance", "low"))
        mode = str(cfg_get_fn("footer.profile.mode", "non_default") or "non_default").strip().lower()
        if mode not in {"always", "non_default", "debug"}:
            mode = "non_default"
        if mode == "debug" and not bool(cfg_get_fn("router.debug", False)):
            return text
        manual_overrides = bool(getattr(state.interaction_profile, "manual_overrides", set()))
        non_default = (style != "neutral") or (sarc != "off") or (snark != "low") or manual_overrides
        if mode == "non_default" and not non_default:
            return strip_profile_footer_lines(text)
        line = f"Profile: {style} | Sarc: {sarc} | Snark: {snark}"
    except Exception:
        return text

    base = strip_profile_footer_lines(text)
    if not base:
        return line
    return f"{base}\n{line}"
