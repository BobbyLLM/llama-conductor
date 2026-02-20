"""Deterministic confidence/source footer normalization.

Goals:
- Keep Mentats output contract untouched.
- Preserve existing explicit source lines (e.g., locked/scratchpad markers).
- Replace/append one deterministic confidence footer line based on router signals.
"""

from __future__ import annotations

import re


_CONF_RE = re.compile(
    r"^\s*Confidence:\s*(low|medium|med|high|top)\s*\|\s*Source:\s*(Model|Docs|User|Contextual|Mixed)\s*$",
    re.IGNORECASE,
)
_CONF_PREFIX_RE = re.compile(r"^\s*Confidence:\s*", re.IGNORECASE)


def _strip_confidence_lines(text: str) -> str:
    lines = (text or "").splitlines()
    kept = [ln for ln in lines if not _CONF_RE.match(ln or "") and not _CONF_PREFIX_RE.match(ln or "")]
    return "\n".join(kept).strip()


def _detect_abstract_source(
    *,
    text: str,
    lock_active: bool,
    scratchpad_grounded: bool,
    has_facts_block: bool,
) -> str:
    t = (text or "")
    low = t.lower()

    if "source: model (not in locked file)" in low:
        return "Model"
    if "source: locked file (" in low:
        return "Docs"
    if "source: scratchpad" in low:
        return "Docs"

    m = _CONF_RE.search(t)
    if m:
        src = (m.group(2) or "").strip().title()
        return src if src in {"Model", "Docs", "User", "Contextual", "Mixed"} else "Model"

    if lock_active:
        return "Model" if "not found in locked source" in low else "Docs"
    if scratchpad_grounded:
        return "Docs"
    if has_facts_block:
        if "source: model" in low:
            return "Model"
        return "Docs"
    return "Model"


def _compute_confidence(
    *,
    source: str,
    lock_active: bool,
    model_fallback: bool,
    has_facts_block: bool,
    rag_hits: int,
    locked_fact_lines: int,
) -> str:
    if model_fallback or source == "Model":
        return "unverified"
    if source in {"User", "Contextual", "Mixed"}:
        return "medium"
    if source == "Docs":
        if lock_active and locked_fact_lines >= 4:
            return "top"
        if (not lock_active) and has_facts_block and rag_hits >= 4:
            return "top"
        return "high"
    return "medium"


def normalize_sources_footer(
    *,
    text: str,
    lock_active: bool,
    scratchpad_grounded: bool,
    has_facts_block: bool,
    rag_hits: int = 0,
    locked_fact_lines: int = 0,
) -> str:
    """Normalize confidence footer deterministically for non-Mentats outputs."""
    t = (text or "").strip()
    if not t:
        return "Confidence: unverified | Source: Model"

    # Mentats contract is separate and must remain unchanged.
    if "sources: vault" in t.lower():
        return t

    source = _detect_abstract_source(
        text=t,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        has_facts_block=has_facts_block,
    )
    model_fallback = source == "Model" and "not found in locked source" in t.lower()
    conf = _compute_confidence(
        source=source,
        lock_active=lock_active,
        model_fallback=model_fallback,
        has_facts_block=has_facts_block,
        rag_hits=max(0, int(rag_hits or 0)),
        locked_fact_lines=max(0, int(locked_fact_lines or 0)),
    )

    base = _strip_confidence_lines(t)
    return f"{base}\n\nConfidence: {conf} | Source: {source}"
