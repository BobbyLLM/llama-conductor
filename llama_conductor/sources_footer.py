"""Deterministic confidence/source footer normalization.

Goals:
- Keep Mentats output contract untouched.
- Preserve existing explicit source lines (e.g., locked/scratchpad markers).
- Replace/append one deterministic confidence footer line based on router signals.
"""

from __future__ import annotations

import re


_CONF_RE = re.compile(
    r"^\s*Confidence:\s*(unverified|low|medium|med|high|top)\s*\|\s*Source:\s*(Model|Docs|User|Contextual|Mixed|Scratchpad|Cheatsheets|Define|Wiki|Web|Codex)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SRC_RE = re.compile(
    r"^\s*Source:\s*(Model|Docs|User|Contextual|Mixed|Scratchpad|Cheatsheets|Define|Wiki|Web|Codex)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONF_PREFIX_RE = re.compile(r"^\s*Confidence:\s*", re.IGNORECASE)
_INLINE_CONF_RE = re.compile(
    r"\s*Confidence:\s*(?:unverified|low|medium|med|high|top)\s*\|\s*Source:\s*(?:Model|Docs|User|Contextual|Mixed|Scratchpad|Cheatsheets|Define|Wiki|Web|Codex)(?:[\s\.\-:]+[A-Za-z]+){0,3}\s*",
    re.IGNORECASE,
)
_INLINE_BROKEN_CONF_RE = re.compile(
    r"\s*Confidence:\s*(?:unverified|low|medium|med|high|top)\s*\|\s*(?:S(?:o|ou|our|ource)?[^\\n]{0,80})$",
    re.IGNORECASE,
)
_INLINE_PROFILE_TAIL_RE = re.compile(
    r"(?:"
    r"\s*(?:\|\s*)?Profile:\s*[^|\n]{1,40}\s*\|\s*Sarc:\s*[^|\n]{1,20}\s*\|\s*Snark:\s*[^|\n]{1,20}"
    r"|"
    r"\s*\|\s*Sarc:\s*[^|\n]{1,20}\s*\|\s*Snark:\s*[^|\n]{1,20}"
    r")\s*$",
    re.IGNORECASE,
)
_INLINE_PROFILE_ANY_RE = re.compile(
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
_INLINE_TRUNC_SOURCE_TAIL_RE = re.compile(
    r"\s*\|\s*S(?:o|ou|our|ource)?[:\s\.\-…]*$",
    re.IGNORECASE,
)
_SIMPLE_SOURCE_LINE_RE = re.compile(
    r"^\s*Source:\s*(Model|Docs|User|Contextual|Mixed|Scratchpad|Cheatsheets|Define|Wiki|Web|Codex)\s*$",
    re.IGNORECASE,
)


def _strip_confidence_lines(text: str) -> str:
    lines = (text or "").splitlines()
    kept = []
    for ln in lines:
        if _CONF_RE.match(ln or "") or _CONF_PREFIX_RE.match(ln or ""):
            continue
        # Remove redundant bare source lines; footer adds canonical source line.
        # Keep detailed source lines (e.g., locked-file/model-fallback variants).
        # Preserve explicit scratchpad provenance line for scratchpad contract tests.
        if _SIMPLE_SOURCE_LINE_RE.match(ln or ""):
            if (ln or "").strip().lower() == "source: scratchpad":
                kept.append((ln or "").strip())
                continue
            continue
        ln2 = _INLINE_CONF_RE.sub("", ln).rstrip()
        ln2 = _INLINE_BROKEN_CONF_RE.sub("", ln2).rstrip()
        ln2 = _INLINE_PROFILE_TAIL_RE.sub("", ln2).rstrip()
        ln2 = _INLINE_PROFILE_ANY_RE.sub("", ln2).rstrip()
        kept.append(ln2)
    return "\n".join(kept).strip()


def _has_explicit_docs_marker(text: str) -> bool:
    low = str(text or "").lower()
    return (
        "source: docs" in low
        or "source: locked file (" in low
        or "sources: vault" in low
    )


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
        return "Scratchpad"

    m = _CONF_RE.search(t)
    if m:
        src = (m.group(2) or "").strip().title()
        return src if src in {"Model", "Docs", "User", "Contextual", "Mixed", "Scratchpad", "Cheatsheets", "Define", "Wiki", "Web", "Codex"} else "Model"
    m_src = _SRC_RE.search(t)
    if m_src:
        src = (m_src.group(1) or "").strip().title()
        return src if src in {"Model", "Docs", "User", "Contextual", "Mixed", "Scratchpad", "Cheatsheets", "Define", "Wiki", "Web", "Codex"} else "Model"

    if lock_active:
        return "Model" if "not found in locked source" in low else "Docs"
    if scratchpad_grounded:
        return "Scratchpad"
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
    if source == "Wiki":
        return "medium"
    if source == "Web":
        return "medium"
    if source == "Cheatsheets":
        return "high"
    if source == "Codex":
        return "high"
    if source == "Define":
        return "medium"
    if source in {"Docs", "Scratchpad"}:
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
    source_override: str = "",
    confidence_override: str = "",
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
    override_source = str(source_override or "").strip().title()
    if override_source in {"Model", "Docs", "User", "Contextual", "Mixed", "Scratchpad", "Cheatsheets", "Define", "Wiki", "Web", "Codex"}:
        source = override_source
    # Structural provenance guard:
    # "User" source must be explicitly assigned by the router lane.
    # Never trust model-emitted "Source: User" lines in body/footer text.
    if source == "User" and override_source != "User":
        source = "Model"
    # Structural provenance guard:
    # "Wiki" source must come from lane override (actual injected wiki facts),
    # never from model-emitted body/footer text.
    if source == "Wiki" and override_source != "Wiki":
        source = "Model"
    if source == "Web" and override_source != "Web":
        source = "Model"
    # Retrieval-miss responses are system miss notices, not grounded evidence.
    # Never badge these as Web/Wiki/User, regardless of upstream attempt lane.
    miss_low = t.lower()
    retrieval_miss = (
        miss_low.startswith("not available in retrieved web facts")
        or miss_low.startswith("not available in retrieved wiki facts")
        or "not available in retrieved web facts." in miss_low
        or "not available in retrieved wiki facts." in miss_low
    )
    if retrieval_miss:
        source = "Model"
    # Provenance floor: do not upgrade to Docs without explicit evidence of a docs lane.
    if (
        source == "Docs"
        and not override_source
        and not lock_active
        and not scratchpad_grounded
        and max(0, int(rag_hits or 0)) <= 0
        and not _has_explicit_docs_marker(t)
    ):
        source = "Model"
    # Provenance floor: do not keep Mixed on model-only turns.
    # "Mixed" must come from an evidence lane or explicit override, not from
    # model-emitted footer/body text.
    if (
        source == "Mixed"
        and not override_source
        and not lock_active
        and not scratchpad_grounded
        and not has_facts_block
        and max(0, int(rag_hits or 0)) <= 0
    ):
        source = "Model"
    model_fallback = source == "Model" and "not found in locked source" in t.lower()
    conf = _compute_confidence(
        source=source,
        lock_active=lock_active,
        model_fallback=model_fallback,
        has_facts_block=has_facts_block,
        rag_hits=max(0, int(rag_hits or 0)),
        locked_fact_lines=max(0, int(locked_fact_lines or 0)),
    )
    override_conf = str(confidence_override or "").strip().lower()
    if retrieval_miss:
        conf = "unverified"
    elif override_conf in {"unverified", "low", "medium", "med", "high", "top"}:
        conf = "medium" if override_conf == "med" else override_conf

    base = _strip_confidence_lines(t)
    base = _INLINE_TRUNC_SOURCE_TAIL_RE.sub("", base).rstrip()
    return f"{base}\n\nConfidence: {conf} | Source: {source}"
