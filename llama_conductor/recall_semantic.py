"""Recall semantic formatting helpers.

Separated from router orchestration to reduce router module size and keep
recall-specific logic testable in isolation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Tuple

_RECALL_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")
_RECALL_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "should", "would", "from", "about", "tell", "me", "more", "all", "any",
    "you", "your", "i", "we", "they", "them", "he", "she", "his", "her", "their", "our",
    "again", "point", "form", "fine", "yes", "no", "not", "mentioned", "mention",
}


def parse_recall_det_payload(text: str) -> Tuple[str, str, List[Dict[str, str]], bool]:
    """Parse recall payload. Returns (draft, query, evidence, parsed_payload)."""
    stripped = re.sub(r"^\s*\[recall_det\]\s*", "", str(text or ""), count=1, flags=re.IGNORECASE).strip()
    if not stripped:
        return "", "", [], False
    try:
        obj = json.loads(stripped)
    except Exception:
        # Backward-compatible path: old payload was raw draft text.
        return stripped, "", [], False
    if not isinstance(obj, dict):
        return stripped, "", [], False
    draft = str(obj.get("draft") or "").strip()
    query = str(obj.get("query") or "").strip()
    raw_evidence = obj.get("evidence", [])
    evidence: List[Dict[str, str]] = []
    if isinstance(raw_evidence, list):
        for entry in raw_evidence[:16]:
            if not isinstance(entry, dict):
                continue
            ev_id = str(entry.get("id") or "").strip()
            ev_type = str(entry.get("matrix_type") or "").strip()
            ev_obj = str(entry.get("object") or "").strip()
            ev_text = str(entry.get("text") or "").strip()
            if not (ev_id or ev_text):
                continue
            evidence.append(
                {
                    "id": ev_id,
                    "matrix_type": ev_type,
                    "object": ev_obj,
                    "text": ev_text,
                }
            )
    return draft, query, evidence, True


def _should_skip_semantic_recall(draft: str) -> bool:
    d = str(draft or "").strip().lower()
    if not d:
        return True
    keep_prefixes = (
        "you substituted the following:",
        "technologies replaced with simpler",
        "you promised the following:",
        "you also mentioned:",
        "yes.\n-",
        "no.\n",
    )
    return any(d.startswith(p) for p in keep_prefixes)


def _recall_sig_tokens(text: str) -> set[str]:
    toks = {m.group(0).lower() for m in _RECALL_TOKEN_RE.finditer(str(text or ""))}
    out = set()
    for tok in toks:
        if len(tok) < 4:
            continue
        if tok.isdigit():
            continue
        if tok in _RECALL_STOPWORDS:
            continue
        out.add(tok)
    return out


def _recall_answer_is_grounded(
    answer: str,
    query: str,
    evidence: List[Dict[str, str]],
    *,
    cfg_get: Callable[[str, Any], Any],
) -> bool:
    """
    Conservative heuristic: reject semantic answers that introduce too many
    out-of-evidence significant terms.
    """
    ans = str(answer or "").strip()
    if not ans:
        return False
    evidence_blob = " ".join(
        f"{e.get('id','')} {e.get('matrix_type','')} {e.get('object','')} {e.get('text','')}"
        for e in (evidence or [])
    )
    allowed = _recall_sig_tokens(evidence_blob) | _recall_sig_tokens(query)
    answer_terms = _recall_sig_tokens(ans)
    if not answer_terms:
        return True
    unknown = sorted(t for t in answer_terms if t not in allowed)
    max_ratio = float(cfg_get("recall.semantic_unknown_ratio_max", 0.28))
    unknown_ratio = len(unknown) / max(1, len(answer_terms))
    if len(answer_terms) <= 4 and len(unknown) >= 1:
        return False
    if len(unknown) >= 4 and unknown_ratio > max_ratio:
        return False
    return True


def synthesize_recall_answer(
    *,
    draft: str,
    query: str,
    evidence: List[Dict[str, str]],
    cfg_get: Callable[[str, Any], Any],
    call_model_prompt: Callable[..., str],
) -> str:
    """
    LLM formatting layer for recall answers, grounded to deterministic evidence.
    Falls back to deterministic draft on uncertainty.
    """
    d = str(draft or "").strip()
    if not d:
        return ""
    if _should_skip_semantic_recall(d):
        return d
    if not evidence:
        return d

    max_tokens = int(cfg_get("recall.semantic_max_tokens", 260))
    temperature = float(cfg_get("recall.semantic_temperature", 0.1))
    top_p = float(cfg_get("recall.semantic_top_p", 0.9))
    role = str(cfg_get("recall.semantic_role", "thinker") or "thinker")

    ev_lines: List[str] = []
    for e in evidence[:10]:
        eid = str(e.get("id") or "").strip() or "E?"
        mt = str(e.get("matrix_type") or "").strip()
        obj = str(e.get("object") or "").strip()
        txt = str(e.get("text") or "").strip()
        meta = " | ".join(x for x in [mt, obj] if x)
        prefix = f"- [{eid}]"
        if meta:
            prefix += f" ({meta})"
        ev_lines.append(f"{prefix} {txt}".rstrip())
    evidence_block = "\n".join(ev_lines)

    prompt = (
        "You are a strict recall formatter.\n"
        "Rewrite the deterministic draft into a concise, human-readable answer.\n"
        "Grounding rules:\n"
        "- Use only the facts from EVIDENCE and DRAFT.\n"
        "- Do not add new entities, products, promises, or tools.\n"
        "- Preserve yes/no verdicts.\n"
        "- Keep answer compact and direct.\n"
        "- If uncertain, keep the draft wording.\n"
        "- Output only final answer text.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"DRAFT:\n{d}\n\n"
        f"EVIDENCE:\n{evidence_block}\n"
    )
    out = call_model_prompt(
        role=role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ).strip()
    if not out:
        return d
    if out.startswith("[router error:") or out.startswith("[model '"):
        return d
    if not _recall_answer_is_grounded(out, query, evidence, cfg_get=cfg_get):
        return d
    return out

