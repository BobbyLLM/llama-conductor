from __future__ import annotations

from typing import Any, Callable, Dict


def build_fs_facts(
    *,
    query: str,
    state: Any,
    build_locked_summ_facts_block_fn: Callable[..., Any] | None,
    build_fs_facts_block_fn: Callable[..., Any] | None,
    cfg_get_fn: Callable[[str, Any], Any],
    fs_top_k: int,
    fs_max_chars: int,
    kb_paths: Dict[str, str],
    vault_kb_name: str,
) -> str:
    """Build facts block from filesystem KBs."""
    q = " ".join((query or "").split()).strip()
    state.rag_last_query = q
    state.rag_last_hits = 0
    state.locked_last_fact_lines = 0

    if not q:
        return ""

    if state.locked_summ_path:
        if not build_locked_summ_facts_block_fn:
            return ""
        max_chars = int(cfg_get_fn("lock.max_chars", max(fs_max_chars, 12000)))
        txt, n = build_locked_summ_facts_block_fn(
            query=q,
            kb=state.locked_summ_kb or "locked",
            file=state.locked_summ_file or "SUMM_locked.md",
            rel_path=state.locked_summ_rel_path or "",
            abs_path=state.locked_summ_path,
            max_chars=max_chars,
        )
        state.rag_last_hits = int(n)
        state.locked_last_fact_lines = int(n)
        return txt or ""

    if not build_fs_facts_block_fn:
        return ""

    kbs = {k for k in state.attached_kbs if k != vault_kb_name}
    if not kbs:
        return ""

    txt = build_fs_facts_block_fn(q, kbs, kb_paths, top_k=fs_top_k, max_chars=fs_max_chars)
    if txt:
        state.rag_last_hits = txt.count("[kb=")
    return txt or ""


def build_vault_facts(
    *,
    query: str,
    state: Any,
    build_rag_block_fn: Callable[..., str] | None,
    vault_kb_name: str,
) -> str:
    """Build facts block from vault."""
    q = " ".join((query or "").split()).strip()
    state.vault_last_query = q
    state.vault_last_hits = 0

    if not q or not build_rag_block_fn:
        return ""

    try:
        txt = build_rag_block_fn(q, attached_kbs={vault_kb_name})
    except Exception:
        return ""

    if txt:
        state.vault_last_hits = max(1, txt.count("\n\n"))
    return txt or ""
