from __future__ import annotations

from typing import Any, Callable, List, Optional


async def handle_mentats_selector(
    *,
    selector: str,
    session_id: str,
    state: Any,
    user_text: str,
    run_sync: Callable[..., Any],
    run_mentats: Optional[Callable[..., str]],
    build_vault_facts: Callable[[str, Any], str],
    call_model_prompt: Callable[..., str],
    no_op_vodka_cls: Any,
    facts_collection: str,
    debug_fn: Callable[[str], None],
) -> Optional[str]:
    """Return Mentats output text when selector is 'mentats'; else None."""
    if selector != "mentats":
        return None

    debug_fn(f"[DEBUG] Mentats selector triggered")
    state.fun_sticky = False
    state.fun_rewrite_sticky = False

    if not run_mentats:
        return "[router] mentats.py not available"

    vault_facts = build_vault_facts(user_text, state)
    if not (vault_facts or "").strip():
        return (
            "[ZARDOZ HATH SPOKEN]\n\n"
            "The Vault contains no relevant knowledge for this query. I cannot reason without authoritative facts.\n\n"
            "Sources: Vault (empty)"
        )

    def _build_rag_block(query: str, collection: str = "vault") -> str:
        return build_vault_facts(query, state)

    def _build_constraints_block(query: str) -> str:
        return ""

    def _call_model_with_critic_temp(
        *,
        role: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        if role == "critic":
            temperature = 0.1
        return call_model_prompt(
            role=role,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    debug_fn(f"[DEBUG] About to call run_mentats with vault_facts length={len(vault_facts)}")
    try:
        text = (
            await run_sync(
                run_mentats,
                session_id,
                user_text,
                [],
                vodka=no_op_vodka_cls(),
                call_model=_call_model_with_critic_temp,
                build_rag_block=_build_rag_block,
                build_constraints_block=_build_constraints_block,
                facts_collection=facts_collection,
                thinker_role="thinker",
                critic_role="critic",
            )
        ).strip()
        debug_fn(f"[DEBUG] run_mentats returned {len(text)} chars")
    except Exception as e:
        debug_fn(f"[DEBUG] run_mentats crashed: {e.__class__.__name__}")
        text = "[router] Mentats crashed"

    if "[ZARDOZ HATH SPOKEN]" not in text:
        text = text.rstrip() + "\n\n[ZARDOZ HATH SPOKEN]"
    if "Sources:" not in text:
        text = text.rstrip() + "\nSources: Vault"
    return text
