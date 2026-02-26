from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


async def maybe_handle_fun_fr_raw(
    *,
    fun_mode: str,
    state: Any,
    user_text: str,
    session_id: str,
    history_text_only: List[dict],
    facts_block: str,
    constraints_block: str,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    lock_active: bool,
    stream: bool,
    sensitive_override_once: bool,
    state_solver_answer: str,
    run_sync: Callable[..., Any],
    run_fun: Optional[Callable[..., str]],
    run_raw: Optional[Callable[..., str]],
    run_serious: Callable[..., str],
    run_fun_rewrite_fallback: Callable[..., str],
    call_model_prompt: Callable[..., str],
    no_op_vodka_cls: Any,
    select_fun_style_seed: Callable[..., Dict[str, Any]],
    serious_max_tokens_for_query: Callable[[str], int],
    is_argumentative_prompt: Callable[[str], bool],
    is_argumentatively_complete: Callable[[str], bool],
    fallback_with_mode_header: Callable[[str, str], str],
    finalize_chat_response: Callable[..., Any],
) -> Optional[Any]:
    if fun_mode == "fun":
        if run_fun is None:
            base = (state_solver_answer or "").strip()
            if not base:
                base = (await run_sync(
                    run_serious,
                    session_id=session_id,
                    user_text=user_text,
                    history=history_text_only,
                    vodka=no_op_vodka_cls(),
                    call_model=call_model_prompt,
                    facts_block=facts_block,
                    constraints_block=constraints_block,
                    thinker_role="thinker",
                    max_tokens=serious_max_tokens_for_query(user_text),
                )).strip()
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base)
            quote = str(sel.get("seed") or "")
            text = f'[FUN] "{quote}"\n\n{base}' if quote else f"[FUN]\n\n{base}"
        else:
            base_preview = (state_solver_answer or "").strip()
            if not base_preview:
                base_preview = (await run_sync(
                    run_serious,
                    session_id=session_id,
                    user_text=user_text,
                    history=history_text_only,
                    vodka=no_op_vodka_cls(),
                    call_model=call_model_prompt,
                    facts_block=facts_block,
                    constraints_block=constraints_block,
                    thinker_role="thinker",
                    max_tokens=serious_max_tokens_for_query(user_text),
                )).strip()
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base_preview)
            pool = list(sel.get("pool") or [])
            seed = str(sel.get("seed") or "")

            styled = (await run_sync(
                run_fun,
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                facts_block=facts_block,
                quote_pool=pool,
                seed_override=seed,
                base_answer_override=base_preview,
                vodka=no_op_vodka_cls(),
                call_model=call_model_prompt,
                thinker_role="thinker",
            )).strip()
            lines = styled.splitlines()
            if lines:
                q = lines[0].strip()
                if q and not (q.startswith('"') and q.endswith('"')):
                    q = '"' + q.strip('"') + '"'
                lines[0] = f"[FUN] {q}" if q else "[FUN]"
                text = "\n".join(lines)
            else:
                text = "[FUN]"

            if is_argumentative_prompt(user_text) and not is_argumentatively_complete(text):
                text = fallback_with_mode_header(text, base_preview)

        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="fun",
            sensitive_override_once=sensitive_override_once,
        )

    if fun_mode == "fun_rewrite":
        base_preview = (state_solver_answer or "").strip()
        if not base_preview:
            base_preview = (await run_sync(
                run_serious,
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                vodka=no_op_vodka_cls(),
                call_model=call_model_prompt,
                facts_block=facts_block,
                constraints_block=constraints_block,
                thinker_role="thinker",
                max_tokens=serious_max_tokens_for_query(user_text),
            )).strip()
        text = (await run_sync(
            run_fun_rewrite_fallback,
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=no_op_vodka_cls(),
            facts_block=facts_block,
            state=state,
            base_override=base_preview,
        )).strip()
        if is_argumentative_prompt(user_text) and not is_argumentatively_complete(text):
            text = fallback_with_mode_header(text, base_preview)

        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="fun_rewrite",
            sensitive_override_once=sensitive_override_once,
        )

    if state.raw_sticky and run_raw:
        text = (await run_sync(
            run_raw,
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=no_op_vodka_cls(),
            call_model=call_model_prompt,
            facts_block=facts_block,
            constraints_block=constraints_block,
            thinker_role="thinker",
        )).strip()
        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="raw",
            sensitive_override_once=sensitive_override_once,
        )

    return None

