"""State-solver orchestration extracted from router_fastapi.

Contains the early deterministic routing flow:
- follow-up handling over prior deterministic decision frames
- semantic choice normalization + optional semantic phrasing refine
- direct state solver dispatch for machine-checkable families
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def maybe_handle_state_solver_early(
    *,
    state: Any,
    user_text: str,
    fun_mode: str,
    lock_active_now: bool,
    stream: bool,
    sensitive_override_once: bool,
    cfg_get: Callable[[str, Any], Any],
    classify_query_family: Callable[[str], str],
    solve_state_transition_query: Callable[[str], Any],
    solve_constraint_followup: Optional[Callable[..., Any]],
    semantic_pick_clarifier_option: Callable[..., str],
    semantic_refine_constraint_choice: Callable[..., str],
    finalize_chat_response: Callable[..., Any],
    debug_fn: Callable[[str], None],
) -> Optional[Any]:
    if (
        fun_mode != ""
        or classify_query_family is None
        or solve_state_transition_query is None
        or not bool(cfg_get("state_solver.enabled", True))
        or not bool(cfg_get("state_solver.auto_route", True))
        or lock_active_now
    ):
        return None

    try:
        if (
            solve_constraint_followup is not None
            and str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
            and isinstance(getattr(state, "deterministic_last_frame", None), dict)
            and bool(getattr(state, "deterministic_last_frame", {}))
        ):
            fr = solve_constraint_followup(
                frame=dict(getattr(state, "deterministic_last_frame", {}) or {}),
                query=user_text,
            )
            if (
                not bool(getattr(fr, "handled", False))
                and isinstance(getattr(fr, "frame", None), dict)
                and bool(getattr(fr, "frame", {}))
            ):
                state.deterministic_last_frame = dict(getattr(fr, "frame", {}) or {})
                state.deterministic_last_reason = str(getattr(fr, "reason", "") or "")

            try:
                fr_reason = str(getattr(fr, "reason", "") or "")
                fr_frame = dict(getattr(fr, "frame", {}) or {})
                if (
                    bool(cfg_get("state_solver.semantic_choice.enabled", True))
                    and fr_reason in ("clarification_still_required", "clarification_user_confused_rephrase")
                    and bool(fr_frame.get("needs_clarification", False))
                    and isinstance(fr_frame.get("clarify_options", None), list)
                ):
                    options = [str(x).strip() for x in (fr_frame.get("clarify_options") or []) if str(x).strip()]
                    picked = semantic_pick_clarifier_option(user_text=user_text, options=options)
                    if picked:
                        fr2 = solve_constraint_followup(frame=fr_frame, query=picked)
                        if bool(getattr(fr2, "handled", False)):
                            fr = fr2
            except Exception:
                pass

            if bool(getattr(fr, "handled", False)):
                txt = str(getattr(fr, "answer", "") or "").strip()
                if txt:
                    try:
                        fr_frame_for_refine = dict(getattr(fr, "frame", {}) or {})
                        fr_reason_for_refine = str(getattr(fr, "reason", "") or "")
                        refined = semantic_refine_constraint_choice(
                            user_text=user_text,
                            base_answer=txt,
                            frame=fr_frame_for_refine,
                            reason=fr_reason_for_refine,
                        )
                        if refined:
                            txt = refined.rstrip() + "\nSource: Mixed"
                    except Exception:
                        pass
                    state.deterministic_last_family = str(getattr(fr, "family", "") or "constraint_decision")
                    state.deterministic_last_reason = str(getattr(fr, "reason", "") or "")
                    state.deterministic_last_answer = txt
                    state.deterministic_last_frame = dict(getattr(fr, "frame", {}) or {})
                    return finalize_chat_response(
                        text=txt,
                        user_text=user_text,
                        state=state,
                        lock_active=lock_active_now,
                        scratchpad_grounded=False,
                        scratchpad_quotes=[],
                        has_facts_block=False,
                        stream=stream,
                        mode="state_solver",
                        sensitive_override_once=sensitive_override_once,
                        bypass_serious_anti_loop=True,
                    )

        qfam0 = classify_query_family(user_text)
        debug_fn(f"[DEBUG] state_solver_early family={qfam0!r} user={user_text!r}")
        if qfam0 in ("state_transition", "constraint_decision"):
            apply_in_raw0 = bool(cfg_get("state_solver.apply_in_raw", False))
            if (not state.raw_sticky) or apply_in_raw0:
                sr0 = solve_state_transition_query(user_text)
                debug_fn(
                    "[DEBUG] state_solver_early_result "
                    f"handled={bool(getattr(sr0, 'handled', False))} "
                    f"fail_loud={bool(getattr(sr0, 'fail_loud', False))} "
                    f"reason={str(getattr(sr0, 'reason', '') or '')!r}"
                )
                if bool(getattr(sr0, "handled", False)):
                    srtxt = str(getattr(sr0, "answer", "") or "").strip()
                    if srtxt:
                        state.deterministic_last_family = str(getattr(sr0, "family", "") or qfam0)
                        state.deterministic_last_reason = str(getattr(sr0, "reason", "") or "")
                        state.deterministic_last_answer = srtxt
                        state.deterministic_last_frame = dict(getattr(sr0, "frame", {}) or {})
                        return finalize_chat_response(
                            text=srtxt,
                            user_text=user_text,
                            state=state,
                            lock_active=lock_active_now,
                            scratchpad_grounded=False,
                            scratchpad_quotes=[],
                            has_facts_block=False,
                            stream=stream,
                            mode="state_solver",
                            sensitive_override_once=sensitive_override_once,
                            bypass_serious_anti_loop=True,
                        )
    except Exception:
        return None

    return None

