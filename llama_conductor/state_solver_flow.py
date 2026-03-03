"""State-solver orchestration extracted from router_fastapi.

Contains the early deterministic routing flow:
- follow-up handling over prior deterministic decision frames
- semantic choice normalization + optional semantic phrasing refine
- direct state solver dispatch for machine-checkable families
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .state_mode_policy import should_skip_state_solver_early


def maybe_handle_state_solver_early(
    *,
    state: Any,
    user_text: str,
    selector: str,
    fun_mode: str,
    lock_active_now: bool,
    stream: bool,
    sensitive_override_once: bool,
    cfg_get: Callable[[str, Any], Any],
    classify_constraint_turn: Optional[Callable[..., str]],
    classify_query_family: Callable[[str], str],
    solve_state_transition_query: Callable[[str], Any],
    solve_constraint_followup: Optional[Callable[..., Any]],
    semantic_pick_clarifier_option: Callable[..., str],
    semantic_refine_constraint_choice: Callable[..., str],
    finalize_chat_response: Callable[..., Any],
    debug_fn: Callable[[str], None],
) -> Optional[Any]:
    if should_skip_state_solver_early(
        fun_mode=fun_mode,
        selector=selector,
        lock_active_now=lock_active_now,
        classify_query_family=classify_query_family,
        solve_state_transition_query=solve_state_transition_query,
        cfg_get=cfg_get,
    ):
        return None

    try:
        constraint_turn_kind = "followup_or_other"
        last_constraint_frame = dict(getattr(state, "deterministic_last_frame", {}) or {})
        if (
            classify_constraint_turn is not None
            and str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
            and bool(last_constraint_frame)
        ):
            try:
                constraint_turn_kind = str(
                    classify_constraint_turn(user_text, frame=last_constraint_frame) or "followup_or_other"
                ).strip().lower()
            except Exception:
                constraint_turn_kind = "followup_or_other"

        if constraint_turn_kind == "asset_shift":
            # Explicit topic/entity shift invalidates sticky constraint frame.
            state.deterministic_last_frame = {}
            state.deterministic_last_family = ""
            state.deterministic_last_reason = "constraint_asset_shift_reset"
            state.deterministic_last_answer = ""

        if (
            solve_constraint_followup is not None
            and constraint_turn_kind not in ("fresh_decision", "asset_shift")
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
        if qfam0 == "other" and constraint_turn_kind == "fresh_decision":
            qfam0 = "constraint_decision"
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


def resolve_state_solver_for_turn(
    *,
    state: Any,
    user_text: str,
    lock_active: bool,
    cfg_get: Callable[[str, Any], Any],
    classify_constraint_turn: Optional[Callable[..., str]],
    classify_query_family: Optional[Callable[[str], str]],
    solve_state_transition_query: Optional[Callable[[str], Any]],
    solve_constraint_followup: Optional[Callable[..., Any]],
    norm_query_text: Callable[[str], str],
    extract_replay_base_answer: Callable[[str], str],
    delegate_decision_match: Callable[[str], bool],
) -> Dict[str, Any]:
    """Resolve deterministic state-solver outputs for the current user turn.

    Consolidates the prior multi-pass router logic into one coordinator while
    preserving behavior and reason labeling.
    """
    query_family = "other"
    state_solver_used = False
    state_solver_fail_loud = False
    state_solver_answer = ""
    state_solver_reason = ""
    state_solver_frame: Dict[str, Any] = {}

    user_q_norm = norm_query_text(user_text)
    prev_user_q_norm = norm_query_text(getattr(state, "last_user_text", ""))
    last_asst_text = str(getattr(state, "last_assistant_text", "") or "")

    # Delegate short-circuit replay from immediately prior deterministic/contextual answer.
    if (
        delegate_decision_match(user_text)
        and ("source: contextual" in last_asst_text.lower())
        and ("quick check: did you mean" not in last_asst_text.lower())
    ):
        replay_base = extract_replay_base_answer(last_asst_text)
        if replay_base:
            state_solver_used = True
            state_solver_reason = "delegate_replay_last_contextual"
            state_solver_answer = replay_base
            state_solver_frame = dict(getattr(state, "deterministic_last_frame", {}) or {})
            query_family = "constraint_decision"

    # Exact-repeat replay cache for active constraint lane.
    try:
        immediate_delegate_replay = (
            bool(delegate_decision_match(user_text))
            and ("source: contextual" in str(getattr(state, "last_assistant_text", "") or "").lower())
            and (
                bool(str(getattr(state, "deterministic_last_answer", "") or "").strip())
                or ("destination" in str(getattr(state, "last_assistant_text", "") or "").lower())
            )
        )
        if (
            bool(cfg_get("state_solver.enabled", True))
            and bool(cfg_get("state_solver.auto_route", True))
            and (
                str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
                or immediate_delegate_replay
            )
            and (
                bool(str(getattr(state, "deterministic_last_answer", "") or "").strip())
                or bool(extract_replay_base_answer(getattr(state, "last_assistant_text", "")))
            )
            and user_q_norm
            and (
                user_q_norm == str(getattr(state, "deterministic_last_query_norm", "") or "")
                or user_q_norm == prev_user_q_norm
            )
        ):
            state_solver_used = True
            state_solver_reason = "replay_exact_query"
            state_solver_answer = str(getattr(state, "deterministic_last_answer", "") or "").strip() or extract_replay_base_answer(
                getattr(state, "last_assistant_text", "")
            )
            state_solver_frame = dict(getattr(state, "deterministic_last_frame", {}) or {})
            query_family = "constraint_decision"
    except Exception:
        pass

    constraint_turn_kind = "followup_or_other"
    last_constraint_frame = dict(getattr(state, "deterministic_last_frame", {}) or {})
    if (
        classify_constraint_turn is not None
        and str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
        and bool(last_constraint_frame)
    ):
        try:
            constraint_turn_kind = str(
                classify_constraint_turn(user_text, frame=last_constraint_frame) or "followup_or_other"
            ).strip().lower()
        except Exception:
            constraint_turn_kind = "followup_or_other"

    if constraint_turn_kind == "asset_shift":
        state.deterministic_last_frame = {}
        state.deterministic_last_family = ""
        state.deterministic_last_reason = "constraint_asset_shift_reset"
        state.deterministic_last_answer = ""
        state.deterministic_last_query_norm = ""

    if classify_query_family is not None:
        try:
            query_family = classify_query_family(user_text)
        except Exception:
            pass
    if query_family == "other" and constraint_turn_kind == "fresh_decision":
        query_family = "constraint_decision"

    # Pass 1: follow-up deterministic decision handling.
    if (
        (not state_solver_used)
        and solve_constraint_followup is not None
        and constraint_turn_kind not in ("fresh_decision", "asset_shift")
        and query_family not in ("state_transition", "constraint_decision")
        and str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
        and isinstance(getattr(state, "deterministic_last_frame", None), dict)
        and bool(getattr(state, "deterministic_last_frame", {}))
        and bool(cfg_get("state_solver.enabled", True))
        and bool(cfg_get("state_solver.auto_route", True))
        and not lock_active
    ):
        try:
            apply_in_raw = bool(cfg_get("state_solver.apply_in_raw", False))
            if (not state.raw_sticky) or apply_in_raw:
                fr0 = solve_constraint_followup(
                    frame=dict(getattr(state, "deterministic_last_frame", {}) or {}),
                    query=user_text,
                )
                fr0_frame = dict(getattr(fr0, "frame", {}) or {})
                if fr0_frame:
                    state.deterministic_last_frame = dict(fr0_frame)
                if bool(getattr(fr0, "handled", False)):
                    state_solver_used = True
                    state_solver_fail_loud = bool(getattr(fr0, "fail_loud", False))
                    _fr0_reason = str(getattr(fr0, "reason", "") or "").strip()
                    state_solver_reason = f"followup_pass1:{_fr0_reason}" if _fr0_reason else "followup_pass1"
                    state_solver_answer = str(getattr(fr0, "answer", "") or "").strip()
                    state_solver_frame = fr0_frame
                    query_family = "constraint_decision"
        except Exception:
            pass

    # Pass 2: direct state/constraint deterministic solve.
    if classify_query_family is not None and solve_state_transition_query is not None:
        try:
            if query_family == "other":
                query_family = classify_query_family(user_text)
            if (
                not state_solver_used
                and bool(cfg_get("state_solver.enabled", True))
                and bool(cfg_get("state_solver.auto_route", True))
                and query_family in ("state_transition", "constraint_decision")
                and not lock_active
            ):
                apply_in_raw = bool(cfg_get("state_solver.apply_in_raw", False))
                if (not state.raw_sticky) or apply_in_raw:
                    sr = solve_state_transition_query(user_text)
                    if bool(getattr(sr, "handled", False)):
                        state_solver_used = True
                        state_solver_fail_loud = bool(getattr(sr, "fail_loud", False))
                        _sr_reason = str(getattr(sr, "reason", "") or "").strip()
                        state_solver_reason = f"state_solver_pass2:{_sr_reason}" if _sr_reason else "state_solver_pass2"
                        state_solver_answer = str(getattr(sr, "answer", "") or "").strip()
                        state_solver_frame = dict(getattr(sr, "frame", {}) or {})
        except Exception:
            pass

    # Pass 3: defensive re-check if still unresolved.
    if (
        (not state_solver_used or not (state_solver_answer or "").strip())
        and query_family in ("state_transition", "constraint_decision")
        and classify_query_family is not None
        and solve_state_transition_query is not None
        and bool(cfg_get("state_solver.enabled", True))
        and bool(cfg_get("state_solver.auto_route", True))
        and not lock_active
    ):
        try:
            apply_in_raw = bool(cfg_get("state_solver.apply_in_raw", False))
            if (not state.raw_sticky) or apply_in_raw:
                sr2 = solve_state_transition_query(user_text)
                if bool(getattr(sr2, "handled", False)):
                    state_solver_used = True
                    state_solver_fail_loud = bool(getattr(sr2, "fail_loud", False))
                    _sr2_reason = str(getattr(sr2, "reason", "") or "").strip()
                    state_solver_reason = f"state_solver_pass3:{_sr2_reason}" if _sr2_reason else "state_solver_pass3"
                    state_solver_answer = str(getattr(sr2, "answer", "") or "").strip()
                    state_solver_frame = dict(getattr(sr2, "frame", {}) or {})
        except Exception:
            pass

    return {
        "query_family": query_family,
        "state_solver_used": state_solver_used,
        "state_solver_fail_loud": state_solver_fail_loud,
        "state_solver_answer": state_solver_answer,
        "state_solver_reason": state_solver_reason,
        "state_solver_frame": state_solver_frame,
        "user_q_norm": user_q_norm,
    }
