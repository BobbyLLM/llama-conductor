"""Correction-bind orchestration extracted from router_fastapi.

Deterministic-first correction handling for disengaged decision-lane follow-ups.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def maybe_handle_correction_bind(
    *,
    state: Any,
    user_text: str,
    history_text_only: List[Dict[str, Any]],
    query_family: str,
    lock_active: bool,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    facts_block: str,
    stream: bool,
    sensitive_override_once: bool,
    cfg_get: Callable[[str, Any], Any],
    call_model_messages: Callable[..., str],
    is_correction_intent_query: Callable[[str], bool],
    is_explicit_reengage_query: Callable[[str], bool],
    extract_numeric_correction: Callable[[str], tuple[str, str, str]],
    resolve_old_from_prior_answer: Callable[..., str],
    fallback_contextual_correction: Callable[..., str],
    strip_got_it_prefix: Callable[[str], str],
    last_assistant_text: Callable[[List[Dict[str, Any]]], str],
    last_user_text_before: Callable[[List[Dict[str, Any]], str], str],
    last_user_text: Callable[[List[Dict[str, Any]]], str],
    last_non_correction_user_text: Callable[[List[Dict[str, Any]], str], str],
    to_km: Callable[[str, str], Optional[float]],
    maybe_apply_consistency_verifier: Callable[..., str],
    finalize_chat_response: Callable[..., Any],
) -> Optional[Any]:
    """Try correction-bind flow and return finalized router response when handled."""
    try:
        fr = dict(getattr(state, "deterministic_last_frame", {}) or {})
        lane_disengaged = (
            str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
            and str(fr.get("kind") or "") == "option_feasibility"
            and bool(fr.get("decision_lane_disengaged", False))
        )
        correction_bind = is_correction_intent_query(user_text) and not is_explicit_reengage_query(user_text)
        if not correction_bind:
            return None

        prev_user_turn = str(getattr(state, "last_user_text", "") or "").strip()
        include_ack = not is_correction_intent_query(prev_user_turn)
        prior_answer = str(getattr(state, "last_assistant_text", "") or "").strip() or last_assistant_text(history_text_only)
        state_last_user = str(getattr(state, "last_user_text", "") or "").strip()
        if state_last_user and not is_correction_intent_query(state_last_user):
            prior_user = state_last_user
        else:
            prior_user = (
                last_non_correction_user_text(history_text_only, current_user_text=user_text)
                or last_user_text_before(history_text_only, user_text)
                or last_user_text(history_text_only)
            )

        def _apply_correction_distance_to_frame(new_v: str, unit_v: str) -> None:
            try:
                km = to_km(new_v, unit_v)
                if km is None:
                    return
                fr_cur = dict(getattr(state, "deterministic_last_frame", {}) or {})
                if (
                    str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
                    and str(fr_cur.get("kind") or "") == "option_feasibility"
                ):
                    fr_cur["distance_km"] = float(km)
                    if bool(fr_cur.get("decision_lane_disengaged", False)):
                        fr_cur["decision_lane_disengaged"] = True
                    state.deterministic_last_frame = fr_cur
            except Exception:
                return

        if prior_answer:
            new_v, unit_v, old_v = extract_numeric_correction(user_text)
            old_v = resolve_old_from_prior_answer(
                new=new_v,
                unit=unit_v,
                old=old_v,
                prior_answer=prior_answer,
            )
            if new_v and unit_v and prior_user:
                fb = fallback_contextual_correction(
                    prior_user=prior_user,
                    new=new_v,
                    unit=unit_v,
                    old=old_v,
                    include_ack=include_ack,
                ).strip()
                if fb:
                    _apply_correction_distance_to_frame(new_v, unit_v)
                    return finalize_chat_response(
                        text=fb,
                        user_text=user_text,
                        state=state,
                        lock_active=lock_active,
                        scratchpad_grounded=scratchpad_grounded,
                        scratchpad_quotes=scratchpad_quotes,
                        has_facts_block=bool((facts_block or "").strip()),
                        stream=stream,
                        mode="serious",
                        sensitive_override_once=sensitive_override_once,
                        bypass_serious_anti_loop=True,
                    )

            role = str(cfg_get("state_solver.correction_bind.role", "thinker") or "thinker").strip() or "thinker"
            max_tokens = int(cfg_get("state_solver.correction_bind.max_tokens", 240))
            temperature = float(cfg_get("state_solver.correction_bind.temperature", 0.0))
            top_p = float(cfg_get("state_solver.correction_bind.top_p", 0.9))
            corr_msgs = [
                {
                    "role": "system",
                    "content": (
                        "Revise the immediately previous assistant answer using the user's correction.\n"
                        "Output the revised answer only.\n"
                        "Rules:\n"
                        "- Stay on the same topic as the previous assistant answer unless user correction explicitly changes topic.\n"
                        "- Apply corrected values/details directly.\n"
                        "- Do not revert to older branches from broader conversation.\n"
                        "- Do not output meta statements like 'distance corrected' or 'recalculate'.\n"
                        "- Keep the answer concrete and practical."
                    ),
                },
                {"role": "user", "content": f"Original user request: {prior_user}"},
                {"role": "assistant", "content": prior_answer},
                {"role": "user", "content": f"Correction to apply: {user_text}\nRewrite the previous answer with this correction."},
            ]
            corrected = str(
                call_model_messages(
                    role=role,
                    messages=corr_msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                or ""
            ).strip()

            if corrected and ("distance has been corrected" in corrected.lower() or "re-evaluated" in corrected.lower()):
                new_v, unit_v, old_v = extract_numeric_correction(user_text)
                old_v = resolve_old_from_prior_answer(
                    new=new_v,
                    unit=unit_v,
                    old=old_v,
                    prior_answer=prior_answer,
                )
                fb = fallback_contextual_correction(
                    prior_user=prior_user,
                    new=new_v,
                    unit=unit_v,
                    old=old_v,
                    include_ack=include_ack,
                ).strip()
                if fb:
                    corrected = fb

            if corrected and not include_ack:
                corrected = strip_got_it_prefix(corrected)
            if corrected:
                corrected = maybe_apply_consistency_verifier(
                    user_text=user_text,
                    draft_text=corrected,
                    query_family=query_family,
                    lock_active=lock_active,
                    state_solver_used=False,
                    prior_user_text=str(getattr(state, "last_user_text", "") or "").strip(),
                )
                new_v, unit_v, _ = extract_numeric_correction(user_text)
                if new_v and unit_v:
                    _apply_correction_distance_to_frame(new_v, unit_v)
                return finalize_chat_response(
                    text=corrected,
                    user_text=user_text,
                    state=state,
                    lock_active=lock_active,
                    scratchpad_grounded=scratchpad_grounded,
                    scratchpad_quotes=scratchpad_quotes,
                    has_facts_block=bool((facts_block or "").strip()),
                    stream=stream,
                    mode="serious",
                    sensitive_override_once=sensitive_override_once,
                    bypass_serious_anti_loop=True,
                )

        # Keep prior behavior: no direct response here; caller can append correction hint.
        _ = lane_disengaged  # explicit reference; lane state is still represented in state frame.
        return None
    except Exception:
        return None
