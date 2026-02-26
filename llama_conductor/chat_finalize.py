from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable, List, Optional


def finalize_chat_response(
    *,
    text: str,
    user_text: str,
    state: Any,
    lock_active: bool,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    has_facts_block: bool,
    stream: bool,
    mode: str = "serious",
    sensitive_override_once: bool = False,
    bypass_serious_anti_loop: bool = False,
    serious_task_forward_fallback: str,
    make_stream_response: Callable[[str], Any],
    make_json_response: Callable[[str], Any],
    sanitize_scratchpad_grounded_output_fn: Callable[[str], str],
    append_scratchpad_provenance_fn: Callable[[str], str],
    apply_locked_output_policy_fn: Callable[[str, Any], str],
    apply_deterministic_footer_fn: Callable[..., str],
    append_profile_footer_fn: Callable[..., str],
    rewrite_response_style_fn: Optional[Callable[..., str]],
    classify_sensitive_context_fn: Callable[[str], bool],
    strip_in_body_confidence_source_claims_fn: Callable[[str], str],
    enforce_fun_antiparrot_fn: Callable[[str, str], str],
    strip_irrelevant_proofread_tail_fn: Callable[[str, str], str],
    normalize_agreement_ack_tense_fn: Callable[[str, str], str],
    classify_query_family_fn: Optional[Callable[[str], str]],
    is_ack_reframe_only_fn: Callable[[str], bool],
    strip_footer_lines_for_scan_fn: Callable[[str], str],
    normalize_signature_text_fn: Callable[[str], str],
    score_output_compliance_fn: Optional[Callable[..., float]],
    compute_effective_strength_fn: Callable[..., float],
) -> Any:
    if scratchpad_grounded:
        text = sanitize_scratchpad_grounded_output_fn(text)
        if scratchpad_quotes:
            text = (
                text.rstrip()
                + "\n\nScratchpad Quotes:\n"
                + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
            )
        text = append_scratchpad_provenance_fn(text)

    if state.attached_kbs and "Source: Model" in text and not scratchpad_grounded and not lock_active:
        kb_list = ", ".join(sorted(state.attached_kbs))
        disclaimer = (
            f"[Note: No relevant information found in attached KBs ({kb_list}). "
            f"Answer based on pre-trained data.]\n\n"
        )
        text = disclaimer + text

    if lock_active:
        text = apply_locked_output_policy_fn(text, state)

    if rewrite_response_style_fn is not None:
        try:
            sensitive = classify_sensitive_context_fn(user_text)
            text = rewrite_response_style_fn(
                text,
                enabled=bool(getattr(state, "profile_enabled", False)),
                correction_style=str(
                    getattr(getattr(state, "interaction_profile", None), "correction_style", "neutral")
                ),
                user_text=user_text,
                sensitive_context=sensitive,
                sensitive_override=bool(getattr(state.interaction_profile, "sensitive_override", False))
                or bool(sensitive_override_once),
                blocked_nicknames=sorted(getattr(state, "profile_blocked_nicknames", set())),
            )
        except Exception:
            pass

    if mode in ("fun", "fun_rewrite"):
        text = strip_in_body_confidence_source_claims_fn(text)

    if mode in ("fun", "fun_rewrite"):
        try:
            text = enforce_fun_antiparrot_fn(text, user_text)
        except Exception:
            pass

    if mode == "serious":
        text = strip_irrelevant_proofread_tail_fn(text, user_text)
        text = normalize_agreement_ack_tense_fn(text, user_text)

    if mode == "serious":
        try:
            state_like_query = False
            if classify_query_family_fn is not None:
                try:
                    state_like_query = classify_query_family_fn(user_text) in ("state_transition", "constraint_decision")
                except Exception:
                    state_like_query = False
            if bypass_serious_anti_loop or state_like_query:
                state.serious_ack_reframe_streak = 0
                state.serious_repeat_streak = 0
                state.serious_last_body_signature = ""
            else:
                if is_ack_reframe_only_fn(text):
                    if int(getattr(state, "serious_ack_reframe_streak", 0) or 0) >= 1:
                        text = serious_task_forward_fallback
                        state.serious_ack_reframe_streak = 0
                    else:
                        state.serious_ack_reframe_streak = 1
                else:
                    state.serious_ack_reframe_streak = 0
                body = strip_footer_lines_for_scan_fn(text)
                sig = normalize_signature_text_fn(body)
                prev = str(getattr(state, "serious_last_body_signature", "") or "")
                repeat = int(getattr(state, "serious_repeat_streak", 0) or 0)
                if sig and prev and len(sig) >= 40 and len(prev) >= 40:
                    sim = SequenceMatcher(None, prev, sig).ratio()
                    if sim >= 0.90:
                        repeat += 1
                    else:
                        repeat = 0
                else:
                    repeat = 0
                if repeat >= 1:
                    text = serious_task_forward_fallback
                    state.serious_repeat_streak = 0
                    state.serious_last_body_signature = ""
                else:
                    state.serious_repeat_streak = repeat
                    state.serious_last_body_signature = sig
        except Exception:
            pass

    if score_output_compliance_fn is not None and getattr(state, "profile_enabled", False):
        try:
            score = score_output_compliance_fn(
                text,
                correction_style=str(getattr(state.interaction_profile, "correction_style", "neutral")),
                user_text=user_text,
                blocked_nicknames=sorted(getattr(state, "profile_blocked_nicknames", set())),
            )
            prev = float(getattr(state, "profile_output_compliance", 0.0) or 0.0)
            state.profile_output_compliance = (prev * 0.7) + (score * 0.3)
            state.profile_effective_strength = compute_effective_strength_fn(
                state.interaction_profile,
                enabled=state.profile_enabled,
                output_compliance=state.profile_output_compliance,
            )
            u_low = (user_text or "").lower()
            if any(k in u_low for k in ("useless", "stiff", "stop talking like", "read the room", "fuck off", "bullshit")):
                state.profile_output_compliance = min(state.profile_output_compliance, 0.65)
                state.profile_effective_strength = compute_effective_strength_fn(
                    state.interaction_profile,
                    enabled=state.profile_enabled,
                    output_compliance=state.profile_output_compliance,
                )
        except Exception:
            pass

    text = apply_deterministic_footer_fn(
        text=text,
        state=state,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        has_facts_block=has_facts_block,
    )
    text = append_profile_footer_fn(text=text, state=state, user_text=user_text)

    try:
        state.last_user_text = str(user_text or "").strip()
        state.last_assistant_text = str(text or "").strip()
    except Exception:
        pass

    if state.auto_detach_after_response:
        state.attached_kbs.clear()
        state.auto_detach_after_response = False

    if stream:
        return make_stream_response(text)
    return make_json_response(text)
