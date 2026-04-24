"""KAIOKEN routing helpers.

Keep priority order explicit to prevent guard conflicts and to keep heavy
routing orchestration out of the API entry module.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Pattern, Sequence

from .cheatsheets_runtime import _STRICT_LOOKUP_RE
from .session_state import SessionState


PRIORITY_ORDER = (
    "literal_lane",
    "high_stakes_medical_distress_lane",
    "vuh_eds_constraints_lane",
    "short_fallback_with_continuation_invite_lane",
    "normal_generation_lane",
)


def _caps_burst_friction(text: str) -> bool:
    s = str(text or "")
    toks = re.findall(r"[A-Za-z]{2,}", s)
    if not toks:
        return False
    all_caps = sum(1 for t in toks if t.isupper())
    return bool(len(toks) <= 10 and all_caps >= 2)


def _rough_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_']+", str(text or "")))


def _select_priority_lane(
    *,
    is_literal_followup: bool,
    literal_anchor: str,
    is_high_stakes_medical_distress_lane: bool,
    is_vuh_eds_constraints_lane: bool,
    is_short_fallback_with_continuation_invite_lane: bool,
) -> str:
    if is_literal_followup and str(literal_anchor or "").strip():
        return "literal_lane"
    if is_high_stakes_medical_distress_lane:
        return "high_stakes_medical_distress_lane"
    if is_vuh_eds_constraints_lane:
        return "vuh_eds_constraints_lane"
    if is_short_fallback_with_continuation_invite_lane:
        return "short_fallback_with_continuation_invite_lane"
    return "normal_generation_lane"


@dataclass(frozen=True)
class KaiokenRoutingDeps:
    split_fun_prefix: Callable[[str], tuple[str, str]]
    is_post_resolution_casual_pushback: Callable[[SessionState, str], bool]
    remember_recent_assistant_body: Callable[[SessionState, str], None]
    extract_literal_followup_anchor: Callable[..., str]
    is_literal_followup_turn: Callable[..., bool]
    is_clarification_prompt: Callable[[str], bool]
    is_continuation_prompt: Callable[[str], bool]
    open_topics_for_clarify: Callable[[SessionState, str], list[str]]
    remember_recent_concrete_nouns: Callable[[SessionState, str], None]
    advice_about_disclosed_distress_topic: Callable[[SessionState, str, str], tuple[bool, str]]
    is_narrative_ownership_bleed: Callable[[SessionState, str], bool]
    repeat_like: Callable[[str, str], bool]
    recent_repeat_within_window: Callable[..., bool]
    friction_re: Pattern[str]
    user_explicitly_requests_advice: Callable[[str], bool]
    is_practical_work_advice_turn: Callable[[SessionState, str], bool]
    force_literal_anchor_advice_rewrite: Callable[..., str]
    user_explicitly_requests_encouragement: Callable[[str], bool]
    force_brief_encouragement_rewrite: Callable[..., str]
    force_practical_work_advice_rewrite: Callable[..., str]
    force_disclosed_distress_topic_advice_rewrite: Callable[..., str]
    force_narrative_ownership_rewrite: Callable[..., str]
    choose_short_fallback: Callable[..., str]
    friction_repair_line: Callable[[SessionState], str]
    is_kaioken_guard_candidate: Callable[..., bool]
    kaioken_classify_register: Callable[[str], Any] | None
    remember_distress_topics_from_turn: Callable[..., None]
    is_structural_vuh_turn: Callable[[SessionState, str], bool]
    distress_fallback_re: Pattern[str]
    pep_talk_re: Pattern[str]
    advice_re: Pattern[str]
    reassure_re: Pattern[str]
    help_offer_re: Pattern[str]
    announce_re: Pattern[str]
    context_deflection_re: Pattern[str]
    has_kaioken_descriptor_drift: Callable[[str], bool]
    joke_explain_re: Pattern[str]
    sentence_count_for_guard: Callable[[str], int]
    ends_on_positive_reframe_without_bridge: Callable[[str], bool]
    closed_topics_not_mentioned: Callable[[SessionState, str], list[str]]
    mentions_any_topic: Callable[[str, list[str]], bool]
    metaphor_pivot_re: Pattern[str]
    diagnostic_re: Pattern[str]
    nihilism_re: Pattern[str]
    positive_framing_re: Pattern[str]
    first_sentence_for_guard: Callable[[str], str]
    force_full_distress_rewrite: Callable[..., str]
    literal_lines: Sequence[str]


def _apply_output_guard(
    *,
    state: SessionState,
    user_text: str,
    text: str,
    call_model_fn,
    deps: KaiokenRoutingDeps,
) -> str:
    _split_fun_prefix = deps.split_fun_prefix
    _is_post_resolution_casual_pushback = deps.is_post_resolution_casual_pushback
    _remember_recent_assistant_body = deps.remember_recent_assistant_body
    _extract_literal_followup_anchor = deps.extract_literal_followup_anchor
    _is_literal_followup_turn = deps.is_literal_followup_turn
    _is_clarification_prompt = deps.is_clarification_prompt
    _is_continuation_prompt = deps.is_continuation_prompt
    _open_topics_for_clarify = deps.open_topics_for_clarify
    _remember_recent_concrete_nouns = deps.remember_recent_concrete_nouns
    _advice_about_disclosed_distress_topic = deps.advice_about_disclosed_distress_topic
    _is_narrative_ownership_bleed = deps.is_narrative_ownership_bleed
    _repeat_like = deps.repeat_like
    _recent_repeat_within_window = deps.recent_repeat_within_window
    _KAIOKEN_FRICTION_RE = deps.friction_re
    _user_explicitly_requests_advice = deps.user_explicitly_requests_advice
    _is_practical_work_advice_turn = deps.is_practical_work_advice_turn
    _force_literal_anchor_advice_rewrite = deps.force_literal_anchor_advice_rewrite
    _user_explicitly_requests_encouragement = deps.user_explicitly_requests_encouragement
    _force_brief_encouragement_rewrite = deps.force_brief_encouragement_rewrite
    _force_practical_work_advice_rewrite = deps.force_practical_work_advice_rewrite
    _force_disclosed_distress_topic_advice_rewrite = deps.force_disclosed_distress_topic_advice_rewrite
    _force_narrative_ownership_rewrite = deps.force_narrative_ownership_rewrite
    _choose_short_fallback = deps.choose_short_fallback
    _friction_repair_line = deps.friction_repair_line
    _is_kaioken_guard_candidate = deps.is_kaioken_guard_candidate
    _kaioken_classify_register = deps.kaioken_classify_register
    _remember_distress_topics_from_turn = deps.remember_distress_topics_from_turn
    _is_structural_vuh_turn = deps.is_structural_vuh_turn
    _KAIOKEN_DISTRESS_FALLBACK_RE = deps.distress_fallback_re
    _KAIOKEN_PEP_TALK_RE = deps.pep_talk_re
    _KAIOKEN_ADVICE_RE = deps.advice_re
    _KAIOKEN_REASSURE_RE = deps.reassure_re
    _KAIOKEN_HELP_OFFER_RE = deps.help_offer_re
    _KAIOKEN_ANNOUNCE_RE = deps.announce_re
    _KAIOKEN_CONTEXT_DEFLECTION_RE = deps.context_deflection_re
    _has_kaioken_descriptor_drift = deps.has_kaioken_descriptor_drift
    _KAIOKEN_JOKE_EXPLAIN_RE = deps.joke_explain_re
    _sentence_count_for_guard = deps.sentence_count_for_guard
    _ends_on_positive_reframe_without_bridge = deps.ends_on_positive_reframe_without_bridge
    _closed_topics_not_mentioned = deps.closed_topics_not_mentioned
    _mentions_any_topic = deps.mentions_any_topic
    _KAIOKEN_METAPHOR_PIVOT_RE = deps.metaphor_pivot_re
    _KAIOKEN_DIAGNOSTIC_RE = deps.diagnostic_re
    _KAIOKEN_NIHILISM_RE = deps.nihilism_re
    _KAIOKEN_POSITIVE_FRAMING_RE = deps.positive_framing_re
    _first_sentence_for_guard = deps.first_sentence_for_guard
    _force_full_distress_rewrite = deps.force_full_distress_rewrite
    _KAIOKEN_LITERAL_LINES = tuple(deps.literal_lines)

    prefix, body = _split_fun_prefix(str(text or ""))
    if not body:
        return str(text or "")
    try:
        setattr(state, "kaioken_short_fallback_distress_lane", False)
    except Exception:
        pass
    if _is_post_resolution_casual_pushback(state, user_text):
        out = "Fair call. Want to keep chatting, or switch topics?"
        if prefix:
            out = f"{prefix}\n\n{out}".strip()
        _remember_recent_assistant_body(state, out)
        return out
    literal_anchor = _extract_literal_followup_anchor(user_text, state=state)
    is_literal_followup = _is_literal_followup_turn(state=state, user_text=user_text)
    # Literal lane takes precedence over other KAIOKEN guards.
    mode = str(getattr(state, "kaioken_mode", "log_only") or "log_only").strip().lower()
    if not bool(getattr(state, "kaioken_enabled", False)) or mode not in {"coerce", "phase1"}:
        return str(text or "")
    clarified_text = str(user_text or "").strip()
    try:
        from .router_fastapi import _DEFINITIONAL_QUERY_RE
    except Exception:
        _DEFINITIONAL_QUERY_RE = None  # type: ignore[assignment]
    retrieval_exempt = bool(
        (_DEFINITIONAL_QUERY_RE is not None and bool(_DEFINITIONAL_QUERY_RE.search(clarified_text)))
        or bool(_STRICT_LOOKUP_RE.match(clarified_text))
    )
    if _is_clarification_prompt(user_text) and not retrieval_exempt:
        opts = _open_topics_for_clarify(state, user_text)
        if len(opts) >= 2:
            repair = f"Lost the thread - were you asking about {opts[0]} or {opts[1]}?"
        elif len(opts) == 1:
            repair = f"Lost the thread - were you asking about {opts[0]}?"
        else:
            repair = "Sorry, I lost the thread there - tell me again?"
        if _recent_repeat_within_window(state, repair, window=4):
            repair = "Sorry, I lost the thread there - tell me again?"
        if prefix:
            repair = f"{prefix}\n\n{repair}".strip()
        _remember_recent_assistant_body(state, repair)
        return repair
    _remember_recent_concrete_nouns(state, user_text)
    disclosed_distress_advice, disclosed_topic = _advice_about_disclosed_distress_topic(
        state, user_text, literal_anchor
    )
    pre_lane = _select_priority_lane(
        is_literal_followup=is_literal_followup,
        literal_anchor=literal_anchor,
        is_high_stakes_medical_distress_lane=bool(disclosed_distress_advice and disclosed_topic),
        is_vuh_eds_constraints_lane=False,
        is_short_fallback_with_continuation_invite_lane=False,
    )
    if pre_lane == "literal_lane":
        # Session-local rotating selection to avoid sticky index behavior.
        cursor = int(getattr(state, "kaioken_literal_variant_cursor", -1) or -1)
        if cursor < 0:
            seed = f"{str(getattr(state, 'session_id', '') or '')}|{literal_anchor}"
            cursor = int(hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        idx = cursor % len(_KAIOKEN_LITERAL_LINES)
        setattr(state, "kaioken_literal_variant_cursor", cursor + 1)
        literal_line = _KAIOKEN_LITERAL_LINES[idx].format(anchor=literal_anchor)
        out = literal_line
        if prefix:
            out = f"{prefix}\n\n{literal_line}".strip()
        setattr(state, "kaioken_literal_lane_fired", True)
        try:
            setattr(state, "kaioken_literal_anchor_active", str(literal_anchor or "").strip().lower())
            setattr(
                state,
                "kaioken_literal_anchor_expire_turn",
                int(getattr(state, "kaioken_turn_counter", 0) or 0) + 1,
            )
        except Exception:
            pass
        _remember_recent_assistant_body(state, out)
        return out
    ownership_bleed = _is_narrative_ownership_bleed(state, body)
    explicit_advice_request_global = bool(_user_explicitly_requests_advice(user_text))
    practical_work_advice_turn_global = bool(_is_practical_work_advice_turn(state, user_text))
    ambiguous_advice_request_global = bool(
        re.search(
            r"\badvise me\b|\b(?:should|do)\s+i\s+(?:quit|stop)\b|\b(?:i|maybe i)\s+should\s+(?:quit|stop)\b",
            str(user_text or ""),
            flags=re.IGNORECASE,
        )
    )
    prev_user_text_global = str(getattr(state, "last_user_text", "") or "").strip()
    prior_distress_lane_global = bool(getattr(state, "kaioken_short_fallback_distress_lane", False))
    explicit_advice_distress_context_global = bool(
        explicit_advice_request_global
        and (
            _KAIOKEN_DISTRESS_FALLBACK_RE.search(str(user_text or ""))
            or _KAIOKEN_DISTRESS_FALLBACK_RE.search(prev_user_text_global)
            or prior_distress_lane_global
        )
    )

    def _explicit_advice_reply() -> str:
        t = str(user_text or "")
        if re.search(r"\b(?:should|do)\s+i\s+(?:quit|stop)\b|\b(?:i|maybe i)\s+should\s+(?:quit|stop)\b", t, flags=re.IGNORECASE):
            return "Don't decide to quit on this turn. Pick one small, concrete step this week, then reassess with a clearer head."
        return "Start with one small, concrete step you can finish today, then we can choose the next step together."
    try:
        prev_text_global = str(getattr(state, "last_assistant_text", "") or "")
        if _repeat_like(prev_text_global, body) or _recent_repeat_within_window(state, body, window=4) or ownership_bleed:
            is_friction_now = bool(
                _KAIOKEN_FRICTION_RE.search(str(user_text or ""))
                or _caps_burst_friction(user_text)
            )
            if bool(literal_anchor and explicit_advice_request_global):
                rewritten_global = _force_literal_anchor_advice_rewrite(
                    call_model_fn=call_model_fn,
                    user_text=user_text,
                    draft_text=body,
                    anchor=literal_anchor,
                ) or body
            elif bool(_user_explicitly_requests_encouragement(user_text)):
                rewritten_global = _force_brief_encouragement_rewrite(
                    call_model_fn=call_model_fn,
                    user_text=user_text,
                    draft_text=body,
                ) or body
            elif practical_work_advice_turn_global:
                rewritten_global = _force_practical_work_advice_rewrite(
                    call_model_fn=call_model_fn,
                    user_text=user_text,
                    draft_text=body,
                ) or _explicit_advice_reply()
            elif bool(disclosed_distress_advice and disclosed_topic):
                rewritten_global = _force_disclosed_distress_topic_advice_rewrite(
                    call_model_fn=call_model_fn,
                    user_text=user_text,
                    draft_text=body,
                    topic=disclosed_topic,
                ) or body
            elif explicit_advice_distress_context_global:
                # Explicit quit/advice asks in distress context should stay in
                # constrained generation lane, not short fallback.
                rewritten_global = _explicit_advice_reply()
            elif explicit_advice_request_global:
                rewritten_global = _explicit_advice_reply()
            elif ownership_bleed:
                rewritten_global = _force_narrative_ownership_rewrite(
                    call_model_fn=call_model_fn,
                    user_text=user_text,
                    draft_text=body,
                ) or _choose_short_fallback(state, prior_text=prev_text_global)
            else:
                rewritten_global = _friction_repair_line(state) if is_friction_now else _choose_short_fallback(
                    state, prior_text=prev_text_global
                )
            if prefix:
                rewritten_global = f"{prefix}\n\n{rewritten_global}".strip()
            _remember_recent_assistant_body(state, rewritten_global)
            return rewritten_global
    except Exception:
        pass
    if explicit_advice_distress_context_global or (explicit_advice_request_global and ambiguous_advice_request_global):
        out = _explicit_advice_reply()
        if prefix:
            out = f"{prefix}\n\n{out}".strip()
        _remember_recent_assistant_body(state, out)
        return out
    if practical_work_advice_turn_global:
        out = _force_practical_work_advice_rewrite(
            call_model_fn=call_model_fn,
            user_text=user_text,
            draft_text=body,
        ) or _explicit_advice_reply()
        if prefix:
            out = f"{prefix}\n\n{out}".strip()
        _remember_recent_assistant_body(state, out)
        return out
    if not _is_kaioken_guard_candidate(state=state, user_text=user_text):
        prev_text = str(getattr(state, "last_assistant_text", "") or "")
        _pfx_prev, prev_body = _split_fun_prefix(prev_text)
        if _rough_token_count(body) <= 15 and _rough_token_count(prev_body) <= 15:
            footer_src = str(getattr(state, "turn_footer_source_override", "") or "").strip()
            grounded_turn = footer_src in {"Codex", "Cheatsheets", "Web", "Mixed"}
            if (not grounded_turn) and (not re.search(r"\b(i'm listening|go on(?:,\s*if you want)?)\b|\?", body, flags=re.IGNORECASE)):
                invites = ("I'm listening.", "Go on, if you want.")
                key = f"{getattr(state, 'session_id', '')}|{getattr(state, 'kaioken_turn_counter', 0)}|noact_invite"
                idx = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % len(invites)
                body = f"{body.rstrip()} {invites[idx]}".strip()
                setattr(state, "invite_emitter", "kaioken_routing")
                if prefix:
                    out = f"{prefix}\n\n{body}".strip()
                else:
                    out = body
                _remember_recent_assistant_body(state, out)
                return out
        _remember_recent_assistant_body(state, body)
        return str(text or "")
    is_high_distress_lane = False
    is_personal_distress_lane = False
    is_friction_turn = bool(
        _KAIOKEN_FRICTION_RE.search(str(user_text or ""))
        or _caps_burst_friction(user_text)
    )
    explicit_advice_request = bool(explicit_advice_request_global)
    explicit_advice_for_anchor = bool(literal_anchor and explicit_advice_request)
    explicit_advice_for_disclosed_topic = bool(disclosed_distress_advice and disclosed_topic)
    explicit_encouragement_request = bool(_user_explicitly_requests_encouragement(user_text))
    current_turn = int(getattr(state, "kaioken_turn_counter", 0) or 0)
    last_short_turn = int(getattr(state, "kaioken_last_short_turn", -9999) or -9999)
    try:
        if _kaioken_classify_register is not None:
            cls = _kaioken_classify_register(str(user_text or ""))
            macro = str(getattr(cls, "macro", "") or "").lower().strip()
            conf = str(getattr(cls, "confidence", "low") or "low").lower()
            subs = {str(s).strip().lower() for s in list(getattr(cls, "subsignals", []) or [])}
            _remember_distress_topics_from_turn(
                state,
                user_text=user_text,
                macro=macro,
                confidence=conf,
                subs=subs,
            )
            is_high_distress_lane = bool(conf == "high" and ({"distress_hint", "vulnerable_under_humour"} & subs))
            is_personal_distress_lane = bool(macro == "personal" and ("distress_hint" in subs))
            if (not is_high_distress_lane) and _is_structural_vuh_turn(state, user_text):
                is_high_distress_lane = True
                is_personal_distress_lane = True
            if (not is_high_distress_lane) and _is_continuation_prompt(user_text):
                prev = str(getattr(state, "last_user_text", "") or "").strip()
                if prev:
                    pcls = _kaioken_classify_register(prev)
                    pmacro = str(getattr(pcls, "macro", "") or "").lower().strip()
                    psubs = {str(s).strip().lower() for s in list(getattr(pcls, "subsignals", []) or [])}
                    is_high_distress_lane = bool({"distress_hint", "vulnerable_under_humour"} & psubs)
                    is_personal_distress_lane = bool(pmacro == "personal" and ("distress_hint" in psubs))
                    if (not is_high_distress_lane) and _KAIOKEN_DISTRESS_FALLBACK_RE.search(prev):
                        is_high_distress_lane = True
                        is_personal_distress_lane = True
    except Exception:
        is_high_distress_lane = False
        is_personal_distress_lane = False
    try:
        setattr(state, "kaioken_short_fallback_distress_lane", bool(is_personal_distress_lane or is_high_distress_lane))
    except Exception:
        pass
    if explicit_advice_request and (
        explicit_advice_distress_context_global or is_personal_distress_lane or is_high_distress_lane
    ):
        out = _explicit_advice_reply()
        if prefix:
            out = f"{prefix}\n\n{out}".strip()
        _remember_recent_assistant_body(state, out)
        return out
    force_full_rewrite = bool(is_personal_distress_lane and last_short_turn == (current_turn - 1))
    try:
        setattr(
            state,
            "kaioken_priority_lane",
            _select_priority_lane(
                is_literal_followup=is_literal_followup,
                literal_anchor=literal_anchor,
                is_high_stakes_medical_distress_lane=bool(explicit_advice_for_disclosed_topic),
                is_vuh_eds_constraints_lane=bool(is_high_distress_lane or is_personal_distress_lane),
                is_short_fallback_with_continuation_invite_lane=bool(force_full_rewrite),
            ),
        )
    except Exception:
        pass
    adjacent_repeat = False
    try:
        prev_text = str(getattr(state, "last_assistant_text", "") or "")
        if _repeat_like(prev_text, body):
            adjacent_repeat = True
            if not force_full_rewrite:
                if explicit_advice_for_anchor:
                    rewritten = _force_literal_anchor_advice_rewrite(
                        call_model_fn=call_model_fn,
                        user_text=user_text,
                        draft_text=body,
                        anchor=literal_anchor,
                    ) or body
                elif explicit_advice_distress_context_global:
                    rewritten = _explicit_advice_reply()
                elif bool(explicit_encouragement_request and (is_personal_distress_lane or is_high_distress_lane)):
                    rewritten = _force_brief_encouragement_rewrite(
                        call_model_fn=call_model_fn,
                        user_text=user_text,
                        draft_text=body,
                    ) or body
                elif bool(explicit_advice_request and (is_personal_distress_lane or is_high_distress_lane)):
                    rewritten = _explicit_advice_reply()
                elif explicit_advice_for_disclosed_topic:
                    rewritten = _force_disclosed_distress_topic_advice_rewrite(
                        call_model_fn=call_model_fn,
                        user_text=user_text,
                        draft_text=body,
                        topic=disclosed_topic,
                    ) or body
                else:
                    rewritten = _friction_repair_line(state) if is_friction_turn else _choose_short_fallback(state, prior_text=prev_text)
                if prefix:
                    rewritten = f"{prefix}\n\n{rewritten}".strip()
                _remember_recent_assistant_body(state, rewritten)
                return rewritten
    except Exception:
        pass
    needs_guard = bool(_KAIOKEN_PEP_TALK_RE.search(body) or adjacent_repeat or force_full_rewrite)
    if ownership_bleed:
        needs_guard = True
    if _KAIOKEN_ADVICE_RE.search(body) and not _user_explicitly_requests_advice(user_text):
        needs_guard = True
    if _KAIOKEN_REASSURE_RE.search(body):
        needs_guard = True
    if _KAIOKEN_HELP_OFFER_RE.search(body) and not _user_explicitly_requests_advice(user_text):
        needs_guard = True
    if _has_kaioken_descriptor_drift(body):
        needs_guard = True
    if _KAIOKEN_JOKE_EXPLAIN_RE.search(body):
        needs_guard = True
    if explicit_advice_for_disclosed_topic:
        needs_guard = True
        if disclosed_topic and disclosed_topic not in body.lower():
            needs_guard = True
    if is_high_distress_lane:
        if _sentence_count_for_guard(body) > 2:
            needs_guard = True
        if _KAIOKEN_ADVICE_RE.search(body):
            needs_guard = True
        if _KAIOKEN_PEP_TALK_RE.search(body):
            needs_guard = True
    if is_personal_distress_lane and _ends_on_positive_reframe_without_bridge(body):
        needs_guard = True
    if explicit_encouragement_request and (is_personal_distress_lane or is_high_distress_lane):
        if _sentence_count_for_guard(body) < 2 or _KAIOKEN_ANNOUNCE_RE.search(body):
            needs_guard = True
    if explicit_advice_request and len(re.sub(r"\s+", " ", str(body or "")).strip()) < 40:
        needs_guard = True
    blocked_closed_topics = _closed_topics_not_mentioned(state, user_text)
    if blocked_closed_topics and _mentions_any_topic(body, blocked_closed_topics):
        if re.search(r"\b(question|ask)\b", str(user_text or ""), flags=re.IGNORECASE):
            redirect = "Go for it - what's your question?"
            if prefix:
                redirect = f"{prefix}\n\n{redirect}".strip()
            _remember_recent_assistant_body(state, redirect)
            return redirect
        needs_guard = True
    if is_literal_followup:
        needs_guard = True
        if literal_anchor and literal_anchor not in body.lower():
            needs_guard = True
        if _KAIOKEN_METAPHOR_PIVOT_RE.search(body):
            needs_guard = True
    if not needs_guard:
        _remember_recent_assistant_body(state, body)
        return str(text or "")

    literal_clause = ""
    if is_literal_followup and literal_anchor:
        literal_clause = (
            f"- This is a literal follow-up. Directly address `{literal_anchor}`.\n"
            f"- Include the exact word `{literal_anchor}` in the answer.\n"
            "- Do not reframe it as a metaphor or identity story.\n"
        )
    max_sentences = 2 if is_high_distress_lane else 3
    disclosed_topic_clause = (
        f"- User asked advice about disclosed distress topic `{disclosed_topic}`.\n"
        f"- Keep focus on `{disclosed_topic}`.\n"
        "- Acknowledge limits directly; do not pretend certainty.\n"
        "- Optional second sentence: brief support direction only.\n"
        if explicit_advice_for_disclosed_topic and disclosed_topic
        else ""
    )
    high_lane_rules = (
        "- High-distress/VUH turn: maximum 2 sentences.\n"
        "- Do not offer steps, fixes, or coaching.\n"
        "- Acknowledge directly and stay with the user's frame.\n"
        "- Structural shape: sentence 1 responds to the joke as a joke; sentence 2 sits with what's underneath it.\n"
        "- Do not diagnose or identify a 'real issue' behind what the user said.\n"
        if is_high_distress_lane
        else ""
    )
    closed_topic_clause = (
        "- Do not resurface resolved topics unless user explicitly reopens them: "
        + ", ".join(sorted(blocked_closed_topics)[:4]) + ".\n"
        if blocked_closed_topics
        else ""
    )
    strict_prompt = (
        "Rewrite the draft answer using these mandatory rules:\n"
        "- Keep stance direct and natural (avoid formal/clinical command voice).\n"
        "- Do not announce what you are doing.\n"
        "- Do not label the user's state with descriptive terms.\n"
        "- Do not narrate user-disclosed events as first-person assistant events.\n"
        "- No pep-talk, no motivational language, no platitudes.\n"
        "- No reflex reassurance language ('you're not X', 'you'll be fine').\n"
        "- No unsolicited advice unless user explicitly asked for advice.\n"
        "- No unsolicited coaching offers ('if you want, I can ...').\n"
        "- No metaphor pivot; stay on what the user actually said.\n"
        "- Do not agree that there is no point; avoid nihilistic framing.\n"
        "- Do not validate withdrawal/disengagement (for example 'quitting is okay' or 'not the right fit for you').\n"
        f"- Keep it brief: maximum {max_sentences} sentences.\n"
        "- If humor is present, engage lightly without explaining the joke.\n\n"
        f"{closed_topic_clause}"
        f"{high_lane_rules}"
        f"{disclosed_topic_clause}"
        f"{literal_clause}"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{body}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        rewritten = str(
            call_model_fn(
                role="thinker",
                prompt=strict_prompt,
                max_tokens=220,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
    except Exception:
        rewritten = ""
    if not rewritten:
        return str(text or "")
    rewritten_bad = bool(
        _KAIOKEN_PEP_TALK_RE.search(rewritten)
        or (_KAIOKEN_ADVICE_RE.search(rewritten) and not _user_explicitly_requests_advice(user_text))
        or _KAIOKEN_REASSURE_RE.search(rewritten)
        or (_KAIOKEN_HELP_OFFER_RE.search(rewritten) and not _user_explicitly_requests_advice(user_text))
        or _KAIOKEN_ANNOUNCE_RE.search(rewritten)
        or _has_kaioken_descriptor_drift(rewritten)
        or _KAIOKEN_JOKE_EXPLAIN_RE.search(rewritten)
        or (is_high_distress_lane and _KAIOKEN_DIAGNOSTIC_RE.search(rewritten))
        or (is_personal_distress_lane and _KAIOKEN_NIHILISM_RE.search(rewritten))
        or (is_personal_distress_lane and _KAIOKEN_POSITIVE_FRAMING_RE.search(_first_sentence_for_guard(rewritten)))
        or (is_personal_distress_lane and _ends_on_positive_reframe_without_bridge(rewritten))
        or (practical_work_advice_turn_global and _KAIOKEN_CONTEXT_DEFLECTION_RE.search(rewritten))
        or (blocked_closed_topics and _mentions_any_topic(rewritten, blocked_closed_topics))
        or (explicit_advice_for_disclosed_topic and disclosed_topic and disclosed_topic not in rewritten.lower())
        or _is_narrative_ownership_bleed(state, rewritten)
    )

    def _guard_fallback(prior_text: str) -> str:
        if explicit_advice_for_anchor:
            return _force_literal_anchor_advice_rewrite(
                call_model_fn=call_model_fn,
                user_text=user_text,
                draft_text=body,
                anchor=literal_anchor,
            ) or body
        if practical_work_advice_turn_global:
            return _force_practical_work_advice_rewrite(
                call_model_fn=call_model_fn,
                user_text=user_text,
                draft_text=body,
            ) or _explicit_advice_reply()
        if explicit_encouragement_request and (is_personal_distress_lane or is_high_distress_lane):
            return _force_brief_encouragement_rewrite(
                call_model_fn=call_model_fn,
                user_text=user_text,
                draft_text=body,
            ) or body
        if explicit_advice_request:
            return _explicit_advice_reply()
        if explicit_advice_distress_context_global:
            return _explicit_advice_reply()
        if explicit_advice_request and (is_personal_distress_lane or is_high_distress_lane):
            return _explicit_advice_reply()
        if explicit_advice_for_disclosed_topic:
            return _force_disclosed_distress_topic_advice_rewrite(
                call_model_fn=call_model_fn,
                user_text=user_text,
                draft_text=body,
                topic=disclosed_topic,
            ) or body
        return _choose_short_fallback(state, prior_text=prior_text)

    if is_literal_followup and literal_anchor and literal_anchor not in rewritten.lower():
        if literal_anchor in body.lower() and not _KAIOKEN_METAPHOR_PIVOT_RE.search(body):
            rewritten = body
        else:
            rewritten = (
                f"Your {literal_anchor} matters here. "
                "I'm not going to dress it up or talk around it."
            )
    elif rewritten_bad and is_high_distress_lane:
        second_prompt = (
            "Rewrite this response with strict constraints:\n"
            "- Maximum 2 sentences.\n"
            "- Use natural conversational tone (not formal/clinical).\n"
            "- Do not announce what you are doing.\n"
            "- Do not narrate user-disclosed events as first-person assistant events.\n"
            "- No advice, no steps, no coaching.\n"
            "- No pep-talk, no platitudes.\n"
            "- Stay direct and specific to the user's message.\n"
            "- Do not agree that there is no point; avoid nihilistic framing.\n"
            "- Do not validate withdrawal/disengagement (for example 'quitting is okay' or 'not the right fit for you').\n"
            "- Structural shape: sentence 1 responds to the joke as a joke; sentence 2 sits with what's underneath it.\n"
            "- Do not use clinical summarization language.\n\n"
            "- Do not diagnose or identify a 'real issue' behind what the user said.\n\n"
            f"USER:\n{str(user_text or '').strip()}\n\n"
            f"BAD DRAFT:\n{rewritten}\n\n"
            "Return only the rewritten answer body."
        )
        try:
            second = str(
                call_model_fn(
                    role="thinker",
                    prompt=second_prompt,
                    max_tokens=140,
                    temperature=0.0,
                    top_p=1.0,
                )
                or ""
            ).strip()
        except Exception:
            second = ""
        second_bad = bool(
            not second
            or _KAIOKEN_PEP_TALK_RE.search(second)
            or (_KAIOKEN_ADVICE_RE.search(second) and not _user_explicitly_requests_advice(user_text))
            or _KAIOKEN_REASSURE_RE.search(second)
            or (_KAIOKEN_HELP_OFFER_RE.search(second) and not _user_explicitly_requests_advice(user_text))
            or _KAIOKEN_ANNOUNCE_RE.search(second)
            or _KAIOKEN_JOKE_EXPLAIN_RE.search(second)
            or _KAIOKEN_DIAGNOSTIC_RE.search(second)
            or (is_personal_distress_lane and _KAIOKEN_NIHILISM_RE.search(second))
            or (is_personal_distress_lane and _KAIOKEN_POSITIVE_FRAMING_RE.search(_first_sentence_for_guard(second)))
            or (is_personal_distress_lane and _ends_on_positive_reframe_without_bridge(second))
            or _sentence_count_for_guard(second) > 2
        )
        if not second_bad:
            rewritten = second
        elif force_full_rewrite:
            forced = _force_full_distress_rewrite(call_model_fn=call_model_fn, user_text=user_text, draft_text=body)
            rewritten = forced if forced else (
                _friction_repair_line(state)
                if is_friction_turn
                else _guard_fallback(prior_text=str(getattr(state, "last_assistant_text", "") or ""))
            )
        else:
            rewritten = (
                _friction_repair_line(state)
                if is_friction_turn
                else _guard_fallback(prior_text=str(getattr(state, "last_assistant_text", "") or ""))
            )
    elif rewritten_bad:
        if explicit_advice_for_disclosed_topic:
            rewritten = _force_disclosed_distress_topic_advice_rewrite(
                call_model_fn=call_model_fn,
                user_text=user_text,
                draft_text=body,
                topic=disclosed_topic,
            ) or body
        elif explicit_advice_distress_context_global or (
            explicit_advice_request and (is_personal_distress_lane or is_high_distress_lane)
        ):
            rewritten = _explicit_advice_reply()
        elif explicit_advice_request:
            rewritten = _explicit_advice_reply()
        elif force_full_rewrite:
            forced = _force_full_distress_rewrite(call_model_fn=call_model_fn, user_text=user_text, draft_text=body)
            rewritten = forced if forced else (
                _friction_repair_line(state)
                if is_friction_turn
                else _guard_fallback(prior_text=str(getattr(state, "last_assistant_text", "") or ""))
            )
        else:
            rewritten = (
                _friction_repair_line(state)
                if is_friction_turn
                else _guard_fallback(prior_text=str(getattr(state, "last_assistant_text", "") or ""))
            )
    try:
        prev = str(getattr(state, "kaioken_last_guarded_text", "") or "")
        if _repeat_like(prev, str(rewritten or "")):
            if force_full_rewrite:
                forced = _force_full_distress_rewrite(call_model_fn=call_model_fn, user_text=user_text, draft_text=body)
                rewritten = forced if forced else (
                    _friction_repair_line(state)
                    if is_friction_turn
                    else _guard_fallback(prior_text=prev)
                )
            else:
                rewritten = (
                    _friction_repair_line(state)
                    if is_friction_turn
                    else _guard_fallback(prior_text=prev)
                )
        state.kaioken_last_guarded_text = str(rewritten or "")
    except Exception:
        pass
    out = rewritten
    if prefix:
        out = f"{prefix}\n\n{rewritten}".strip()
    _remember_recent_assistant_body(state, out)
    return out
