"""FastAPI orchestration layer for llama-conductor.

Responsibilities:
- expose API routes
- route requests across command/selectors/pipelines
- apply shared post-processing and response normalization
"""

from __future__ import annotations
import hashlib
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

# Core modules (always required)
from .__about__ import __version__
from .config import (
    cfg_get,
    ROLES,
    KB_PATHS,
    VAULT_KB_NAME,
    FS_TOP_K,
    FS_MAX_CHARS,
    ROUTER_DEBUG,
    ROUTER_DEBUG_LOG_USER_TEXT,
)
from .session_state import get_state, SessionState, KAIOKEN_CLOSED_THREAD_TTL_TURNS
from .helpers import (
    has_images_in_messages,
    has_image_signal,
    has_mentats_in_recent_history,
    normalize_history,
    last_user_message,
    is_command as _is_command,
    split_selector as _split_selector,
)
from .model_calls import call_model_prompt, call_model_messages
from .streaming import make_openai_response as _make_openai_response, wrap_text_as_sse as _stream_sse
from .commands import handle_command
from .interaction_profile import (
    build_constraints_block as build_profile_constraints_block,
    classify_sensitive_context,
    compute_effective_strength,
    effective_profile,
    has_non_default_style,
    update_profile_from_user_turn,
)
from .privacy_utils import safe_preview, short_hash
# Required modules
from .vodka_filter import Filter as VodkaFilter, purge_session_memory_jsonl, purge_vodka_ctx_facts
from .serious import run_serious

# Optional modules (feature-detected)
try:
    from .fun import run_fun  # type: ignore
except Exception:
    run_fun = None  # type: ignore

try:
    from .mentats import run_mentats  # type: ignore
except Exception:
    run_mentats = None  # type: ignore

try:
    from .fs_rag import build_fs_facts_block, build_locked_summ_facts_block  # type: ignore
except Exception:
    build_fs_facts_block = None  # type: ignore
    build_locked_summ_facts_block = None  # type: ignore

try:
    from .pipelines import run_raw  # type: ignore
except Exception:
    run_raw = None  # type: ignore

try:
    from .rag import build_rag_block as _build_rag_block  # type: ignore
except Exception:
    _build_rag_block = None  # type: ignore

try:
    from .scratchpad_sidecar import (  # type: ignore
        maybe_capture_command_output,
        build_scratchpad_facts_block,
        wants_exhaustive_query,
        build_scratchpad_dump_text,
        list_scratchpad_records,
        capture_scratchpad_output,
        purge_session_kb_jsonl,
    )
except Exception:
    maybe_capture_command_output = None  # type: ignore
    build_scratchpad_facts_block = None  # type: ignore
    wants_exhaustive_query = None  # type: ignore
    build_scratchpad_dump_text = None  # type: ignore
    list_scratchpad_records = None  # type: ignore
    capture_scratchpad_output = None  # type: ignore
    purge_session_kb_jsonl = None  # type: ignore

try:
    from .sources_footer import normalize_sources_footer  # type: ignore
except Exception:
    normalize_sources_footer = None  # type: ignore

try:
    from .sidecars import handle_wiki_query as _sidecar_wiki_query  # type: ignore
except Exception:
    _sidecar_wiki_query = None  # type: ignore

try:
    from .cheatsheets_runtime import (  # type: ignore
        resolve_cheatsheets_turn as _resolve_cheatsheets_turn,
        load_cheatsheets_entries as _load_cheatsheets_entries,
        get_cheatsheets_parse_warnings as _get_cheatsheets_parse_warnings,
    )
except Exception:
    _resolve_cheatsheets_turn = None  # type: ignore
    _load_cheatsheets_entries = None  # type: ignore
    _get_cheatsheets_parse_warnings = None  # type: ignore

try:
    from .style_adapter import rewrite_response_style, score_output_compliance  # type: ignore
except Exception:
    rewrite_response_style = None  # type: ignore
    score_output_compliance = None  # type: ignore

from .recall_semantic import (
    parse_recall_det_payload as _parse_recall_det_payload,
    synthesize_recall_answer as _synthesize_recall_answer,
)
from .correction_bind import maybe_handle_correction_bind as _maybe_handle_correction_bind
from .state_solver_flow import (
    maybe_handle_state_solver_early as _maybe_handle_state_solver_early,
    resolve_state_solver_for_turn as _resolve_state_solver_for_turn,
)
from .chat_preflight import handle_preflight as _handle_preflight
from .vodka_runtime import (
    apply_vodka_runtime as _apply_vodka_runtime,
    preset_for_session as _vr_preset_for_session,
    VODKA_PRESET_MAP as _VODKA_PRESET_MAP,
)
from .chat_modes import handle_mentats_selector as _handle_mentats_selector
from .chat_dispatch import (
    handle_vision_ocr_selector as _handle_vision_ocr_selector,
    resolve_fun_mode as _resolve_fun_mode,
)
from .chat_mode_execution import maybe_handle_fun_fr_raw as _maybe_handle_fun_fr_raw
from .fun_style import (
    run_fun_rewrite_fallback,
    select_fun_style_seed as _select_fun_style_seed,
)
from .chat_facts import (
    build_fs_facts as _cf_build_fs_facts,
    build_vault_facts as _cf_build_vault_facts,
)
from .chat_aliases import soft_alias_command as _chat_soft_alias_command
from .chat_finalize import finalize_chat_response as _chat_finalize_response
from .chat_profile_state import (
    update_blocked_nicknames as _update_blocked_nicknames,
    requires_sensitive_confirm as _requires_sensitive_confirm,
)
from .chat_postprocess import (
    scratchpad_quote_lines as _pp_scratchpad_quote_lines,
    sanitize_scratchpad_grounded_output as _pp_sanitize_scratchpad_grounded_output,
    append_scratchpad_provenance as _pp_append_scratchpad_provenance,
    apply_scratchpad_strict_policy as _pp_apply_scratchpad_strict_policy,
    apply_benchmark_contract_policy as _pp_apply_benchmark_contract_policy,
    rewrite_source_line as _pp_rewrite_source_line,
    apply_image_footer as _pp_apply_image_footer,
    lock_constraints_block as _pp_lock_constraints_block,
    apply_locked_output_policy as _pp_apply_locked_output_policy,
    apply_deterministic_footer as _pp_apply_deterministic_footer,
    append_profile_footer as _pp_append_profile_footer,
)
from .chat_semantic import (
    maybe_apply_consistency_verifier as _maybe_apply_consistency_verifier,
    semantic_pick_clarifier_option as _semantic_pick_clarifier_option,
    semantic_refine_constraint_choice as _semantic_refine_constraint_choice,
)
from .kaioken_literal import (
    _extract_literal_followup_anchor as _kl_extract_literal_followup_anchor,
    _first_clause_for_anchor_parse as _kl_first_clause_for_anchor_parse,
    _is_literal_followup_turn as _kl_is_literal_followup_turn,
    _literal_anchor_is_primary_subject as _kl_literal_anchor_is_primary_subject,
    _resolve_recent_anchor as _kl_resolve_recent_anchor,
)
from .kaioken_guards import (
    _choose_short_fallback as _kg_choose_short_fallback,
    _normalize_for_repeat_check as _kg_normalize_for_repeat_check,
    _recent_repeat_within_window as _kg_recent_repeat_within_window,
    _remember_recent_assistant_body as _kg_remember_recent_assistant_body,
    _repeat_like as _kg_repeat_like,
    _sentence_count_for_guard as _kg_sentence_count_for_guard,
    _split_fun_prefix as _kg_split_fun_prefix,
)
from .kaioken_classify import (
    _is_clarification_prompt as _kc_is_clarification_prompt,
    _is_continuation_prompt as _kc_is_continuation_prompt,
    _is_resolution_signal as _kc_is_resolution_signal,
    _is_topic_switch_prompt as _kc_is_topic_switch_prompt,
    _user_explicitly_requests_advice as _kc_user_explicitly_requests_advice,
    _user_explicitly_requests_encouragement as _kc_user_explicitly_requests_encouragement,
)
from .kaioken_routing import (
    KaiokenRoutingDeps as _KaiokenRoutingDeps,
    _apply_output_guard as _kr_apply_output_guard,
)
from .router_correction_utils import (
    is_correction_intent_query as _is_correction_intent_query,
    evaluate_structural_correction_intent as _evaluate_structural_correction_intent,
    is_explicit_reengage_query as _is_explicit_reengage_query,
    extract_numeric_correction as _extract_numeric_correction,
    unit_canonical as _unit_canonical,
    units_match as _units_match,
    extract_first_num_unit as _extract_first_num_unit,
    to_km as _to_km,
    resolve_old_from_prior_answer as _resolve_old_from_prior_answer,
    strip_got_it_prefix as _strip_got_it_prefix,
    fallback_contextual_correction as _fallback_contextual_correction,
    last_assistant_text as _last_assistant_text,
    last_user_text as _last_user_text,
    last_user_text_before as _last_user_text_before,
    last_non_correction_user_text as _last_non_correction_user_text,
)
from .router_response_utils import (
    strip_in_body_confidence_source_claims as _strip_in_body_confidence_source_claims,
    strip_behavior_announcement_sentences as _strip_behavior_announcement_sentences,
    is_argumentative_prompt as _is_argumentative_prompt,
    is_argumentatively_complete as _is_argumentatively_complete,
    fallback_with_mode_header as _fallback_with_mode_header,
    strip_footer_lines_for_scan as _strip_footer_lines_for_scan,
    is_ack_reframe_only as _is_ack_reframe_only,
    normalize_signature_text as _normalize_signature_text,
    enforce_fun_antiparrot as _enforce_fun_antiparrot,
    strip_irrelevant_proofread_tail as _rr_strip_irrelevant_proofread_tail,
    serious_max_tokens_for_query as _rr_serious_max_tokens_for_query,
    normalize_agreement_ack_tense as _rr_normalize_agreement_ack_tense,
)
try:
    from .kaioken import (  # type: ignore
        append_kaioken_telemetry as _kaioken_append_telemetry,
        classify_register as _kaioken_classify_register,
    )
except Exception:
    _kaioken_append_telemetry = None  # type: ignore
    _kaioken_classify_register = None  # type: ignore
try:
    from .state_reasoning import (  # type: ignore
        classify_constraint_turn,
        classify_query_family,
        solve_state_transition_query,
        solve_state_transition_followup,
        is_followup_consistency_query,
        solve_constraint_followup,
    )
except Exception:
    classify_constraint_turn = None  # type: ignore
    classify_query_family = None  # type: ignore
    solve_state_transition_query = None  # type: ignore
    solve_state_transition_followup = None  # type: ignore
    is_followup_consistency_query = None  # type: ignore
    solve_constraint_followup = None  # type: ignore


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

async def _run_sync(func, /, *args, **kwargs):
    """Run blocking/sync work off the event loop to keep router responsive."""
    return await run_in_threadpool(func, *args, **kwargs)


def _dbg(msg: str) -> None:
    if ROUTER_DEBUG:
        print(msg, flush=True)


def _debug_user_fragment(text: str) -> str:
    raw = str(text or "")
    if ROUTER_DEBUG_LOG_USER_TEXT:
        return safe_preview(raw, max_len=240)
    return f"<redacted len={len(raw)} sha={short_hash(raw)}>"


_CHEATSHEETS_DIR = Path(__file__).resolve().parent / "cheatsheets"
try:
    if _load_cheatsheets_entries is not None:
        _load_cheatsheets_entries(_CHEATSHEETS_DIR)
except Exception:
    pass


def _emit_kaioken_telemetry(
    *,
    state: SessionState,
    session_id: str,
    user_text: str,
    route_class: str,
    outcome: Optional[Dict[str, Any]] = None,
) -> None:
    """Fail-open KAIOKEN telemetry emitter (phase 0 sensor-only)."""
    if _kaioken_append_telemetry is None:
        return
    try:
        enabled = bool(getattr(state, "kaioken_enabled", bool(cfg_get("kaioken.enabled", True))))
        mode = str(getattr(state, "kaioken_mode", str(cfg_get("kaioken.mode", "log_only") or "log_only")) or "log_only").strip().lower()
        state.kaioken_turn_counter = int(getattr(state, "kaioken_turn_counter", 0) or 0) + 1
        _kaioken_append_telemetry(
            session_id=session_id,
            turn_index=state.kaioken_turn_counter,
            user_text=str(user_text or ""),
            route_class=str(route_class or "control"),
            enabled=enabled,
            mode=mode,
            log_all_routes=bool(cfg_get("kaioken.log_all_routes", True)),
            session_hash_salt=str(cfg_get("kaioken.session_hash_salt", "kaioken-v1") or "kaioken-v1"),
            extra=dict(outcome or {}),
        )
    except Exception:
        # Never alter runtime behavior because of telemetry failure.
        pass


def _build_kaioken_soft_constraints(*, state: SessionState, user_text: str) -> str:
    """Phase-1 soft guidance for model-chat lane only (fail-open, high-confidence only)."""
    if _kaioken_classify_register is None:
        return ""
    if not bool(getattr(state, "kaioken_enabled", False)):
        return ""
    mode = str(getattr(state, "kaioken_mode", "log_only") or "log_only").strip().lower()
    if mode not in {"coerce", "phase1"}:
        return ""
    try:
        cls = _kaioken_classify_register(str(user_text or ""))
    except Exception:
        return ""
    switch_anchor_tokens: list[str] = []
    switch_prior_tokens: list[str] = []
    force_switch_hint = False
    try:
        threads = _get_kaioken_threads(state)
        open_rows: list[tuple[int, Dict[str, Any]]] = []
        for _tid, meta in threads.items():
            if not isinstance(meta, dict):
                continue
            if bool(meta.get("closed", False)):
                continue
            open_rows.append((int(meta.get("last_turn", -9999) or -9999), meta))
        has_open_thread = bool(open_rows)
        turn_counter = int(getattr(state, "kaioken_turn_counter", 0) or 0)
        last_switch_turn = int(getattr(state, "kaioken_last_topic_switch_turn", 0) or 0)
        is_switch_turn = _is_topic_switch_prompt(user_text)
        is_post_switch_turn = bool(
            (last_switch_turn > 0)
            and ((turn_counter - last_switch_turn) == 1)
            and (not _is_continuation_prompt(user_text))
            and (not _is_clarification_prompt(user_text))
            and (not _is_resolution_signal(user_text))
        )
        force_switch_hint = bool(has_open_thread and (is_switch_turn or is_post_switch_turn))
        if force_switch_hint:
            switch_anchor_tokens = _collect_topic_candidates(str(user_text or ""), closed_topics=set())[:4]
            open_rows.sort(key=lambda x: x[0], reverse=True)
            switch_prior_tokens = sorted(_thread_tokens(open_rows[0][1]))[:4] if open_rows else []
    except Exception:
        force_switch_hint = False
        switch_anchor_tokens = []
        switch_prior_tokens = []
    conf = str(getattr(cls, "confidence", "low") or "low").lower()
    subs = {str(s).strip().lower() for s in list(getattr(cls, "subsignals", []) or [])}
    macro = str(getattr(cls, "macro", "working") or "working").strip().lower()
    if conf != "high":
        if _is_continuation_prompt(user_text):
            prev = str(getattr(state, "last_user_text", "") or "").strip()
            if prev:
                try:
                    pcls = _kaioken_classify_register(prev)
                    psubs = {str(s).strip().lower() for s in list(getattr(pcls, "subsignals", []) or [])}
                    pmacro = str(getattr(pcls, "macro", "working") or "working").strip().lower()
                    if {"distress_hint", "vulnerable_under_humour"} & psubs:
                        subs = psubs
                        macro = pmacro
                        conf = "high"
                    elif _KAIOKEN_DISTRESS_FALLBACK_RE.search(prev):
                        subs = {"distress_hint"}
                        macro = "personal"
                        conf = "high"
                except Exception:
                    pass
        if conf != "high" and (not force_switch_hint):
            return ""
    literal_anchor = _extract_literal_followup_anchor(str(user_text or ""), state=state)
    literal_followup = bool(literal_anchor)
    hints: List[str] = []

    if "directive" in subs:
        hints.append("Follow user constraints/format exactly before adding extra detail.")
    if "friction" in subs:
        hints.append("User is pushing back; match energy without escalating or dismissing. Do not lecture.")

    if "vulnerable_under_humour" in subs:
        hints.extend([
            "User is mixing humor with pain/stress.",
            "Engage with the joke first, then acknowledge what is underneath.",
            "Do NOT explain or narrate the joke.",
            "Do NOT switch into generic motivational language.",
            "Do NOT give unsolicited advice unless user explicitly asks for it.",
            "Keep it short (2-5 sentences), grounded, and human.",
        ])
    elif "distress_hint" in subs:
        hints.extend([
            "User is signaling real self-doubt/distress.",
            "Acknowledge directly and briefly.",
            "Do NOT use platitudes, pep-talk slogans, or generic reassurance.",
            "Do NOT give unsolicited advice unless user explicitly asks for it.",
            "Keep response short (2-5 sentences).",
        ])
    elif macro == "working":
        hints.append("Keep response concise, task-first, and explicit about concrete next steps.")
    elif macro == "personal":
        hints.append("Use calm, supportive tone; acknowledge briefly, then prioritize practical help.")
    elif macro == "casual":
        hints.append("Keep tone light but maintain factual accuracy and directness.")
    if literal_followup:
        hints.extend([
            f"This is a literal follow-up about `{literal_anchor}`.",
            f"Answer `{literal_anchor}` directly and concretely.",
            "Do NOT pivot into metaphor, identity analysis, or broad reframing.",
        ])
    closed_topics = _closed_topics_not_mentioned(state, user_text)
    if closed_topics:
        hints.append(
            "Do not resurface resolved topics unless the user explicitly asks to reopen them: "
            + ", ".join(sorted(closed_topics)[:4])
        )
    if force_switch_hint:
        hints.append("Topic switch detected while prior thread remains open; prioritize the new topic now.")
        if switch_anchor_tokens:
            hints.append("Anchor this response to current topic tokens: " + ", ".join(switch_anchor_tokens))
        if switch_prior_tokens:
            hints.append("Do not continue prior thread tokens unless user explicitly reopens them: " + ", ".join(switch_prior_tokens))

    if not hints:
        return ""
    lines = "\n".join(f"- {h}" for h in hints)
    return "KAIOKEN guidance (soft):\n" + lines


_KAIOKEN_PEP_TALK_RE = re.compile(
    r"\b("
    r"you(?:'|’)re not|you are not|you got this|keep going|keep pushing|keep debugging|push through|start small|"
    r"focus on|many developers|age (?:isn(?:'|’)t|is not)|"
    r"progress over perfection|you(?:'|’)re human|you are human|"
    r"take breaks|ergonomic|posture|chair|monitor|"
    r"stop comparing|temporary setbacks?|just a setback"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_METAPHOR_PIVOT_RE = re.compile(
    r"\b("
    r"metaphor|symbolic|represents|reflects|the real pain is|"
    r"about feeling outdated|about your doubt|comparison to others|comparing yourself"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_ADVICE_RE = re.compile(
    r"\b("
    r"you should|try\b|start\b|start with|start by|focus on|consider|"
    r"take breaks?|rest\b|use an ergonomic|posture|chair|monitor|"
    r"do this|next step|keep coding|ship what you have|let(?:'|’)s\b|lets\b"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_REASSURE_RE = re.compile(
    r"\b("
    r"you(?:'|’)re not|you are not|it(?:'|’)s okay|it is okay|"
    r"you(?:'|’)ll be fine|you will be fine|you(?:'|’)ve got this|you got this|"
    r"not a flaw in you|not a sign (?:you(?:'|’)re|you are) failing|not failing"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_HELP_OFFER_RE = re.compile(
    r"\b("
    r"if you want, i can|how can i help|i can help|"
    r"i can walk through|i can talk through|i can mirror|for example[: ]"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_ANNOUNCE_RE = re.compile(
    r"\b("
    r"i'?ll keep this direct and non-preachy|"
    r"keeping this direct and grounded|"
    r"keeping this direct|"
    r"keeping it direct|"
    r"keep this direct|"
    r"keep it direct|"
    r"staying direct|"
    r"stay direct|"
    r"i'?ll keep responses tighter(?:\s+from now on)?|"
    r"i'?m already operating in direct mode|"
    r"i'?ll give (?:you )?straight talk without fluff|"
    r"understood\.?\s*no more meta|"
    r"give me the exact task and desired output format"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_DISTRESS_FALLBACK_RE = re.compile(
    r"\b("
    r"fraud|too old|too broken|mass-irrelevant|can't do this|cannot do this|"
    r"i don't know why i bother|falling behind|overloaded|venting"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_LITERAL_LINES = (
    "Your {anchor} matters here. I won't dress it up or pivot away from it.",
    "That's real. I'm not going to reframe your {anchor}.",
    "Yeah. That's the actual thing - your {anchor}.",
    "That one is literal. Your {anchor} is the thing.",
    "Yeah. Your {anchor}'s the thing that doesn't negotiate.",
)
_KAIOKEN_SHORT_FALLBACKS = (
    "I hear you.",
    "I hear you. That's not nothing.",
    "That sits heavy.",
    "Yeah, that's the real thing.",
    "That's a lot.",
    "Go on.",
)
_KAIOKEN_POSITIVE_FRAMING_RE = re.compile(
    r"\b(great move|good choice|smart decision|wise decision|excellent choice|solid move)\b",
    re.IGNORECASE,
)
_KAIOKEN_ACK_RE = re.compile(
    r"\b(i hear you|i get it|that sits heavy|fair|valid|makes sense|i know)\b",
    re.IGNORECASE,
)
_KAIOKEN_BRIDGE_RE = re.compile(
    r"\b(but|and yet|still|though|even so|yet)\b",
    re.IGNORECASE,
)
_KAIOKEN_END_REFRAME_RE = re.compile(
    r"\b("
    r"not cheating|just using|still doing|doing the thinking|"
    r"just faster|progress|you can still|you will still"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_NIHILISM_RE = re.compile(
    r"\b("
    r"no point|pointless|nothing matters|i don't know what the point is|"
    r"no reason to|why bother|doesn't matter anyway|"
    r"not (?:the )?right fit(?: right now)?|"
    r"maybe (?:this|it|coding) (?:isn't|is not) for you|"
    r"(?:might|may) not be for you|"
    r"(?:right|best) (?:fit|path) for you|"
    r"(?:if|when)\s+(?:\w+\s+){0,4}(?:quit|quitting|stop(?:ping)?|give up|walk away)(?:\s+\w+){0,10}\s+(?:is|feels|seems|sounds)?\s*(?:okay|right|valid|honest)|"
    r"better off (?:quitting|stopping|walking away)|"
    r"(?:you should|should you) (?:quit|stop)|"
    r"give up|walk away|not worth (?:it|continuing)"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_FRICTION_RE = re.compile(
    r"\b("
    r"rude|asshole|tone deaf|you misread|you misunderstood|wtf|"
    r"not until you|apologize|apologise|what got up your butt|fuck off"
    r")\b",
    re.IGNORECASE,
)


def _is_caps_burst_friction(text: str) -> bool:
    s = str(text or "")
    toks = re.findall(r"[A-Za-z]{2,}", s)
    if not toks:
        return False
    all_caps = sum(1 for t in toks if t.isupper())
    return bool(len(toks) <= 10 and all_caps >= 2)
_KAIOKEN_JOKE_EXPLAIN_RE = re.compile(
    r"\b("
    r"you(?:'|’)re comparing|underneath that joke|using humor to mask|"
    r"the punchline|you(?:'|’)re using humor|joke is that|what you really mean"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_DESCRIPTOR_TERM_RE = re.compile(
    r"\b("
    r"unstable|instability|broken|fraud|worthless|damaged|weak"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_OWNERSHIP_DESCRIPTOR_RE = re.compile(
    r"\b("
    r"(?:you(?:(?:'|’)re| are)\s+(?:\w+\s+){0,3}?(?:unstable|broken|worthless|damaged|weak|fraud(?:ulent)?))|"
    r"(?:your\s+(?:\w+\s+){0,3}?(?:instability|brokenness|fraud|worthlessness|damage|weakness))"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_DIAGNOSTIC_RE = re.compile(
    r"\b("
    r"the real issue|the issue is|what this means is|root cause|core issue|underlying issue|"
    r"you(?:'|’)re still treating|you are still treating|you need to|the problem is|"
    r"the real problem|diagnos(?:e|is|ing)|unresolved issues"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_LITERAL_FOLLOWUP_RE = re.compile(
    r"^(?:and\s+)?(?:what about|how about|and)?\s*(?:my|the)\s+([a-z0-9][a-z0-9\-']{2,})\??$",
    re.IGNORECASE,
)


def _has_kaioken_descriptor_drift(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    if not _KAIOKEN_OWNERSHIP_DESCRIPTOR_RE.search(t):
        return False
    return bool(_KAIOKEN_DESCRIPTOR_TERM_RE.search(t))


def _is_kaioken_guard_candidate(*, state: SessionState, user_text: str) -> bool:
    if _kaioken_classify_register is None:
        return False
    if not bool(getattr(state, "kaioken_enabled", False)):
        return False
    mode = str(getattr(state, "kaioken_mode", "log_only") or "log_only").strip().lower()
    if mode not in {"coerce", "phase1"}:
        return False
    try:
        cur_text = str(user_text or "")
        # Domain carry: keep guard candidacy active when a clinical/anatomical
        # topic is already in session state and referenced again.
        clinical_terms = {
            "spine", "back", "disc", "herniation", "l5", "s1", "c5", "knee", "shoulder", "neck",
        }
        recent_nouns = {str(x or "").strip().lower() for x in list(getattr(state, "kaioken_recent_user_nouns", []) or [])}
        distress_topics = {str(x or "").strip().lower() for x in set(getattr(state, "kaioken_distress_topics", set()) or set())}
        clinical_state = {t for t in (recent_nouns | distress_topics) if t in clinical_terms}
        cur_terms = {str(x or "").strip().lower() for x in _extract_concrete_nouns(cur_text)}
        if clinical_state and any(t for t in cur_terms if t in clinical_state):
            return True
        if _user_explicitly_requests_advice(cur_text):
            # Advice asks should stay in KAIOKEN guard lane when a distress
            # context is already active, even if this turn itself is short.
            if bool(set(getattr(state, "kaioken_distress_topics", set()) or set())):
                return True
            prev = str(getattr(state, "last_user_text", "") or "").strip()
            if prev:
                if _KAIOKEN_DISTRESS_FALLBACK_RE.search(prev):
                    return True
                prev_cls = _kaioken_classify_register(prev)
                prev_subs = {str(s).strip().lower() for s in list(getattr(prev_cls, "subsignals", []) or [])}
                if {"distress_hint", "vulnerable_under_humour"} & prev_subs:
                    return True
        if _KAIOKEN_DISTRESS_FALLBACK_RE.search(cur_text):
            return True
        cls = _kaioken_classify_register(cur_text)
        conf = str(getattr(cls, "confidence", "low") or "low").lower()
        subs = {str(s).strip().lower() for s in list(getattr(cls, "subsignals", []) or [])}
        if conf == "high" and ({"distress_hint", "vulnerable_under_humour"} & subs):
            return True
        if _is_continuation_prompt(cur_text):
            prev = str(getattr(state, "last_user_text", "") or "").strip()
            if prev:
                prev_cls = _kaioken_classify_register(prev)
                prev_subs = {str(s).strip().lower() for s in list(getattr(prev_cls, "subsignals", []) or [])}
                if {"distress_hint", "vulnerable_under_humour"} & prev_subs:
                    return True
                if _KAIOKEN_DISTRESS_FALLBACK_RE.search(prev):
                    return True
        # Conservative follow-up inheritance for ultra-short turns
        # (e.g., "And my spine?") immediately after a high-confidence
        # distress/VUH turn.
        if len(re.findall(r"[A-Za-z0-9_']+", cur_text)) <= 12:
            prev = str(getattr(state, "last_user_text", "") or "")
            if prev.strip():
                prev_cls = _kaioken_classify_register(prev)
                prev_conf = str(getattr(prev_cls, "confidence", "low") or "low").lower()
                prev_subs = {str(s).strip().lower() for s in list(getattr(prev_cls, "subsignals", []) or [])}
                if prev_conf == "high" and ({"distress_hint", "vulnerable_under_humour"} & prev_subs):
                    return True
        return False
    except Exception:
        return False


def _split_fun_prefix(text: str) -> tuple[str, str]:
    return _kg_split_fun_prefix(text)


def _normalize_for_repeat_check(text: str) -> str:
    return _kg_normalize_for_repeat_check(text)


def _sentence_count_for_guard(text: str) -> int:
    return _kg_sentence_count_for_guard(text)


def _repeat_like(a: str, b: str) -> bool:
    return _kg_repeat_like(a, b)


def _recent_repeat_within_window(state: SessionState, text: str, *, window: int = 4) -> bool:
    return _kg_recent_repeat_within_window(state, text, window=window)


def _remember_recent_assistant_body(state: SessionState, text: str, *, max_items: int = 8) -> None:
    _kg_remember_recent_assistant_body(state, text, max_items=max_items)


def _choose_short_fallback(state: SessionState, *, prior_text: str = "") -> str:
    return _kg_choose_short_fallback(state, _KAIOKEN_SHORT_FALLBACKS, prior_text=prior_text)


_LITERAL_ANCHOR_STOPWORDS = {
    "this", "that", "it", "thing", "stuff", "one", "mine", "myself", "yourself",
    "my", "the", "what", "about", "and", "umm", "um", "uh", "hmm", "tips", "advice",
}


_ANCHOR_QUERY_STOPWORDS = _LITERAL_ANCHOR_STOPWORDS | {
    "any", "got", "have", "has", "had", "give", "tell", "show", "with", "for", "from",
    "into", "onto", "your", "you", "me", "i", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "can", "could", "should", "would", "will", "to", "of", "in", "on",
    "hot", "takes", "take", "thoughts", "thought", "pro", "advice", "tips", "tip", "help",
}

_KAIOKEN_LOW_SALIENCE_TOPIC_RE = re.compile(
    r"\b(home|morning|train|today|thing|stuff|problem|issue)\b",
    re.IGNORECASE,
)
_KAIOKEN_WORK_CONTEXT_RE = re.compile(
    r"\b("
    r"patient|patients|appointment|appointments|clinic|chiro|practice|"
    r"no[\s-]?show|cancell?(?:ation|ed|ing|s)|booking|bookings|schedule|"
    r"waitlist|caseload|client|clients"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_CONTEXT_DEFLECTION_RE = re.compile(
    r"\b("
    r"can(?:'|â€™)?t be certain what|cannot be certain what|"
    r"without more detail|if you clarify|need more detail|"
    r"unclear what .* means"
    r")\b",
    re.IGNORECASE,
)
_KAIOKEN_CLARIFY_OPTION_STOPWORDS = {
    "yeah", "yep", "yes", "no", "nah", "well", "who", "what", "where", "when", "why", "how",
    "was", "were", "is", "are", "am", "did", "does", "do", "just", "really", "okay", "ok",
    "hmm", "umm", "uh", "sorry", "tell", "said", "say", "mean", "meant", "you", "your", "my",
    "mine", "ours", "their", "this", "that", "these", "those",
}
_KAIOKEN_HUMOR_CUE_RE = re.compile(
    r"\b(haha|lol|lmao|joke|vibes|prayer)\b|(?:\bas\b.+\bas\b)",
    re.IGNORECASE,
)


def _is_continuation_prompt(text: str) -> bool:
    return _kc_is_continuation_prompt(text)


def _is_clarification_prompt(text: str) -> bool:
    return _kc_is_clarification_prompt(text)


def _is_topic_switch_prompt(text: str) -> bool:
    return _kc_is_topic_switch_prompt(text)


def _is_resolution_signal(text: str) -> bool:
    return _kc_is_resolution_signal(text)


def _extract_concrete_nouns(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", str(text or "").lower())
    out: list[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in _LITERAL_ANCHOR_STOPWORDS:
            continue
        if t.isdigit():
            continue
        out.append(t)
    return out


def _normalize_topic_token(tok: str) -> str:
    t = str(tok or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
    t = re.sub(r"'s$", "", t)
    if not re.match(r"^[a-z0-9][a-z0-9'-]{2,}$", t):
        return ""
    # Conservative singularization only; avoid fragment stems like "giving" -> "giv".
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def _extract_primary_topic(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    m = re.search(r"\b(?:my|the)\s+([a-z0-9][a-z0-9\-']{1,})\b", s, flags=re.IGNORECASE)
    if m:
        cand = _normalize_topic_token(str(m.group(1) or ""))
        if cand and cand not in _LITERAL_ANCHOR_STOPWORDS and not _KAIOKEN_LOW_SALIENCE_TOPIC_RE.search(cand):
            return cand
    nouns = [_normalize_topic_token(x) for x in _extract_concrete_nouns(s)]
    nouns = [x for x in nouns if x and x not in _LITERAL_ANCHOR_STOPWORDS and not _KAIOKEN_LOW_SALIENCE_TOPIC_RE.search(x)]
    return nouns[0] if nouns else ""


def _collect_topic_candidates(text: str, *, closed_topics: set[str] | None = None) -> list[str]:
    closed = set(closed_topics or set())
    cand: list[str] = []
    primary = _extract_primary_topic(text)
    if primary:
        cand.append(primary)
    cand.extend([_normalize_topic_token(x) for x in _extract_concrete_nouns(text)])
    out: list[str] = []
    for x in cand:
        if not x:
            continue
        if x in _LITERAL_ANCHOR_STOPWORDS:
            continue
        if x in closed:
            continue
        if _KAIOKEN_LOW_SALIENCE_TOPIC_RE.search(x):
            continue
        if x in out:
            continue
        out.append(x)
    return out


_KAIOKEN_THREAD_NO_OVERLAP_SAME_MACRO_LIMIT = 2


def _classify_macro_for_thread(text: str) -> str:
    try:
        if _kaioken_classify_register is None:
            return ""
        cls = _kaioken_classify_register(str(text or ""))
        return str(getattr(cls, "macro", "") or "").strip().lower()
    except Exception:
        return ""


def _thread_tokens(meta: Dict[str, Any]) -> set[str]:
    raw = meta.get("tokens", set())
    out: set[str] = set()
    if isinstance(raw, set):
        iterable = raw
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []
    for tok in iterable:
        norm = _normalize_topic_token(str(tok or ""))
        if norm:
            out.add(norm)
    return out


def _get_kaioken_threads(state: SessionState) -> Dict[str, Dict[str, Any]]:
    raw = getattr(state, "kaioken_threads", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    clean: Dict[str, Dict[str, Any]] = {}
    for tid, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        clean[str(tid)] = dict(meta)
    setattr(state, "kaioken_threads", clean)
    return clean


def _next_kaioken_thread_id(state: SessionState) -> str:
    seq = int(getattr(state, "kaioken_thread_seq", 0) or 0) + 1
    setattr(state, "kaioken_thread_seq", seq)
    return f"t{seq}"


def _new_kaioken_thread(
    state: SessionState,
    *,
    tokens: list[str],
    macro: str,
    turn_counter: int,
) -> str:
    threads = _get_kaioken_threads(state)
    tid = _next_kaioken_thread_id(state)
    toks = set(tokens)
    threads[tid] = {
        "tokens": toks,
        "closed": False,
        "closed_at_turn": -1,
        "macro": str(macro or ""),
        "last_turn": int(turn_counter),
        "head": str(tokens[0] if tokens else ""),
        "no_overlap_same_macro_streak": 0,
    }
    setattr(state, "kaioken_threads", threads)
    return tid


def _expire_closed_threads(state: SessionState, turn_counter: int) -> None:
    threads = _get_kaioken_threads(state)
    remove_ids: list[str] = []
    for tid, meta in list(threads.items()):
        if not bool(meta.get("closed", False)):
            continue
        closed_at = int(meta.get("closed_at_turn", -9999) or -9999)
        if closed_at < 0:
            continue
        if (int(turn_counter) - closed_at) > int(KAIOKEN_CLOSED_THREAD_TTL_TURNS):
            remove_ids.append(tid)
    for tid in remove_ids:
        threads.pop(tid, None)
    active_tid = str(getattr(state, "kaioken_active_thread_id", "") or "").strip()
    if active_tid and active_tid not in threads:
        setattr(state, "kaioken_active_thread_id", "")
    setattr(state, "kaioken_threads", threads)


def _close_kaioken_thread_by_id(state: SessionState, thread_id: str, *, turn_counter: int) -> None:
    tid = str(thread_id or "").strip()
    if not tid:
        return
    threads = _get_kaioken_threads(state)
    meta = threads.get(tid)
    if not isinstance(meta, dict):
        return
    meta["closed"] = True
    meta["closed_at_turn"] = int(turn_counter)
    meta["last_turn"] = int(turn_counter)
    meta["no_overlap_same_macro_streak"] = 0
    threads[tid] = meta
    setattr(state, "kaioken_threads", threads)


def _close_most_recent_open_kaioken_thread(state: SessionState, *, turn_counter: int) -> None:
    threads = _get_kaioken_threads(state)
    open_rows: list[tuple[int, str]] = []
    for tid, meta in threads.items():
        if bool(meta.get("closed", False)):
            continue
        open_rows.append((int(meta.get("last_turn", -9999) or -9999), tid))
    if not open_rows:
        return
    open_rows.sort(key=lambda x: x[0])
    _close_kaioken_thread_by_id(state, open_rows[-1][1], turn_counter=turn_counter)


def _sync_legacy_topics_from_threads(state: SessionState) -> None:
    threads = _get_kaioken_threads(state)
    active_tid = str(getattr(state, "kaioken_active_thread_id", "") or "").strip()
    open_topics: list[str] = []
    closed_topics: set[str] = set()
    active_topic = ""
    for _last_turn, tid, meta in sorted(
        ((int(m.get("last_turn", -9999) or -9999), t, m) for t, m in threads.items()),
        key=lambda x: x[0],
    ):
        toks = sorted(_thread_tokens(meta))
        if bool(meta.get("closed", False)):
            closed_topics.update(toks)
            continue
        open_topics.extend(toks)
        if tid == active_tid:
            head = _normalize_topic_token(str(meta.get("head", "") or ""))
            if head:
                active_topic = head
            elif toks:
                active_topic = toks[0]
    dedup: list[str] = []
    for tok in open_topics:
        if tok in dedup:
            continue
        dedup.append(tok)
    setattr(state, "kaioken_open_topics", dedup[-8:])
    setattr(state, "kaioken_closed_topics", set(closed_topics))
    setattr(state, "kaioken_active_topic", active_topic)


def _is_post_resolution_casual_pushback(state: SessionState, user_text: str) -> bool:
    t = str(user_text or "").strip()
    if not t:
        return False
    if _is_topic_switch_prompt(t) or _is_resolution_signal(t):
        return False
    if _user_explicitly_requests_advice(t):
        return False
    cur_turn = int(getattr(state, "kaioken_turn_counter", 0) or 0)
    last_res = int(getattr(state, "kaioken_last_resolution_turn", -9999) or -9999)
    if (cur_turn - last_res) > 2:
        return False
    mentioned = {_normalize_topic_token(x) for x in _extract_concrete_nouns(t)}
    closed_topics = set(getattr(state, "kaioken_closed_topics", set()) or set())
    if any(x for x in mentioned if x and x in closed_topics):
        return False
    tok_count = len(re.findall(r"[A-Za-z0-9_']+", t))
    has_social_friction = bool(_KAIOKEN_FRICTION_RE.search(t) or re.search(r"[!?]", t))
    return bool(tok_count <= 20 and has_social_friction)


def _update_kaioken_topic_state_preturn(state: SessionState, user_text: str) -> None:
    t = str(user_text or "").strip()
    if not t:
        return
    _remember_recent_user_turn(state, t)
    try:
        turn_counter = int(getattr(state, "kaioken_turn_counter", 0) or 0)
        _expire_closed_threads(state, turn_counter)
        threads = _get_kaioken_threads(state)
        active_tid = str(getattr(state, "kaioken_active_thread_id", "") or "").strip()
        if active_tid and (
            active_tid not in threads or bool((threads.get(active_tid) or {}).get("closed", False))
        ):
            active_tid = ""
            setattr(state, "kaioken_active_thread_id", "")

        if _is_topic_switch_prompt(t):
            active_tid = ""
            setattr(state, "kaioken_active_thread_id", "")
            setattr(state, "kaioken_last_topic_switch_turn", turn_counter)

        if _is_resolution_signal(t):
            if active_tid:
                _close_kaioken_thread_by_id(state, active_tid, turn_counter=turn_counter)
            else:
                _close_most_recent_open_kaioken_thread(state, turn_counter=turn_counter)
            active_tid = ""
            setattr(state, "kaioken_active_thread_id", "")
            setattr(state, "kaioken_last_resolution_turn", turn_counter)

        if (
            (not _is_clarification_prompt(t))
            and (not _is_continuation_prompt(t))
            and (not _is_resolution_signal(t))
            and (not _is_topic_switch_prompt(t))
            and (not _is_post_resolution_casual_pushback(state, t))
        ):
            turn_tokens = _collect_topic_candidates(t, closed_topics=set())
            turn_macro = _classify_macro_for_thread(t)
            if turn_tokens:
                threads = _get_kaioken_threads(state)
                if not active_tid:
                    active_tid = _new_kaioken_thread(
                        state,
                        tokens=turn_tokens,
                        macro=turn_macro,
                        turn_counter=turn_counter,
                    )
                else:
                    meta = dict(threads.get(active_tid) or {})
                    if (not meta) or bool(meta.get("closed", False)):
                        active_tid = _new_kaioken_thread(
                            state,
                            tokens=turn_tokens,
                            macro=turn_macro,
                            turn_counter=turn_counter,
                        )
                    else:
                        active_tokens = _thread_tokens(meta)
                        cur_tokens = set(turn_tokens)
                        overlap = bool(active_tokens.intersection(cur_tokens))
                        active_macro = str(meta.get("macro", "") or "").strip().lower()
                        if overlap:
                            merged = set(active_tokens)
                            merged.update(cur_tokens)
                            meta["tokens"] = merged
                            if turn_macro:
                                meta["macro"] = turn_macro
                            meta["last_turn"] = turn_counter
                            meta["head"] = str(turn_tokens[0] if turn_tokens else meta.get("head", ""))
                            meta["no_overlap_same_macro_streak"] = 0
                            threads[active_tid] = meta
                            setattr(state, "kaioken_threads", threads)
                        else:
                            same_macro = bool(turn_macro and active_macro and turn_macro == active_macro)
                            streak = int(meta.get("no_overlap_same_macro_streak", 0) or 0)
                            if same_macro:
                                streak += 1
                                if streak > int(_KAIOKEN_THREAD_NO_OVERLAP_SAME_MACRO_LIMIT):
                                    active_tid = _new_kaioken_thread(
                                        state,
                                        tokens=turn_tokens,
                                        macro=turn_macro,
                                        turn_counter=turn_counter,
                                    )
                                else:
                                    merged = set(active_tokens)
                                    merged.update(cur_tokens)
                                    meta["tokens"] = merged
                                    meta["macro"] = active_macro or turn_macro
                                    meta["last_turn"] = turn_counter
                                    meta["head"] = str(turn_tokens[0] if turn_tokens else meta.get("head", ""))
                                    meta["no_overlap_same_macro_streak"] = streak
                                    threads[active_tid] = meta
                                    setattr(state, "kaioken_threads", threads)
                            else:
                                active_tid = _new_kaioken_thread(
                                    state,
                                    tokens=turn_tokens,
                                    macro=turn_macro,
                                    turn_counter=turn_counter,
                                )

        setattr(state, "kaioken_active_thread_id", active_tid)
        _sync_legacy_topics_from_threads(state)
    except Exception:
        pass


def _open_topics_for_clarify(state: SessionState, user_text: str) -> list[str]:
    try:
        open_topics = list(getattr(state, "kaioken_open_topics", []) or [])
        closed_topics = set(getattr(state, "kaioken_closed_topics", set()) or set())
        recent_user = list(getattr(state, "kaioken_recent_user_turns", []) or [])
        user_terms = {_normalize_topic_token(x) for x in _extract_concrete_nouns(user_text)}
        out: list[str] = []
        for t in reversed(open_topics):
            if t in closed_topics:
                continue
            if str(t or "").strip().lower() in _KAIOKEN_CLARIFY_OPTION_STOPWORDS:
                continue
            if not re.match(r"^[a-z0-9][a-z0-9'-]{2,}$", str(t or "").lower()):
                continue
            if _KAIOKEN_LOW_SALIENCE_TOPIC_RE.search(str(t or "").lower()):
                continue
            # Clarify options must be grounded in complete tokens actually seen in recent user turns.
            if not any(re.search(rf"\b{re.escape(str(t).lower())}\b", str(rt or "").lower()) for rt in recent_user):
                continue
            if t in user_terms:
                continue
            if t in out:
                continue
            out.append(t)
            if len(out) >= 2:
                break
        return out
    except Exception:
        return []


def _closed_topics_not_mentioned(state: SessionState, user_text: str) -> list[str]:
    try:
        turn_counter = int(getattr(state, "kaioken_turn_counter", 0) or 0)
        _expire_closed_threads(state, turn_counter)
        threads = _get_kaioken_threads(state)
        user_terms = {_normalize_topic_token(x) for x in _extract_concrete_nouns(user_text)}
        blocked: set[str] = set()
        for _tid, meta in threads.items():
            if not bool(meta.get("closed", False)):
                continue
            tokens = _thread_tokens(meta)
            if not tokens:
                continue
            # User mention acts as explicit reopen for that closed thread.
            if user_terms.intersection(tokens):
                continue
            blocked.update(tokens)
        if blocked:
            return sorted(t for t in blocked if t)
        closed_topics = set(getattr(state, "kaioken_closed_topics", set()) or set())
        if not closed_topics:
            return []
        return [t for t in closed_topics if t and t not in user_terms]
    except Exception:
        return []


def _mentions_any_topic(text: str, topics: list[str]) -> bool:
    s = str(text or "").lower()
    if not s:
        return False
    for t in topics:
        tt = str(t or "").strip().lower()
        if not tt:
            continue
        if re.search(rf"\b{re.escape(tt)}\b", s):
            return True
    return False


def _remember_distress_topics_from_turn(
    state: SessionState,
    *,
    user_text: str,
    macro: str,
    confidence: str,
    subs: set[str],
) -> None:
    try:
        if str(macro or "").strip().lower() != "personal":
            return
        conf = str(confidence or "").strip().lower()
        psubs = {str(x).strip().lower() for x in set(subs or set())}
        if not (
            ("distress_hint" in psubs)
            or ("vulnerable_under_humour" in psubs)
            or (conf == "high" and bool(psubs))
            or _KAIOKEN_DISTRESS_FALLBACK_RE.search(str(user_text or ""))
        ):
            return
        known = set(getattr(state, "kaioken_distress_topics", set()) or set())
        for tok in _collect_topic_candidates(str(user_text or ""), closed_topics=set()):
            known.add(tok)
        setattr(state, "kaioken_distress_topics", known)
    except Exception:
        pass


def _is_structural_vuh_turn(state: SessionState, user_text: str) -> bool:
    s = str(user_text or "").strip().lower()
    if not s:
        return False
    if not _KAIOKEN_HUMOR_CUE_RE.search(s):
        return False
    known = set(getattr(state, "kaioken_distress_topics", set()) or set())
    mentioned = {_normalize_topic_token(x) for x in _extract_concrete_nouns(s)}
    topic_overlap = bool(known and mentioned and any(t in known for t in mentioned if t))
    return bool(topic_overlap or _KAIOKEN_DISTRESS_FALLBACK_RE.search(s))


def _advice_about_disclosed_distress_topic(state: SessionState, user_text: str, literal_anchor: str = "") -> tuple[bool, str]:
    text = str(user_text or "")
    if not _user_explicitly_requests_advice(text):
        return False, ""
    known = set(getattr(state, "kaioken_distress_topics", set()) or set())
    if not known:
        return False, ""
    anchor = str(literal_anchor or "").strip().lower()
    if anchor and anchor in known:
        return True, anchor
    candidates = _collect_topic_candidates(text, closed_topics=set())
    for c in candidates:
        if c in known:
            return True, c
    active = str(getattr(state, "kaioken_literal_anchor_active", "") or "").strip().lower()
    if active and active in known:
        return True, active
    # Ambiguous advice asks (for example "should I quit?" / "advise me")
    # should bind to the most recent disclosed distress topic, not fall
    # through to generic short-fallback replies.
    if re.search(r"\b(?:should|do)\s+i\s+(?:quit|stop)\b|\badvise me\b", text, flags=re.IGNORECASE):
        open_topics = list(getattr(state, "kaioken_open_topics", []) or [])
        for tok in reversed(open_topics):
            t = str(tok or "").strip().lower()
            if t and t in known:
                return True, t
        if known:
            recent_nouns = [str(x or "").strip().lower() for x in list(getattr(state, "kaioken_recent_user_nouns", []) or [])]
            for tok in reversed(recent_nouns):
                if tok and tok in known:
                    return True, tok
            return True, sorted(known)[0]
    return False, ""


def _remember_recent_concrete_nouns(state: SessionState, user_text: str, *, max_items: int = 24) -> None:
    try:
        recent = list(getattr(state, "kaioken_recent_user_nouns", []) or [])
        recent.extend(_extract_concrete_nouns(user_text))
        # Keep duplicates: repeated mentions are salience signal for semantic resolution.
        setattr(state, "kaioken_recent_user_nouns", recent[-max_items:])
    except Exception:
        pass


def _remember_recent_user_turn(state: SessionState, user_text: str, *, max_items: int = 10) -> None:
    try:
        t = str(user_text or "").strip()
        if not t:
            return
        recent = list(getattr(state, "kaioken_recent_user_turns", []) or [])
        recent.append(t)
        setattr(state, "kaioken_recent_user_turns", recent[-max_items:])
    except Exception:
        pass


def _extract_event_action_tokens(text: str) -> set[str]:
    s = str(text or "").strip().lower()
    if not s:
        return set()
    irregular = {
        "went", "got", "had", "made", "saw", "felt", "lost", "found", "told",
        "walked", "drove", "watched", "called", "tried", "broke", "hurt", "locked",
    }
    toks = set(re.findall(r"\b[a-z][a-z0-9'-]{2,}\b", s))
    actions = {t for t in toks if t in irregular or t.endswith("ed")}
    return actions


def _is_first_person_event_narration(text: str) -> bool:
    first = _first_sentence_for_guard(str(text or ""))
    if not first:
        return False
    if not re.search(r"\b(i|my)\b", first, flags=re.IGNORECASE):
        return False
    # Structural: first-person + past-event action.
    return bool(
        re.search(
            r"\bi\s+(?:went|got|had|walked|watched|drove|saw|felt|lost|found|called|tried|broke|hurt|locked|\w+ed)\b",
            first,
            flags=re.IGNORECASE,
        )
    )


def _is_narrative_ownership_bleed(state: SessionState, assistant_text: str) -> bool:
    if not _is_first_person_event_narration(assistant_text):
        return False
    a_sent = _first_sentence_for_guard(str(assistant_text or ""))
    a_nouns = {x for x in _extract_concrete_nouns(a_sent) if x and x not in _LITERAL_ANCHOR_STOPWORDS}
    a_actions = _extract_event_action_tokens(a_sent)
    if not a_nouns or not a_actions:
        return False
    try:
        recent_user = list(getattr(state, "kaioken_recent_user_turns", []) or [])
    except Exception:
        recent_user = []
    # Match against near history with distance: cluster requires noun+action overlap.
    for ut in reversed(recent_user[-6:]):
        u_nouns = {x for x in _extract_concrete_nouns(ut) if x and x not in _LITERAL_ANCHOR_STOPWORDS}
        u_actions = _extract_event_action_tokens(ut)
        if not u_nouns or not u_actions:
            continue
        if (a_nouns & u_nouns) and (a_actions & u_actions):
            return True
    return False


def _force_narrative_ownership_rewrite(*, call_model_fn, user_text: str, draft_text: str) -> str:
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- Do NOT adopt the user's event as first-person narration.\n"
        "- Do NOT use first-person event claims ('I went', 'my car', 'I had to').\n"
        "- Keep 1-3 sentences, direct and natural.\n"
        "- Stay with the user's perspective (second person or neutral phrasing).\n"
        "- No behavior announcements.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=140,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _resolve_recent_anchor(state: SessionState, query_text: str = "") -> str:
    return _kl_resolve_recent_anchor(state, query_text)


def _first_sentence_for_guard(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)
    return (parts[0] if parts else t).strip()


def _sentence_split_for_guard(text: str) -> list[str]:
    t = str(text or "").strip()
    if not t:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p and str(p).strip()]


def _first_clause_for_anchor_parse(text: str) -> str:
    return _kl_first_clause_for_anchor_parse(text)


def _ends_on_positive_reframe_without_bridge(text: str) -> bool:
    parts = _sentence_split_for_guard(text)
    if len(parts) < 2:
        return False
    last = parts[-1]
    has_reframe_end = bool(_KAIOKEN_POSITIVE_FRAMING_RE.search(last) or _KAIOKEN_END_REFRAME_RE.search(last))
    if not has_reframe_end:
        return False
    # Distress turns should not end on a reframe unless there is explicit
    # acknowledgement/bridge language that preserves concern before reframing.
    has_ack = bool(_KAIOKEN_ACK_RE.search(text))
    has_bridge = bool(_KAIOKEN_BRIDGE_RE.search(text))
    return bool(not (has_ack and has_bridge))


def _friction_repair_line(state: SessionState) -> str:
    opts = (
        "You're right to call that out. What do you want me to answer right now?",
        "Fair call. Give me the exact question you want answered, and I'll answer it cleanly.",
    )
    key = f"{getattr(state, 'kaioken_turn_counter', 0)}|{str(getattr(state, 'session_id', '') or '')}|friction"
    idx = int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % len(opts)
    return opts[idx]


def _force_full_distress_rewrite(*, call_model_fn, user_text: str, draft_text: str) -> str:
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- 2 to 4 sentences.\n"
        "- Direct, human, and specific to the user's message.\n"
        "- No pep-talk slogans, no platitudes, no motivational framing.\n"
        "- No nihilistic agreement ('no point', 'why bother').\n"
        "- Do not validate withdrawal/disengagement (for example 'quitting is okay' or 'not the right fit for you').\n"
        "- No behavior announcements ('I'll keep this direct', etc.).\n"
        "- No unsolicited advice unless the user explicitly asked for advice.\n"
        "- Do not summarize the whole conversation.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=180,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _force_literal_anchor_advice_rewrite(
    *,
    call_model_fn,
    user_text: str,
    draft_text: str,
    anchor: str,
) -> str:
    a = str(anchor or "").strip()
    if not a:
        return ""
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- User explicitly asked for advice about a literal anchor topic.\n"
        f"- Anchor topic: `{a}`.\n"
        "- Keep it to 2-3 sentences.\n"
        "- Stay on this anchor topic (no topic pivot).\n"
        "- Provide practical guidance if possible.\n"
        "- If specifics are uncertain, acknowledge limits clearly without pretending certainty.\n"
        "- No pep-talk slogans, no platitudes, no motivational framing.\n"
        "- No behavior announcements.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=180,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _force_disclosed_distress_topic_advice_rewrite(
    *,
    call_model_fn,
    user_text: str,
    draft_text: str,
    topic: str,
) -> str:
    t = str(topic or "").strip()
    if not t:
        return ""
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- User asked for advice about a previously disclosed personal/distress topic.\n"
        f"- Topic: `{t}`.\n"
        "- Keep it to 1-2 sentences.\n"
        "- First sentence: direct acknowledgment of limits (do not pretend certainty).\n"
        "- Second sentence (optional): brief support direction (appropriate professional/support channel).\n"
        "- Stay on this topic. No pivot to unrelated topics.\n"
        "- No pep-talk slogans, no motivational framing, no behavior announcements.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=110,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _extract_literal_followup_anchor(user_text: str, *, state: SessionState | None = None) -> str:
    return _kl_extract_literal_followup_anchor(user_text, state=state)


def _literal_anchor_is_primary_subject(user_text: str, anchor: str) -> bool:
    return _kl_literal_anchor_is_primary_subject(user_text, anchor)


def _is_literal_followup_turn(*, state: SessionState, user_text: str) -> bool:
    return _kl_is_literal_followup_turn(
        state=state,
        user_text=user_text,
        is_kaioken_guard_candidate_fn=_is_kaioken_guard_candidate,
    )


def _user_explicitly_requests_advice(user_text: str) -> bool:
    return _kc_user_explicitly_requests_advice(user_text)


def _user_explicitly_requests_encouragement(user_text: str) -> bool:
    return _kc_user_explicitly_requests_encouragement(user_text)


def _force_brief_encouragement_rewrite(*, call_model_fn, user_text: str, draft_text: str) -> str:
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- User explicitly requested encouragement.\n"
        "- Keep it to 2-3 sentences maximum.\n"
        "- Be warm and direct, not preachy.\n"
        "- No behavior announcements.\n"
        "- No long motivational essay.\n"
        "- Stay anchored to what the user actually said.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=160,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _is_practical_work_advice_turn(state: SessionState, user_text: str) -> bool:
    t = str(user_text or "").strip()
    if not t:
        return False
    if not _user_explicitly_requests_advice(t):
        return False
    recent = " ".join(
        [
            str(getattr(state, "last_user_text", "") or ""),
            " ".join(str(x or "") for x in list(getattr(state, "kaioken_recent_user_turns", []) or [])[-6:]),
            " ".join(str(x or "") for x in list(getattr(state, "kaioken_open_topics", []) or [])),
            t,
        ]
    )
    return bool(_KAIOKEN_WORK_CONTEXT_RE.search(recent))


def _force_practical_work_advice_rewrite(*, call_model_fn, user_text: str, draft_text: str) -> str:
    prompt = (
        "Rewrite the response with strict constraints:\n"
        "- User asked for practical advice in an already-established work context.\n"
        "- Keep it to 2-4 sentences.\n"
        "- Give concrete operational steps the user can try immediately.\n"
        "- Do NOT deflect with generic clarification requests when context is already established.\n"
        "- Stay on the user's work context and requested problem.\n"
        "- No behavior announcements, no meta-summary.\n\n"
        f"USER:\n{str(user_text or '').strip()}\n\n"
        f"DRAFT:\n{str(draft_text or '').strip()}\n\n"
        "Return only the rewritten answer body."
    )
    try:
        out = str(
            call_model_fn(
                role="thinker",
                prompt=prompt,
                max_tokens=170,
                temperature=0.0,
                top_p=1.0,
            )
            or ""
        ).strip()
        return out
    except Exception:
        return ""


def _is_personal_reassurance_or_tips_request(user_text: str) -> bool:
    t = str(user_text or "").strip().lower()
    if not t:
        return False
    wants_reassure_or_tips = bool(
        re.search(
            r"\b("
            r"reassure me|can you reassure|please reassure|need reassurance|"
            r"any tips|tips\??|give me tips|advice\??|what should i do|what do i do|help me here"
            r")\b",
            t,
            flags=re.IGNORECASE,
        )
    )
    if not wants_reassure_or_tips:
        return False
    if _kaioken_classify_register is None:
        return True
    try:
        cls = _kaioken_classify_register(str(user_text or ""))
        macro = str(getattr(cls, "macro", "") or "").strip().lower()
        return macro == "personal"
    except Exception:
        return True


_KAIOKEN_ROUTING_DEPS = _KaiokenRoutingDeps(
    split_fun_prefix=_split_fun_prefix,
    is_post_resolution_casual_pushback=_is_post_resolution_casual_pushback,
    remember_recent_assistant_body=_remember_recent_assistant_body,
    extract_literal_followup_anchor=_extract_literal_followup_anchor,
    is_literal_followup_turn=_is_literal_followup_turn,
    is_clarification_prompt=_is_clarification_prompt,
    is_continuation_prompt=_is_continuation_prompt,
    open_topics_for_clarify=_open_topics_for_clarify,
    remember_recent_concrete_nouns=_remember_recent_concrete_nouns,
    advice_about_disclosed_distress_topic=_advice_about_disclosed_distress_topic,
    is_narrative_ownership_bleed=_is_narrative_ownership_bleed,
    repeat_like=_repeat_like,
    recent_repeat_within_window=_recent_repeat_within_window,
    friction_re=_KAIOKEN_FRICTION_RE,
    user_explicitly_requests_advice=_user_explicitly_requests_advice,
    is_practical_work_advice_turn=_is_practical_work_advice_turn,
    force_literal_anchor_advice_rewrite=_force_literal_anchor_advice_rewrite,
    user_explicitly_requests_encouragement=_user_explicitly_requests_encouragement,
    force_brief_encouragement_rewrite=_force_brief_encouragement_rewrite,
    force_practical_work_advice_rewrite=_force_practical_work_advice_rewrite,
    force_disclosed_distress_topic_advice_rewrite=_force_disclosed_distress_topic_advice_rewrite,
    force_narrative_ownership_rewrite=_force_narrative_ownership_rewrite,
    choose_short_fallback=_choose_short_fallback,
    friction_repair_line=_friction_repair_line,
    is_kaioken_guard_candidate=_is_kaioken_guard_candidate,
    kaioken_classify_register=_kaioken_classify_register,
    remember_distress_topics_from_turn=_remember_distress_topics_from_turn,
    is_structural_vuh_turn=_is_structural_vuh_turn,
    distress_fallback_re=_KAIOKEN_DISTRESS_FALLBACK_RE,
    pep_talk_re=_KAIOKEN_PEP_TALK_RE,
    advice_re=_KAIOKEN_ADVICE_RE,
    reassure_re=_KAIOKEN_REASSURE_RE,
    help_offer_re=_KAIOKEN_HELP_OFFER_RE,
    announce_re=_KAIOKEN_ANNOUNCE_RE,
    context_deflection_re=_KAIOKEN_CONTEXT_DEFLECTION_RE,
    has_kaioken_descriptor_drift=_has_kaioken_descriptor_drift,
    joke_explain_re=_KAIOKEN_JOKE_EXPLAIN_RE,
    sentence_count_for_guard=_sentence_count_for_guard,
    ends_on_positive_reframe_without_bridge=_ends_on_positive_reframe_without_bridge,
    closed_topics_not_mentioned=_closed_topics_not_mentioned,
    mentions_any_topic=_mentions_any_topic,
    metaphor_pivot_re=_KAIOKEN_METAPHOR_PIVOT_RE,
    diagnostic_re=_KAIOKEN_DIAGNOSTIC_RE,
    nihilism_re=_KAIOKEN_NIHILISM_RE,
    positive_framing_re=_KAIOKEN_POSITIVE_FRAMING_RE,
    first_sentence_for_guard=_first_sentence_for_guard,
    force_full_distress_rewrite=_force_full_distress_rewrite,
    literal_lines=_KAIOKEN_LITERAL_LINES,
)


def _maybe_apply_kaioken_output_guard(
    *,
    state: SessionState,
    user_text: str,
    text: str,
    call_model_fn,
) -> str:
    return _kr_apply_output_guard(
        state=state,
        user_text=user_text,
        text=text,
        call_model_fn=call_model_fn,
        deps=_KAIOKEN_ROUTING_DEPS,
    )


def _apply_closed_topic_suppression(
    *,
    state: SessionState,
    user_text: str,
    text: str,
) -> str:
    """Session-state contract: suppress resurfacing of closed topics independent of guard candidacy."""
    try:
        if not bool(getattr(state, "kaioken_enabled", False)):
            return str(text or "")
        mode = str(getattr(state, "kaioken_mode", "log_only") or "log_only").strip().lower()
        if mode not in {"coerce", "phase1"}:
            return str(text or "")
        blocked_closed_topics = _closed_topics_not_mentioned(state, user_text)
        if not blocked_closed_topics:
            return str(text or "")
        prefix, body = _split_fun_prefix(str(text or ""))
        if not body or not _mentions_any_topic(body, blocked_closed_topics):
            return str(text or "")
        if re.search(r"\b(question|ask)\b", str(user_text or ""), flags=re.IGNORECASE):
            redirect = "Go for it - what's your question?"
        else:
            redirect = "All good - we can leave that closed. What do you want to focus on now?"
        return f"{prefix}\n\n{redirect}".strip() if prefix else redirect
    except Exception:
        return str(text or "")


def _is_explicit_citation_request(text: str) -> bool:
    s = str(text or "")
    if not s:
        return False
    patterns = [
        r"\bcite\b",
        r"\bcitation(?:s)?\b",
        r"\b(?:list|show|provide|give)\s+(?:me\s+)?(?:the\s+)?(?:references|sources|provenance)\b",
        r"\bwhat\s+(?:are|were)\s+(?:your|the)\s+(?:references|sources)\b",
    ]
    return any(re.search(p, s, flags=re.I) for p in patterns)


def _is_structured_schema_task_prompt(text: str) -> bool:
    s = str(text or "")
    if not s:
        return False
    if not re.search(r"\boutput\s+exactly\b", s, flags=re.I):
        return False
    header_markers = [
        "a_argument:",
        "b_argument:",
        "decision:",
        "justification:",
        "stakeholder_a_priorities:",
        "stakeholder_b_priorities:",
        "difference_summary:",
        "what_changed:",
        "no_longer_valid:",
        "updated_conclusion:",
        "contradictions:",
        "source_priority:",
        "conclusion:",
    ]
    low = s.lower()
    return any(h in low for h in header_markers)


_IMAGE_EMITTER_ENABLED = bool(cfg_get("router.image_emitter.enabled", False))
_IMAGE_EMITTER_MARKER = "MOA_IMAGE_EMITTER_V1"
_IMAGE_EMITTER_PATH = str(
    cfg_get(
        "router.image_emitter.path",
        str(Path(__file__).resolve().parents[1] / "TEST_ARTIFACTS_VALIDATION" / "image-emitter.jsonl"),
    )
)


def _summarize_message_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {
            "kind": "str",
            "len": len(content),
            "preview": safe_preview(content, max_len=120),
            "has_image_signal": bool(has_image_signal(content)),
        }
    if isinstance(content, list):
        type_counts: Dict[str, int] = {}
        image_blocks = 0
        text_blocks = 0
        for b in content:
            if isinstance(b, dict):
                t = str(b.get("type", "")).strip().lower() or "dict"
                type_counts[t] = int(type_counts.get(t, 0)) + 1
                if t in {"text", "input_text"}:
                    text_blocks += 1
                if has_image_signal(b):
                    image_blocks += 1
            elif isinstance(b, str):
                type_counts["str"] = int(type_counts.get("str", 0)) + 1
        return {
            "kind": "list",
            "len": len(content),
            "types": type_counts,
            "text_blocks": text_blocks,
            "image_blocks": image_blocks,
            "has_image_signal": bool(has_image_signal(content)),
        }
    if isinstance(content, dict):
        return {
            "kind": "dict",
            "keys": sorted([str(k) for k in list(content.keys())[:20]]),
            "has_image_signal": bool(has_image_signal(content)),
        }
    return {"kind": type(content).__name__, "has_image_signal": bool(has_image_signal(content))}


def _last_user_content_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return _summarize_message_content(m.get("content", ""))
    return {"kind": "none"}


def _emit_image_trace(stage: str, payload: Dict[str, Any]) -> None:
    if not _IMAGE_EMITTER_ENABLED:
        return
    event = {
        "marker": _IMAGE_EMITTER_MARKER,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        **(payload or {}),
    }
    try:
        print(f"[IMAGE-EMITTER] {json.dumps(event, ensure_ascii=False)}", flush=True)
    except Exception:
        pass
    try:
        p = Path(_IMAGE_EMITTER_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _norm_query_text(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip()


def _extract_replay_base_answer(last_assistant_text: str) -> str:
    lines: List[str] = []
    for ln in str(last_assistant_text or "").splitlines():
        s = (ln or "").strip()
        if not s:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        low = s.lower()
        if s.startswith("[FUN]") or s.startswith("[FUN REWRITE]"):
            continue
        if low.startswith("confidence:") or low.startswith("source:") or low.startswith("profile:"):
            continue
        lines.append(s)
    return "\n".join(lines).strip()


_KAIOKEN_TASK_FORWARD_FALLBACK = (
    "What do you need help with right now?"
)
_SERIOUS_TASK_FORWARD_FALLBACK = (
    "I didn't catch that. Could you rephrase?"
)
_DELEGATE_DECISION_RE = re.compile(
    r"\b(you tell me|you decide|you choose|your call|pick for me|choose for me|idk|i dont know|i don't know|not sure)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_QUERY_RE = re.compile(
    r"^\s*(?:(?:so|but|and|well|ok|okay|right|yeah|yep|uh|um)\s+)*(?:"
    r"what(?:'s|s|\s+is)\s+[^\n?]{1,96}(?:\s+exactly)?\??|"
    r"definition\s+of\s+[^\n?]{1,96}\??|"
    r"define\s+[^\n?]{1,96}\??|"
    r"what\s+does\s+[^\n?]{1,96}\s+mean\??|"
    r"[^\n?]{1,96}\s+(?:etymology|origin)\??"
    r")\s*$",
    re.IGNORECASE,
)


def _finalize_chat_response(
    *,
    text: str,
    user_text: str,
    state: SessionState,
    facts_block: str = "",
    lock_active: bool,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    has_facts_block: bool,
    stream: bool,
    mode: str = "serious",
    sensitive_override_once: bool = False,
    bypass_serious_anti_loop: bool = False,
    deterministic_state_solver: bool = False,
    deterministic_output_locked: bool = False,
    scratchpad_lock_miss: bool | None = None,
    scratchpad_lock_miss_indices: List[int] | None = None,
):
    selected_task_forward_fallback = (
        _KAIOKEN_TASK_FORWARD_FALLBACK
        if bool(getattr(state, "kaioken_enabled", False))
        else _SERIOUS_TASK_FORWARD_FALLBACK
    )
    return _chat_finalize_response(
        text=text,
        user_text=user_text,
        state=state,
        facts_block=facts_block,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        scratchpad_quotes=scratchpad_quotes,
        has_facts_block=has_facts_block,
        stream=stream,
        mode=mode,
        sensitive_override_once=sensitive_override_once,
        bypass_serious_anti_loop=bypass_serious_anti_loop,
        deterministic_state_solver=deterministic_state_solver,
        deterministic_output_locked=deterministic_output_locked,
        scratchpad_lock_miss=scratchpad_lock_miss,
        scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
        serious_task_forward_fallback=selected_task_forward_fallback,
        make_stream_response=lambda t: StreamingResponse(_stream_sse(t), media_type="text/event-stream"),
        make_json_response=lambda t: JSONResponse(_make_openai_response(t)),
        sanitize_scratchpad_grounded_output_fn=_pp_sanitize_scratchpad_grounded_output,
        append_scratchpad_provenance_fn=_pp_append_scratchpad_provenance,
        apply_scratchpad_strict_policy_fn=_pp_apply_scratchpad_strict_policy,
        apply_locked_output_policy_fn=_pp_apply_locked_output_policy,
        apply_benchmark_contract_policy_fn=_pp_apply_benchmark_contract_policy,
        rewrite_source_line_fn=_pp_rewrite_source_line,
        apply_deterministic_footer_fn=(
            lambda **kw: _pp_apply_deterministic_footer(
                **kw,
                normalize_sources_footer_fn=normalize_sources_footer,
            )
        ),
        append_profile_footer_fn=(
            lambda **kw: _pp_append_profile_footer(
                **kw,
                cfg_get_fn=cfg_get,
                effective_profile_fn=effective_profile,
            )
        ),
        rewrite_response_style_fn=rewrite_response_style,
        classify_sensitive_context_fn=classify_sensitive_context,
        strip_in_body_confidence_source_claims_fn=_strip_in_body_confidence_source_claims,
        strip_behavior_announcement_sentences_fn=_strip_behavior_announcement_sentences,
        enforce_fun_antiparrot_fn=_enforce_fun_antiparrot,
        strip_irrelevant_proofread_tail_fn=_rr_strip_irrelevant_proofread_tail,
        normalize_agreement_ack_tense_fn=_rr_normalize_agreement_ack_tense,
        classify_query_family_fn=classify_query_family,
        is_ack_reframe_only_fn=_is_ack_reframe_only,
        strip_footer_lines_for_scan_fn=_strip_footer_lines_for_scan,
        normalize_signature_text_fn=_normalize_signature_text,
        score_output_compliance_fn=score_output_compliance,
        compute_effective_strength_fn=compute_effective_strength,
    )


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI()


def _format_router_exception(exc: Exception) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if ROUTER_DEBUG:
        tb = traceback.format_exc()
        log = f"[router][unhandled][{ts}] {exc.__class__.__name__}: {exc}\n{tb}"
    else:
        log = f"[router][unhandled][{ts}] {exc.__class__.__name__}"
    try:
        print(log, flush=True)
    except Exception:
        pass
    if ROUTER_DEBUG:
        return f"[router error: unhandled {exc.__class__.__name__}: {exc}]"
    return f"[router error: unhandled {exc.__class__.__name__}]"


@app.middleware("http")
async def _chat_exception_guard(request: Request, call_next):
    """Fail-soft for chat route so clients do not see transport-level failures."""
    try:
        return await call_next(request)
    except Exception as exc:
        text = _format_router_exception(exc)
        if request.url.path == "/v1/chat/completions":
            return JSONResponse(_make_openai_response(text), status_code=200)
        return JSONResponse({"ok": False, "error": text}, status_code=500)


@app.on_event("startup")
async def _startup_runtime() -> None:
    try:
        base = str(cfg_get("vodka.storage_dir", "") or "").strip()
        subdir = str(cfg_get("vodka.subdir", "vodka") or "vodka").strip()
        purged = purge_session_memory_jsonl(base, vodka_subdir=subdir)
        _dbg(f"[DEBUG] startup session-memory purge deleted={purged}")
        stats = purge_vodka_ctx_facts(
            base,
            vodka_subdir=subdir,
            max_ctx_items=int(cfg_get("vodka.max_ctx_items", 3000) or 3000),
        )
        _dbg(f"[DEBUG] startup vodka-facts janitor stats={stats}")
    except Exception:
        pass
    try:
        if purge_session_kb_jsonl is not None:
            base = str(cfg_get("scratchpad.storage_dir", "") or "").strip()
            if not base:
                base = str(cfg_get("vodka.storage_dir", "") or "").strip()
            stats = purge_session_kb_jsonl(base)
            _dbg(f"[DEBUG] startup session-kb janitor stats={stats}")
    except Exception:
        pass
@app.on_event("shutdown")
async def _shutdown_runtime() -> None:
    return None

@app.get("/healthz")
def healthz():
    return {"ok": True, "version": __version__}


@app.get("/v1/models")
def v1_models():
    """OpenAI-compatible models endpoint."""
    models = []
    for role, model in (ROLES or {}).items():
        if model:
            models.append({"id": model, "object": "model"})
    # Advertise router meta-model
    models.append({"id": "moa-router", "object": "model"})
    return {"object": "list", "data": models}


def _session_id_from_request(req: Request, body: Dict[str, Any]) -> str:
    """Extract session ID from request."""
    # Prefer explicit chat/session headers.
    sid = (
        req.headers.get("x-chat-id")
        or req.headers.get("x-session-id")
        or req.headers.get("x-openwebui-chat-id")
    )
    if sid:
        return sid.strip()

    # Common body-level chat/session ids.
    for key in ("chat_id", "conversation_id", "session_id", "id"):
        v = body.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = body.get("metadata")
    if isinstance(meta, dict):
        for key in ("chat_id", "conversation_id", "session_id", "id"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # Fallback to OpenAI user field
    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()

    # Last resort: client addr
    try:
        host = req.client.host if req.client else "local"
    except Exception:
        host = "local"

    # Final fallback: stable per-host id.
    return f"sess-{host}"


def _stage_openwebui_title_bypass(*, user_text_raw: str) -> Optional[str]:
    """Return title JSON payload when request is an OpenWebUI auto-title task."""
    def _is_openwebui_title_task(t: str) -> bool:
        t_l = t.lower()
        return (
            "generate a concise" in t_l
            and "word title" in t_l
            and "json format" in t_l
            and "<chat_history>" in t_l
        )

    def _openwebui_title_for(t: str) -> str:
        t_l = t.lower()
        if "router" in t_l or "fastapi" in t_l:
            return "Router Debugging"
        return "Chat Summary"

    if not _is_openwebui_title_task(user_text_raw):
        return None
    title_debug = bool(cfg_get("router.debug", False))
    if title_debug:
        _dbg("[DEBUG] openwebui title task bypass")
    return json.dumps({"title": _openwebui_title_for(user_text_raw)}, ensure_ascii=False)


async def _stage_auto_vision_inline(
    *,
    session_id: str,
    raw_messages: List[Dict[str, Any]],
    request_has_images: bool,
    user_text_raw: str,
    state: SessionState,
    stream: bool,
) -> Optional[Any]:
    """Auto-route image-bearing non-command turns to vision role."""
    is_session_command = _is_command(user_text_raw)
    if not request_has_images or is_session_command:
        return None

    if state.fun_sticky or state.fun_rewrite_sticky:
        state.fun_sticky = False
        state.fun_rewrite_sticky = False

    _emit_image_trace(
        "auto_vision_call",
        {
            "session_id": session_id,
            "stream": bool(stream),
            "request_has_images": bool(request_has_images),
            "is_command": bool(is_session_command),
            "last_user_content": _last_user_content_summary(raw_messages),
        },
    )

    text = await _run_sync(
        call_model_messages,
        role="vision",
        messages=raw_messages,
        max_tokens=700,
        temperature=0.2,
        top_p=0.9,
    )
    text = _pp_apply_image_footer(str(text or ""))
    if stream:
        return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
    return JSONResponse(_make_openai_response(text))


def _request_has_images(*, body: Dict[str, Any], raw_messages: List[Dict[str, Any]]) -> bool:
    """Detect image inputs across current and fallback request shapes."""
    if has_images_in_messages(raw_messages):
        return True

    # Last user turn often carries side-channel file metadata.
    for i in range(len(raw_messages) - 1, -1, -1):
        msg = raw_messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if has_image_signal(msg.get("images")):
            return True
        if has_image_signal(msg.get("files")):
            return True
        if has_image_signal(msg.get("attachments")):
            return True
        break

    # Some clients send attachments top-level.
    for key in ("images", "image", "files", "attachments", "input_image", "input_images"):
        if has_image_signal(body.get(key)):
            return True

    return has_image_signal(body.get("messages"))


def _extract_image_urls(value: Any) -> List[str]:
    """Extract image URLs/data-URLs from mixed payload shapes."""
    out: List[str] = []
    seen: set[str] = set()

    def add_url(s: str) -> None:
        u = str(s or "").strip()
        if not u:
            return
        ul = u.lower()
        # Keep to URLs/data-URLs that llama-server can consume directly.
        if not (ul.startswith("data:image/") or ul.startswith("http://") or ul.startswith("https://")):
            return
        if u in seen:
            return
        seen.add(u)
        out.append(u)

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            add_url(v)
            return
        if isinstance(v, list):
            for it in v:
                walk(it)
            return
        if not isinstance(v, dict):
            return

        typ = str(v.get("type", "")).strip().lower()
        mime = str(v.get("mime_type") or v.get("content_type") or v.get("mimetype") or "").strip().lower()

        image_url = v.get("image_url")
        if isinstance(image_url, str):
            add_url(image_url)
        elif isinstance(image_url, dict):
            add_url(str(image_url.get("url") or ""))

        for k in ("url", "uri", "src"):
            val = v.get(k)
            if isinstance(val, str):
                add_url(val)

        data = v.get("data")
        if isinstance(data, str):
            d = data.strip()
            dl = d.lower()
            if dl.startswith("data:image/"):
                add_url(d)
            elif mime.startswith("image/"):
                # Some clients provide raw base64 plus MIME.
                add_url(f"data:{mime};base64,{d}")

        # Recurse across common container keys.
        for k in ("image", "images", "image_url", "file", "files", "attachments", "content", "parts"):
            if k in v:
                walk(v.get(k))

        # If explicit image type/mime exists, also recurse into nested dict/list values.
        if typ in {"image", "image_url", "input_image"} or mime.startswith("image/"):
            for nv in v.values():
                if isinstance(nv, (dict, list)):
                    walk(nv)

    walk(value)
    return out


def _coerce_to_content_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for b in content:
            if isinstance(b, dict):
                blocks.append(dict(b))
            elif isinstance(b, str) and b.strip():
                blocks.append({"type": "text", "text": b})
        return blocks
    if isinstance(content, str):
        t = content.strip()
        return [{"type": "text", "text": t}] if t else []
    if isinstance(content, dict):
        return [dict(content)]
    return []


def _augment_messages_with_request_images(
    *,
    body: Dict[str, Any],
    raw_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure uploaded image(s) are present inside the last user message content blocks.
    This bridges clients that send image metadata outside `messages[].content`.
    """
    msgs: List[Dict[str, Any]] = [dict(m) if isinstance(m, dict) else {} for m in (raw_messages or [])]
    if not msgs:
        msgs = [{"role": "user", "content": ""}]

    # Find last user message (or create one).
    user_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            user_idx = i
            break
    if user_idx < 0:
        msgs.append({"role": "user", "content": ""})
        user_idx = len(msgs) - 1

    last_user = msgs[user_idx]
    content = last_user.get("content", "")
    blocks = _coerce_to_content_blocks(content)
    if any(has_image_signal(b) for b in blocks):
        return msgs

    # Gather images from request-level and message-level side channels.
    side_channels: List[Any] = [
        body.get("images"),
        body.get("image"),
        body.get("files"),
        body.get("attachments"),
        body.get("input_image"),
        body.get("input_images"),
        last_user.get("images"),
        last_user.get("files"),
        last_user.get("attachments"),
    ]
    image_urls: List[str] = []
    for chan in side_channels:
        image_urls.extend(_extract_image_urls(chan))

    # Deduplicate while preserving order.
    dedup_urls: List[str] = []
    seen: set[str] = set()
    for u in image_urls:
        if u in seen:
            continue
        seen.add(u)
        dedup_urls.append(u)

    if not dedup_urls:
        return msgs

    has_text = any(
        isinstance(b, dict) and str(b.get("type", "")).lower() in {"text", "input_text"} and str(b.get("text", "")).strip()
        for b in blocks
    )
    if not has_text:
        blocks.append({"type": "text", "text": "What is in this image?"})

    for u in dedup_urls:
        blocks.append({"type": "image_url", "image_url": {"url": u}})

    last_user["content"] = blocks
    msgs[user_idx] = last_user
    return msgs


def _stage_pending_trust_choice(
    *,
    state: SessionState,
    session_id: str,
    stream: bool,
    raw_messages: List[Dict[str, Any]],
    user_text_raw: str,
    selector: str,
    user_text: str,
) -> Tuple[bool, Optional[Any], str, str, str, List[Dict[str, Any]]]:
    """Handle A/B/C/... selection when trust recommendations are pending."""
    if not state.pending_trust_recommendations:
        return False, None, selector, user_text, user_text_raw, raw_messages

    user_choice = user_text_raw.strip().upper()
    if not (user_choice in ['A', 'B', 'C', 'D', 'E'] and len(user_text_raw.strip()) == 1):
        return False, None, selector, user_text, user_text_raw, raw_messages

    chosen_rec = None
    for rec in state.pending_trust_recommendations:
        if rec['rank'] == user_choice:
            chosen_rec = rec
            break

    if not chosen_rec:
        valid_choices = ', '.join(r['rank'] for r in state.pending_trust_recommendations)
        text = f"[router] Invalid choice. Valid options: {valid_choices}"
        resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
        return True, resp, selector, user_text, user_text_raw, raw_messages

    command = chosen_rec['command']
    original_query = state.pending_trust_query

    # Clear pending state first.
    state.pending_trust_query = ""
    state.pending_trust_recommendations = []
    state.pending_trust_judge_command = ""

    # Guided trust macro: choose A -> prompt paste -> auto add + auto judge.
    if str(chosen_rec.get("flow", "") or "").strip() == "scratch_then_judge":
        judge_cmd = str(chosen_rec.get("judge_command", "") or "").strip()
        if not judge_cmd or not _is_command(judge_cmd):
            text = "[trust] scratch->judge flow misconfigured (missing judge command)"
            resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
            return True, resp, selector, user_text, user_text_raw, raw_messages
        state.pending_trust_judge_command = judge_cmd
        text = (
            "[PASTE EVIDENCE FOR JUDGE]\n"
            "Paste source text now (or type CANCEL).\n"
            "Enter anything else to run ungrounded >>judge."
        )
        resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
        return True, resp, selector, user_text, user_text_raw, raw_messages

    _cmd_norm = (command or "").lstrip()
    if _cmd_norm.startswith("»"):
        _cmd_norm = ">>" + _cmd_norm[1:]
    if _cmd_norm.lower() == '>>attach all' and original_query:
        state.auto_query_after_attach = original_query
        state.auto_detach_after_response = True

    if _is_command(command):
        try:
            cmd_reply = handle_command(command, state=state, session_id=session_id)
            if cmd_reply is not None:
                if state.auto_query_after_attach:
                    auto_query = state.auto_query_after_attach
                    state.auto_query_after_attach = ""
                    user_text_raw = auto_query
                    selector, user_text = _split_selector(user_text_raw)
                    for i in range(len(raw_messages) - 1, -1, -1):
                        if raw_messages[i].get("role") == "user":
                            raw_messages[i]["content"] = auto_query
                            break
                    return False, None, selector, user_text, user_text_raw, raw_messages
                if maybe_capture_command_output is not None:
                    try:
                        maybe_capture_command_output(
                            session_id=session_id,
                            state=state,
                            cmd_text=command,
                            reply_text=cmd_reply,
                        )
                    except Exception:
                        pass
                resp = StreamingResponse(_stream_sse(cmd_reply), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(cmd_reply))
                return True, resp, selector, user_text, user_text_raw, raw_messages
        except Exception as e:
            text = f"[router error: {e.__class__.__name__}: {e}]"
            resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
            return True, resp, selector, user_text, user_text_raw, raw_messages
    elif command.startswith('##'):
        user_text_raw = command
        selector, user_text = _split_selector(user_text_raw)
        for i in range(len(raw_messages) - 1, -1, -1):
            if raw_messages[i].get("role") == "user":
                raw_messages[i]["content"] = command
                break
    else:
        user_text_raw = command
        selector, user_text = _split_selector(user_text_raw)
        for i in range(len(raw_messages) - 1, -1, -1):
            if raw_messages[i].get("role") == "user":
                raw_messages[i]["content"] = command
                break

    return False, None, selector, user_text, user_text_raw, raw_messages


def _stage_pending_trust_judge_context(
    *,
    state: SessionState,
    session_id: str,
    stream: bool,
    raw_messages: List[Dict[str, Any]],
    user_text_raw: str,
    selector: str,
    user_text: str,
) -> Tuple[bool, Optional[Any], str, str, str, List[Dict[str, Any]]]:
    """Handle guided paste step for trust -> scratch -> judge flow."""
    judge_cmd = str(getattr(state, "pending_trust_judge_command", "") or "").strip()
    if not judge_cmd:
        return False, None, selector, user_text, user_text_raw, raw_messages

    def _respond(text: str) -> Tuple[bool, Optional[Any], str, str, str, List[Dict[str, Any]]]:
        resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
        return True, resp, selector, user_text, user_text_raw, raw_messages

    incoming = str(user_text_raw or "")
    trimmed = incoming.strip()

    # Allow explicit cancel from paste step.
    if trimmed.upper() in {"CANCEL", ">>CANCEL"}:
        state.pending_trust_judge_command = ""
        return _respond("[trust] scratch->judge canceled.")

    # Empty / whitespace-only input: run ungrounded judge path explicitly.
    if not trimmed:
        note = "[NO CONTEXT ADDED; RUNNING DEFAULT >>JUDGE RANKING]"
        prev_attached = set(state.attached_kbs or set())
        prev_lock = set(getattr(state, "scratchpad_locked_indices", set()) or set())
        try:
            # Force ungrounded fallback for this one guided invocation.
            state.attached_kbs.discard("scratchpad")
            state.scratchpad_locked_indices.clear()
            judge_reply = handle_command(judge_cmd, state=state, session_id=session_id)
        finally:
            state.attached_kbs = set(prev_attached)
            state.scratchpad_locked_indices = set(int(i) for i in prev_lock if int(i) > 0)
            state.pending_trust_judge_command = ""
        if judge_reply is None:
            return _respond(f"{note}\n\n[trust] judge execution failed")
        return _respond(f"{note}\n\n{judge_reply}")

    # Non-empty input: treat as scratch context and run judge immediately.
    if capture_scratchpad_output is None:
        state.pending_trust_judge_command = ""
        return _respond("[scratchpad] add unavailable")

    state.attached_kbs.add("scratchpad")
    rec = capture_scratchpad_output(
        session_id=session_id,
        source_command=">>trust paste",
        text=incoming,
    )
    ack = "[scratchpad] add failed"
    if rec:
        ack = f"[scratchpad] added sha={str(rec.get('sha256', ''))[:12]}"

    judge_reply = handle_command(judge_cmd, state=state, session_id=session_id)
    state.pending_trust_judge_command = ""
    if judge_reply is None:
        return _respond(f"{ack}\n\n[trust] judge execution failed")
    return _respond(f"{ack}\n\n{judge_reply}")


def _stage_pending_lock_confirmation(
    *,
    state: SessionState,
    session_id: str,
    stream: bool,
    selector: str,
    user_text_raw: str,
) -> Tuple[bool, Optional[Any]]:
    """Handle pending lock Y/N confirmation path."""
    if not (selector == "" and state.pending_lock_candidate):
        return False, None

    yn = (user_text_raw or "").strip().lower()
    if yn in ("y", "yes"):
        lock_target = state.pending_lock_candidate
        state.pending_lock_candidate = ""
        try:
            cmd_reply = handle_command(f">>lock {lock_target}", state=state, session_id=session_id)
        except Exception as e:
            text = f"[router error: lock confirm crashed: {e.__class__.__name__}: {e}]"
            resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
            return True, resp
        text = cmd_reply or f"[router] lock target not found in attached KBs: {lock_target}"
        resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
        return True, resp
    if yn in ("n", "no"):
        state.pending_lock_candidate = ""
        text = "[router] lock suggestion cancelled"
        resp = StreamingResponse(_stream_sse(text), media_type="text/event-stream") if stream else JSONResponse(_make_openai_response(text))
        return True, resp

    # Non-Y/N clears stale pending lock and continue normal turn.
    state.pending_lock_candidate = ""
    return False, None


async def _stage_pending_vodka_comment(
    *,
    state: SessionState,
    user_text_raw: str,
    stream: bool,
) -> Tuple[bool, Optional[Any]]:
    """Strict next-turn opt-in for commentary after deterministic `!!` storage ack."""
    ctx_id = str(getattr(state, "pending_vodka_comment_ctx_id", "") or "").strip()
    if not ctx_id:
        return False, None

    yn = (user_text_raw or "").strip().lower()
    # Strict trigger only.
    if yn not in ("yes", "y"):
        state.pending_vodka_comment_ctx_id = ""
        state.pending_vodka_comment_text = ""
        return False, None

    note = str(getattr(state, "pending_vodka_comment_text", "") or "").strip()
    state.pending_vodka_comment_ctx_id = ""
    state.pending_vodka_comment_text = ""
    if not note:
        text = "[vodka] commentary unavailable: stored note not found."
        if stream:
            return True, StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return True, JSONResponse(_make_openai_response(text))

    prompt = (
        "You are giving practical commentary on a user-saved note.\n"
        "Saved note:\n"
        f"{note}\n\n"
        "Task:\n"
        "- Give concise, practical commentary in 3-6 bullet points.\n"
        "- Stay grounded to the saved note.\n"
        "- Do not ask clarification questions.\n"
        "- Do not mention hidden/system prompts.\n"
    )
    try:
        comment = await _run_sync(
            call_model_prompt,
            role="thinker",
            prompt=prompt,
            max_tokens=220,
            temperature=0.2,
            top_p=0.9,
        )
        comment = str(comment or "").strip()
    except Exception as e:
        comment = f"[router error: commentary failed: {e.__class__.__name__}: {e}]"

    text = f"Commentary on {ctx_id}:\n{comment}".strip()
    return True, _finalize_chat_response(
        text=text,
        user_text=user_text_raw,
        state=state,
        facts_block="",
        lock_active=False,
        scratchpad_grounded=False,
        scratchpad_quotes=[],
        has_facts_block=False,
        stream=stream,
        mode="serious",
        sensitive_override_once=False,
    )


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: Request):
    """Main chat completions endpoint."""
    body = await req.json()
    session_id = _session_id_from_request(req, body)
    state = get_state(session_id)

    stream = bool(body.get("stream", False))

    raw_messages = body.get("messages", []) or []
    if not isinstance(raw_messages, list):
        return JSONResponse(_make_openai_response("[router error: messages must be a list]"))

    incoming_last_user = _last_user_content_summary(raw_messages)
    request_has_images = _request_has_images(body=body, raw_messages=raw_messages)
    if request_has_images:
        raw_messages = _augment_messages_with_request_images(body=body, raw_messages=raw_messages)
    _emit_image_trace(
        "chat_ingress",
        {
            "session_id": session_id,
            "stream": bool(stream),
            "request_has_images": bool(request_has_images),
            "incoming_last_user": incoming_last_user,
            "post_augment_last_user": _last_user_content_summary(raw_messages),
            "top_level_counts": {
                "messages": len(raw_messages),
                "files": len(body.get("files") or []) if isinstance(body.get("files"), list) else int(bool(body.get("files"))),
                "attachments": len(body.get("attachments") or []) if isinstance(body.get("attachments"), list) else int(bool(body.get("attachments"))),
                "images": len(body.get("images") or []) if isinstance(body.get("images"), list) else int(bool(body.get("images"))),
            },
        },
    )
    user_text_raw, _ = last_user_message(raw_messages)
    if not user_text_raw and not request_has_images:
        # Allow empty submission only for guided trust paste step.
        if not str(getattr(state, "pending_trust_judge_command", "") or "").strip():
            return JSONResponse(_make_openai_response("[router error: no user message]"))
        user_text_raw = ""
    if not user_text_raw and request_has_images:
        user_text_raw = "[image]"

    title_json = _stage_openwebui_title_bypass(user_text_raw=user_text_raw)
    if title_json is not None:
        return JSONResponse(_make_openai_response(title_json))

    auto_vision_resp = await _stage_auto_vision_inline(
        session_id=session_id,
        raw_messages=raw_messages,
        request_has_images=request_has_images,
        user_text_raw=user_text_raw,
        state=state,
        stream=stream,
    )
    if auto_vision_resp is not None:
        return auto_vision_resp
    if request_has_images:
        _emit_image_trace(
            "auto_vision_miss",
            {
                "session_id": session_id,
                "is_command": bool(_is_command(user_text_raw)),
                "user_text_raw": safe_preview(user_text_raw, max_len=120),
                "last_user_content": _last_user_content_summary(raw_messages),
            },
        )

    # Check for per-turn selectors (##) FIRST
    selector, user_text = _split_selector(user_text_raw)
    sensitive_override_once = False
    _update_blocked_nicknames(state, user_text_raw)

    trust_handled, trust_resp, selector, user_text, user_text_raw, raw_messages = _stage_pending_trust_choice(
        state=state,
        session_id=session_id,
        stream=stream,
        raw_messages=raw_messages,
        user_text_raw=user_text_raw,
        selector=selector,
        user_text=user_text,
    )
    if trust_handled:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "pending_trust_choice"},
        )
        return trust_resp

    trust_judge_handled, trust_judge_resp, selector, user_text, user_text_raw, raw_messages = _stage_pending_trust_judge_context(
        state=state,
        session_id=session_id,
        stream=stream,
        raw_messages=raw_messages,
        user_text_raw=user_text_raw,
        selector=selector,
        user_text=user_text,
    )
    if trust_judge_handled:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "pending_trust_judge_context"},
        )
        return trust_judge_resp

    lock_handled, lock_resp = _stage_pending_lock_confirmation(
        state=state,
        session_id=session_id,
        stream=stream,
        selector=selector,
        user_text_raw=user_text_raw,
    )
    if lock_handled:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "pending_lock_confirmation"},
        )
        return lock_resp

    vodka_comment_handled, vodka_comment_resp = await _stage_pending_vodka_comment(
        state=state,
        user_text_raw=user_text_raw,
        stream=stream,
    )
    if vodka_comment_handled:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "pending_vodka_comment"},
        )
        return vodka_comment_resp

    preflight_handled, preflight_resp, selector, user_text, user_text_raw, raw_messages, sensitive_override_once = _handle_preflight(
        state=state,
        session_id=session_id,
        stream=stream,
        selector=selector,
        user_text=user_text,
        user_text_raw=user_text_raw,
        raw_messages=raw_messages,
        split_selector=_split_selector,
        is_command=_is_command,
        requires_sensitive_confirm=(
            lambda s, t: _requires_sensitive_confirm(
                state=s,
                user_text=t,
                classify_sensitive_context_fn=classify_sensitive_context,
            )
        ),
        handle_command=handle_command,
        maybe_capture_command_output=maybe_capture_command_output,
        soft_alias_command=(
            lambda t, s: _chat_soft_alias_command(
                text=t,
                state=s,
                is_command_fn=_is_command,
                kb_paths=KB_PATHS,
                vault_kb_name=VAULT_KB_NAME,
            )
        ),
        make_stream_response=lambda t: StreamingResponse(_stream_sse(t), media_type="text/event-stream"),
        make_json_response=lambda t: JSONResponse(_make_openai_response(t)),
    )
    if preflight_handled:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "preflight"},
        )
        return preflight_resp

    _dbg(f"[DEBUG] selector={selector!r}, user_text={_debug_user_fragment(user_text)}")

    # Update KAIOKEN topic state before any downstream early-return paths
    # (for example Vodka hard-command short-circuit) so closure/switch signals
    # in this user turn are persisted deterministically.
    _update_kaioken_topic_state_preturn(state, user_text)
    kaioken_macro_for_vodka = "working"
    kaioken_confidence_for_vodka = "low"
    try:
        if _kaioken_classify_register is not None:
            cls_now = _kaioken_classify_register(str(user_text or ""))
            macro_raw = str(getattr(cls_now, "macro", "") or "").strip().lower()
            conf_raw = str(getattr(cls_now, "confidence", "") or "").strip().lower()
            if macro_raw in {"working", "casual", "personal"}:
                kaioken_macro_for_vodka = macro_raw
            if conf_raw in {"low", "medium", "high", "med"}:
                kaioken_confidence_for_vodka = "medium" if conf_raw == "med" else conf_raw
    except Exception:
        kaioken_macro_for_vodka = "working"
        kaioken_confidence_for_vodka = "low"

    vodka, raw_messages, _NoOpVodka, vodka_meta = _apply_vodka_runtime(
        state=state,
        raw_messages=raw_messages,
        session_id=session_id,
        kaioken_macro=kaioken_macro_for_vodka,
        kaioken_confidence=kaioken_confidence_for_vodka,
        cfg_get=cfg_get,
        VodkaFilter=VodkaFilter,
        preset_for_session=(lambda s, c: _vr_preset_for_session(s, c, _VODKA_PRESET_MAP)),
        preset_map=_VODKA_PRESET_MAP,
        debug_fn=_dbg,
    )

    # Check if Vodka already answered hard commands (?? list, !! nuke)
    if raw_messages and raw_messages[-1].get("role") == "assistant":
        last_msg_content = raw_messages[-1].get("content", "")
        if isinstance(last_msg_content, str) and (
            "[vodka]" in last_msg_content.lower() or
            "[vodka memory store]" in last_msg_content.lower() or
            "[recall_det]" in last_msg_content.lower()
        ):
            suppress_recall_fastpath = bool(
                "[recall_det]" in last_msg_content.lower()
                and _is_personal_reassurance_or_tips_request(user_text_raw)
            )
            if not suppress_recall_fastpath and "[recall_det]" in last_msg_content.lower():
                blocked_closed_topics = _closed_topics_not_mentioned(state, user_text_raw)
                if blocked_closed_topics and _mentions_any_topic(last_msg_content, blocked_closed_topics):
                    suppress_recall_fastpath = True
            if suppress_recall_fastpath:
                # Keep explicit reassurance/tips requests in generation lane.
                # Do not short-circuit into deterministic recall dumps.
                raw_messages = raw_messages[:-1]
            else:
            # Arm strict next-turn commentary lane only for deterministic !! add acks.
                added_ctx = str(vodka_meta.get("_vodka_added_ctx_id", "") or "").strip()
                if added_ctx:
                    state.pending_vodka_comment_ctx_id = added_ctx
                    state.pending_vodka_comment_text = str(vodka_meta.get("_vodka_added_text", "") or "")
                elif "[vodka] nuked" in last_msg_content.lower() or "[vodka] forget=" in last_msg_content.lower():
                    state.pending_vodka_comment_ctx_id = ""
                    state.pending_vodka_comment_text = ""
                # This is a Vodka hard command answer - return it directly
                out_text = last_msg_content
                if "[recall_det]" in out_text.lower():
                    draft, recall_query, evidence, is_payload = _parse_recall_det_payload(out_text)
                    semantic_enabled = bool(cfg_get("recall.semantic_layer_enabled", True))
                    if is_payload and semantic_enabled:
                        out_text = _synthesize_recall_answer(
                            draft=draft,
                            query=recall_query,
                            evidence=evidence,
                            cfg_get=cfg_get,
                            call_model_prompt=call_model_prompt,
                        )
                    else:
                        out_text = draft
                    if not out_text:
                        out_text = re.sub(r"^\s*\[recall_det\]\s*", "", last_msg_content, count=1, flags=re.IGNORECASE).strip()
                _emit_kaioken_telemetry(
                    state=state,
                    session_id=session_id,
                    user_text=user_text_raw,
                    route_class="control",
                    outcome={"stage": "vodka_hard_command"},
                )
                if stream:
                    return StreamingResponse(_stream_sse(out_text), media_type="text/event-stream")
                return JSONResponse(_make_openai_response(out_text))

    history_text_only = normalize_history(raw_messages)
    try:
        # Hydrate per-request state from normalized history so repeat guards
        # can compare against the true previous assistant turn across requests.
        state.last_assistant_text = str(_last_assistant_text(history_text_only) or "")
    except Exception:
        pass

    if state.profile_enabled:
        try:
            state.profile_turn_counter += 1
            update_profile_from_user_turn(
                state.interaction_profile,
                state.profile_turn_counter,
                user_text_raw,
            )
            state.profile_effective_strength = compute_effective_strength(
                state.interaction_profile,
                enabled=state.profile_enabled,
                output_compliance=state.profile_output_compliance,
            )
        except Exception:
            pass
    else:
        state.profile_effective_strength = 0.0

    vision_text = await _handle_vision_ocr_selector(
        selector=selector,
        raw_messages=raw_messages,
        run_sync=_run_sync,
        call_model_messages=call_model_messages,
    )
    if vision_text is not None:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="vision",
            outcome={"stage": "vision_selector"},
        )
        vision_text = _pp_apply_image_footer(str(vision_text or ""))
        if stream:
            return StreamingResponse(_stream_sse(vision_text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(vision_text))

    mentats_text = await _handle_mentats_selector(
        selector=selector,
        session_id=session_id,
        state=state,
        user_text=user_text,
        run_sync=_run_sync,
        run_mentats=run_mentats,
        build_vault_facts=(
            lambda q, s: _cf_build_vault_facts(
                query=q,
                state=s,
                build_rag_block_fn=_build_rag_block,
                vault_kb_name=VAULT_KB_NAME,
            )
        ),
        call_model_prompt=call_model_prompt,
        no_op_vodka_cls=_NoOpVodka,
        facts_collection=VAULT_KB_NAME,
        debug_fn=lambda msg: _dbg(f"{msg}, user_text={_debug_user_fragment(user_text)}"),
    )
    if mentats_text is not None:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="control",
            outcome={"stage": "mentats_selector"},
        )
        if stream:
            return StreamingResponse(_stream_sse(mentats_text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(mentats_text))

    fun_mode, fun_block = _resolve_fun_mode(
        selector=selector,
        state=state,
        history_text_only=history_text_only,
        has_mentats_in_recent_history=has_mentats_in_recent_history,
    )
    if fun_block:
        if stream:
            return StreamingResponse(_stream_sse(fun_block), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(fun_block))

    # Deterministic early route for machine-checkable state transitions.
    lock_active_now = bool(state.locked_summ_path)
    # Keep explicit distress-advice asks in model generation lane; do not
    # short-circuit into deterministic state-solver responses.
    kaioken_mode_now = str(getattr(state, "kaioken_mode", "log_only") or "log_only").strip().lower()
    skip_state_solver_early = bool(
        bool(getattr(state, "kaioken_enabled", False))
        and kaioken_mode_now in {"coerce", "phase1"}
        and _user_explicitly_requests_advice(user_text)
        and (
            bool(set(getattr(state, "kaioken_distress_topics", set()) or set()))
            or _KAIOKEN_DISTRESS_FALLBACK_RE.search(str(user_text or ""))
            or _KAIOKEN_DISTRESS_FALLBACK_RE.search(str(getattr(state, "last_user_text", "") or ""))
        )
    )
    early_state = None if skip_state_solver_early else _maybe_handle_state_solver_early(
        state=state,
        user_text=user_text,
        selector=selector,
        fun_mode=fun_mode,
        lock_active_now=lock_active_now,
        stream=stream,
        sensitive_override_once=sensitive_override_once,
        cfg_get=cfg_get,
        classify_constraint_turn=classify_constraint_turn,
        classify_query_family=classify_query_family,
        solve_state_transition_query=solve_state_transition_query,
        solve_state_transition_followup=solve_state_transition_followup,
        solve_constraint_followup=solve_constraint_followup,
        semantic_pick_clarifier_option=(
            lambda **kw: _semantic_pick_clarifier_option(
                **kw,
                call_model_prompt_fn=call_model_prompt,
            )
        ),
        semantic_refine_constraint_choice=(
            lambda **kw: _semantic_refine_constraint_choice(
                **kw,
                call_model_prompt_fn=call_model_prompt,
            )
        ),
        finalize_chat_response=_finalize_chat_response,
        debug_fn=lambda msg: _dbg(msg.replace(user_text, _debug_user_fragment(user_text))),
    )
    if early_state is not None:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="deterministic",
            outcome={"stage": "state_solver_early"},
        )
        return early_state

    # Default: serious reasoning
    # Classify once and carry through this turn for retrieval/guard shaping.
    turn_macro = "working"
    turn_subsignals: set[str] = set()
    try:
        if _kaioken_classify_register is not None:
            cls_now = _kaioken_classify_register(str(user_text or ""))
            turn_macro = str(getattr(cls_now, "macro", "working") or "working").strip().lower()
            turn_subsignals = {
                str(s).strip().lower()
                for s in list(getattr(cls_now, "subsignals", []) or [])
                if str(s or "").strip()
            }
    except Exception:
        turn_macro = "working"
        turn_subsignals = set()
    state.turn_kaioken_macro = str(turn_macro or "").strip().lower()

    # Bug 2b (upstream gate): suppress filesystem KB retrieval on casual/personal turns
    # unless the user has explicitly locked a source.
    if turn_macro in {"casual", "personal"} and not bool(lock_active_now):
        facts_block = ""
    else:
        facts_block = _cf_build_fs_facts(
            query=user_text,
            state=state,
            build_locked_summ_facts_block_fn=build_locked_summ_facts_block,
            build_fs_facts_block_fn=build_fs_facts_block,
            cfg_get_fn=cfg_get,
            fs_top_k=FS_TOP_K,
            fs_max_chars=FS_MAX_CHARS,
            kb_paths=KB_PATHS,
            vault_kb_name=VAULT_KB_NAME,
        )
    # Reset per-turn retrieval overrides before optional static/wiki grounding.
    state.turn_footer_source_override = ""
    state.turn_footer_confidence_override = ""
    state.turn_retrieval_track = ""
    state.turn_local_knowledge_line = ""
    state.turn_cheatsheets_warning_line = ""
    state.turn_cheatsheets_warning_key = ""
    cheatsheets_constraints_block = ""
    cheatsheets_deterministic_answer = ""
    # Cheatsheets retrieval (Track A) + optional wiki recovery (Track B).
    # Track B is intentionally working-only by design; do not harmonize silently.
    try:
        if _resolve_cheatsheets_turn is not None:
            macro = turn_macro
            subsignals = set(turn_subsignals)
            cheat = _resolve_cheatsheets_turn(
                user_text=str(user_text or ""),
                macro=macro,
                subsignals=subsignals,
                cheatsheets_dir=_CHEATSHEETS_DIR,
                has_existing_facts=bool((facts_block or "").strip()),
                prior_distress_carry=bool(
                    bool(getattr(state, "kaioken_short_fallback_distress_lane", False))
                    or bool(set(getattr(state, "kaioken_distress_topics", set()) or set()))
                ),
                prior_user_text=str(getattr(state, "last_user_text", "") or ""),
                track_b_enabled=bool(cfg_get("cheatsheets.track_b.enabled", False)),
                wiki_lookup_fn=_sidecar_wiki_query,
            )
            state.turn_footer_source_override = str(cheat.footer_source or "").strip()
            state.turn_footer_confidence_override = str(cheat.footer_confidence or "").strip().lower()
            state.turn_retrieval_track = str(cheat.track or "").strip()
            state.turn_local_knowledge_line = str(cheat.local_knowledge_line or "").strip()
            state.turn_kaioken_macro = str(macro or "").strip().lower()
            cheatsheets_constraints_block = str(cheat.constraints_block or "").strip()
            cheatsheets_deterministic_answer = str(cheat.deterministic_answer or "").strip()
            # Non-blocking JSONL parse diagnostics: surface exact file/line issue to user
            # so malformed cheatsheet entries can be fixed quickly.
            if _get_cheatsheets_parse_warnings is not None:
                try:
                    parse_warns = [str(w).strip() for w in list(_get_cheatsheets_parse_warnings() or ()) if str(w).strip()]
                except Exception:
                    parse_warns = []
                if parse_warns:
                    preview = "; ".join(parse_warns[:2])
                    if len(parse_warns) > 2:
                        preview = f"{preview}; +{len(parse_warns) - 2} more"
                    state.turn_cheatsheets_warning_line = (
                        f"[cheatsheets warning] {preview}. Fix the JSONL entry and retry."
                    )
                    state.turn_cheatsheets_warning_key = hashlib.sha1(
                        "\n".join(sorted(parse_warns)).encode("utf-8")
                    ).hexdigest()
            if cheat.facts_block:
                facts_block = (
                    f"{facts_block}\n\n{cheat.facts_block}".strip()
                    if facts_block
                    else str(cheat.facts_block).strip()
                )
    except Exception:
        pass
    lock_active = lock_active_now
    scratchpad_quotes: List[str] = []
    scratchpad_grounded = False
    scratchpad_lock_miss = False
    scratchpad_lock_miss_indices: List[int] = []
    constraints_block = ""
    if state.profile_enabled:
        try:
            if state.profile_effective_strength >= 0.35 or has_non_default_style(state.interaction_profile):
                constraints_block = build_profile_constraints_block(state.interaction_profile, user_text)
        except Exception:
            constraints_block = ""
    if lock_active:
        lock_constraints = _pp_lock_constraints_block(state.locked_summ_file)
        constraints_block = (
            f"{lock_constraints}\n\n{constraints_block}".strip() if constraints_block else lock_constraints
        )
    if cheatsheets_constraints_block:
        constraints_block = (
            f"{constraints_block}\n\n{cheatsheets_constraints_block}".strip()
            if constraints_block
            else cheatsheets_constraints_block
        )
    scratchpad_exhaustive = False
    scratchpad_locked_indices = set(
        int(i)
        for i in (getattr(state, "scratchpad_locked_indices", set()) or set())
        if str(i).strip().isdigit() and int(i) > 0
    )
    scratchpad_exhaustive_mode = str(cfg_get("scratchpad.exhaustive_response_mode", "raw") or "raw").strip().lower()
    if (not lock_active) and "scratchpad" in state.attached_kbs and build_scratchpad_facts_block is not None:
        try:
            if wants_exhaustive_query is not None:
                scratchpad_exhaustive = bool(wants_exhaustive_query(user_text))
            sp_top_k = int(cfg_get("scratchpad.top_k", 3))
            sp_max_chars = int(cfg_get("scratchpad.max_chars", 1200))
            sp_block = build_scratchpad_facts_block(
                session_id=session_id,
                query=user_text,
                top_k=sp_top_k,
                max_chars=sp_max_chars,
                locked_indices=scratchpad_locked_indices,
            )
            if sp_block:
                scratchpad_grounded = True
                scratchpad_quotes = _pp_scratchpad_quote_lines(sp_block, query=user_text)
                if not scratchpad_quotes:
                    # Robustness: if query-filtered extraction misses due
                    # tokenization edge cases, fall back to any quote span.
                    scratchpad_quotes = _pp_scratchpad_quote_lines(sp_block, query="")
                facts_block = f"{facts_block}\n\n{sp_block}".strip() if facts_block else sp_block
                scratchpad_constraints = (
                    "Grounding mode: SCRATCHPAD.\n"
                    "- Prefer FACTS provided in this turn (scratchpad facts).\n"
                    "- Give a complete direct answer in normal prose.\n"
                    "- Synthesize from provided facts; do not just dump quote snippets.\n"
                    "- If you infer beyond explicit facts, mark that portion as model supplement."
                )
                if re.search(r"\b(same|similar|difference|different|compare|conceptual(?:ly)?)\b", user_text or "", re.I):
                    scratchpad_constraints += (
                        "\n- For comparison questions, answer with an explicit verdict first "
                        "(e.g., Yes/No/Partly), then explain using at least two grounded facts."
                    )
                constraints_block = (
                    f"{scratchpad_constraints}\n\n{constraints_block}".strip()
                    if constraints_block
                    else scratchpad_constraints
                )
            else:
                has_query_tokens = bool(re.search(r"[A-Za-z0-9]{3,}", user_text or ""))
                if scratchpad_locked_indices and has_query_tokens:
                    scratchpad_lock_miss = True
                    scratchpad_lock_miss_indices = sorted(scratchpad_locked_indices)
        except Exception:
            pass

    # Deterministic reference mode for strict scratch grounding.
    cite_like_query = _is_explicit_citation_request(user_text)
    structured_schema_task_prompt = _is_structured_schema_task_prompt(user_text)
    if (
        scratchpad_grounded
        and cite_like_query
        and (not structured_schema_task_prompt)
        and list_scratchpad_records is not None
    ):
        try:
            recs = list_scratchpad_records(session_id, limit=1000000)
            if scratchpad_locked_indices:
                filtered = []
                for i, rec in enumerate(recs, 1):
                    if i in scratchpad_locked_indices:
                        filtered.append((i, rec))
            else:
                filtered = list(enumerate(recs, 1))
            rows = []
            for idx, rec in filtered[:8]:
                src = str(rec.get("source_command", "unknown") or "unknown")
                sha = str(rec.get("sha256", "") or "")[:12]
                txt = str(rec.get("text", "") or "").strip()
                preview = " ".join(txt.split())
                if len(preview) > 220:
                    preview = preview[:219].rstrip() + "..."
                rows.append(f"- [{idx}] cmd={src} sha={sha} quote=\"{preview}\"")
            if rows:
                text = "[Scratch]\n\nReferences:\n" + "\n".join(rows)
            else:
                text = (
                    "[Scratch]\n\nNo scratchpad references are available for the current selection."
                )
            _emit_kaioken_telemetry(
                state=state,
                session_id=session_id,
                user_text=user_text_raw,
                route_class="sidecar",
                outcome={"stage": "scratch_citation_reference_mode"},
            )
            text = _pp_append_scratchpad_provenance(text)
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        except Exception:
            pass

    # Deterministic dump path for explicit exhaustive intents.
    if (
        scratchpad_grounded
        and scratchpad_exhaustive
        and scratchpad_exhaustive_mode == "raw"
        and build_scratchpad_dump_text is not None
    ):
        try:
            dump_text = build_scratchpad_dump_text(
                session_id=session_id,
                query=user_text,
                locked_indices=scratchpad_locked_indices,
            )
            text = dump_text.strip() if dump_text else "[scratchpad] empty"
            text = f"[Scratch]\n\n{text}".strip()
            _emit_kaioken_telemetry(
                state=state,
                session_id=session_id,
                user_text=user_text_raw,
                route_class="sidecar",
                outcome={"stage": "scratch_exhaustive_dump"},
            )
            text = _pp_append_scratchpad_provenance(text)
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        except Exception:
            pass

    solver_state = _resolve_state_solver_for_turn(
        state=state,
        user_text=user_text,
        lock_active=lock_active,
        cfg_get=cfg_get,
        classify_constraint_turn=classify_constraint_turn,
        classify_query_family=classify_query_family,
        solve_state_transition_query=solve_state_transition_query,
        solve_state_transition_followup=solve_state_transition_followup,
        solve_constraint_followup=solve_constraint_followup,
        norm_query_text=_norm_query_text,
        extract_replay_base_answer=_extract_replay_base_answer,
        delegate_decision_match=(lambda t: bool(_DELEGATE_DECISION_RE.search(t))),
    )
    query_family = str(solver_state.get("query_family") or "other")
    state_solver_used = bool(solver_state.get("state_solver_used"))
    state_solver_fail_loud = bool(solver_state.get("state_solver_fail_loud"))
    state_solver_answer = str(solver_state.get("state_solver_answer") or "")
    state_solver_reason = str(solver_state.get("state_solver_reason") or "")
    state_solver_frame = dict(solver_state.get("state_solver_frame") or {})
    user_q_norm = str(solver_state.get("user_q_norm") or "")

    # KAIOKEN phase1 soft guidance is model-chat only and should apply
    # consistently to both serious and fun/fun-rewrite generation paths.
    kaioken_hint = _build_kaioken_soft_constraints(state=state, user_text=user_text)
    if kaioken_hint:
        constraints_block = (
            f"{constraints_block}\n\n{kaioken_hint}".strip()
            if constraints_block
            else kaioken_hint
        )
    kaioken_literal_followup = bool(_is_literal_followup_turn(state=state, user_text=user_text))
    kaioken_guard_state: Dict[str, Any] = {"applied": False}

    # Deterministic strict-lookup lane: for fully grounded cheatsheet
    # "who/what is" style queries, return the exact stored definition verbatim.
    # This bypasses thinker paraphrase while preserving normal footer pipeline.
    if cheatsheets_deterministic_answer and (not lock_active):
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="deterministic",
            outcome={"stage": "cheatsheets_strict_lookup"},
        )
        return _finalize_chat_response(
            text=cheatsheets_deterministic_answer,
            user_text=user_text,
            state=state,
            facts_block=facts_block,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="serious",
            sensitive_override_once=sensitive_override_once,
            deterministic_output_locked=True,
            scratchpad_lock_miss=scratchpad_lock_miss,
            scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
        )

    def _kaioken_postguard(text: str) -> str:
        pre = str(text or "")
        suppressed = _apply_closed_topic_suppression(
            state=state,
            user_text=user_text,
            text=pre,
        )
        rewritten = _maybe_apply_kaioken_output_guard(
            state=state,
            user_text=user_text,
            text=suppressed,
            call_model_fn=call_model_prompt,
        )
        if str(rewritten or "") != pre:
            kaioken_guard_state["applied"] = True
        return rewritten

    mode_resp = await _maybe_handle_fun_fr_raw(
        fun_mode=fun_mode,
        state=state,
        user_text=user_text,
        session_id=session_id,
        history_text_only=history_text_only,
        facts_block=facts_block,
        constraints_block=constraints_block,
        scratchpad_grounded=scratchpad_grounded,
        scratchpad_quotes=scratchpad_quotes,
        lock_active=lock_active,
        stream=stream,
        sensitive_override_once=sensitive_override_once,
        state_solver_answer=state_solver_answer,
        run_sync=_run_sync,
        run_fun=run_fun,
        run_raw=run_raw,
        run_serious=run_serious,
        run_fun_rewrite_fallback=run_fun_rewrite_fallback,
        call_model_prompt=call_model_prompt,
        no_op_vodka_cls=_NoOpVodka,
        select_fun_style_seed=_select_fun_style_seed,
        serious_max_tokens_for_query=(
            lambda q: _rr_serious_max_tokens_for_query(q, int(cfg_get("serious.max_tokens", 384)))
        ),
        is_argumentative_prompt=_is_argumentative_prompt,
        is_argumentatively_complete=_is_argumentatively_complete,
        fallback_with_mode_header=_fallback_with_mode_header,
        finalize_chat_response=(
            lambda **kw: _finalize_chat_response(
                scratchpad_lock_miss=scratchpad_lock_miss,
                scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
                **kw,
            )
        ),
        kaioken_postguard_fn=(
            _kaioken_postguard
        ),
    )
    if mode_resp is not None:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="model_chat",
            outcome={
                "stage": "mode_execution",
                "mode": str(fun_mode or "serious"),
                "kaioken_hint_applied": bool(kaioken_hint),
                "kaioken_guard_applied": bool(kaioken_guard_state.get("applied", False)),
                "kaioken_literal_followup": bool(kaioken_literal_followup),
                "kaioken_mode": str(getattr(state, "kaioken_mode", "log_only") or "log_only"),
            },
        )
        if state_solver_used and state_solver_answer:
            state.deterministic_last_family = str(query_family or "state_transition")
            state.deterministic_last_reason = str(state_solver_reason or "")
            state.deterministic_last_answer = str(state_solver_answer or "")
            state.deterministic_last_query_norm = user_q_norm
            if isinstance(state_solver_frame, dict) and state_solver_frame:
                state.deterministic_last_frame = dict(state_solver_frame)
        return mode_resp

    # Normal serious
    if state_solver_used and state_solver_answer:
        _emit_kaioken_telemetry(
            state=state,
            session_id=session_id,
            user_text=user_text_raw,
            route_class="deterministic",
            outcome={"stage": "state_solver_resolved"},
        )
        state.deterministic_last_family = str(query_family or "state_transition")
        state.deterministic_last_reason = str(state_solver_reason or "")
        state.deterministic_last_answer = str(state_solver_answer or "")
        state.deterministic_last_query_norm = user_q_norm
        if isinstance(state_solver_frame, dict) and state_solver_frame:
            state.deterministic_last_frame = dict(state_solver_frame)
        return _finalize_chat_response(
            text=state_solver_answer,
            user_text=user_text,
            state=state,
            facts_block=facts_block,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="serious",
            sensitive_override_once=sensitive_override_once,
            bypass_serious_anti_loop=True,
            scratchpad_lock_miss=scratchpad_lock_miss,
            scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
        )

    if int(getattr(state, "serious_ack_reframe_streak", 0) or 0) >= 1:
        anti_loop = (
            "Anti-loop rule:\n"
            "- Previous turn already used acknowledgement/reframe.\n"
            "- Do NOT output another meta acknowledgement about tone/process.\n"
            "- Provide direct task-forward content in <=2 sentences."
        )
        constraints_block = f"{constraints_block}\n\n{anti_loop}".strip() if constraints_block else anti_loop

    # When deterministic decision lane has been explicitly disengaged and user turn
    # is a correction (e.g., "I meant X not Y"), bind correction to immediately prior
    # model answer unless user explicitly asks to re-engage deterministic lane.
    try:
        prior_answer_for_struct = str(getattr(state, "last_assistant_text", "") or "").strip() or _last_assistant_text(history_text_only)
        recent_nouns_for_struct = [
            str(x or "").strip().lower()
            for x in list(getattr(state, "kaioken_recent_user_nouns", []) or [])
            if str(x or "").strip()
        ]
        recent_thread_tokens_for_struct: list[str] = []
        try:
            threads = _get_kaioken_threads(state)
            active_tid = str(getattr(state, "kaioken_active_thread_id", "") or "").strip()
            active_meta = threads.get(active_tid) if active_tid else None
            if isinstance(active_meta, dict):
                recent_thread_tokens_for_struct = sorted(_thread_tokens(active_meta))
        except Exception:
            recent_thread_tokens_for_struct = []

        structural_corr = _evaluate_structural_correction_intent(
            user_text=user_text,
            prior_assistant_text=prior_answer_for_struct,
            recent_user_nouns=recent_nouns_for_struct,
            recent_thread_tokens=recent_thread_tokens_for_struct,
        )
        correction_bind_structural = bool(structural_corr.get("is_structural", False))
        # Primary path: structural intent from deterministic state/turn shape.
        # Regex path remains as fallback debt only.
        correction_bind_regex_fallback = bool(
            _is_correction_intent_query(user_text) and not _is_explicit_reengage_query(user_text)
        )
        correction_bind = bool(correction_bind_structural or correction_bind_regex_fallback)
        correction_is_strong = False
        skip_correction_bind = False
        if correction_bind:
            low = str(user_text or "").lower()
            is_definitional_turn = bool(_DEFINITIONAL_QUERY_RE.search(str(user_text or "")))
            new_v, unit_v, _old_v = _extract_numeric_correction(user_text)
            has_numeric_rest = bool(new_v and unit_v)
            has_i_meant = bool(
                re.search(r"\b(i meant|no i meant|sorry i meant)\b", low, flags=re.IGNORECASE)
            )
            has_contradiction_target = bool(
                re.search(
                    r"\bnot\b[^\n]{0,96}\b(?:but|rather than|instead of)\b|\b(?:rather than|instead of)\b",
                    low,
                    flags=re.IGNORECASE,
                )
            )
            has_explicit_rest = bool(has_i_meant and has_contradiction_target)
            has_not_but_rest = bool(re.search(r"\bnot\b[^\n]{0,64}\bbut\b", low, flags=re.IGNORECASE))
            lexical_strong = bool(has_numeric_rest or has_explicit_rest or has_not_but_rest)
            correction_is_strong = bool(correction_bind_structural or lexical_strong)
            if correction_bind_structural:
                try:
                    if _kaioken_classify_register is not None:
                        cls_cur = _kaioken_classify_register(str(user_text or ""))
                        macro_cur = str(getattr(cls_cur, "macro", "") or "").strip().lower()
                        conf_label = str(getattr(cls_cur, "confidence", "low") or "low").strip().lower()
                        conf_medium_or_higher = conf_label in {"high", "medium", "med"}
                        if macro_cur == "working" and (not conf_medium_or_higher):
                            skip_correction_bind = True
                except Exception:
                    pass
            if is_definitional_turn:
                # Low-blast exemption: definitional pivots should answer directly,
                # not rewrite the previous assistant turn as a "correction".
                skip_correction_bind = True
            if not correction_is_strong:
                macro_personal = False
                macro_casual = False
                conf_label = "low"
                try:
                    if _kaioken_classify_register is not None:
                        cls_cur = _kaioken_classify_register(str(user_text or ""))
                        macro_cur = str(getattr(cls_cur, "macro", "") or "").strip().lower()
                        macro_personal = macro_cur == "personal"
                        macro_casual = macro_cur == "casual"
                        conf_label = str(getattr(cls_cur, "confidence", "low") or "low").strip().lower()
                except Exception:
                    macro_personal = False
                    macro_casual = False
                    conf_label = "low"
                prior_distress_carry = bool(
                    bool(getattr(state, "kaioken_short_fallback_distress_lane", False))
                    or bool(set(getattr(state, "kaioken_distress_topics", set()) or set()))
                )
                conf_medium_or_higher = conf_label in {"high", "medium", "med"}
                skip_correction_bind = bool(
                    skip_correction_bind
                    or macro_personal
                    or macro_casual
                    or (not conf_medium_or_higher)
                    or prior_distress_carry
                )
        if correction_bind and not skip_correction_bind:
            correction_resp = _maybe_handle_correction_bind(
                state=state,
                user_text=user_text,
                history_text_only=history_text_only,
                query_family=query_family,
                lock_active=lock_active,
                scratchpad_grounded=scratchpad_grounded,
                scratchpad_quotes=scratchpad_quotes,
                facts_block=facts_block,
                stream=stream,
                sensitive_override_once=sensitive_override_once,
                cfg_get=cfg_get,
                call_model_messages=call_model_messages,
                is_correction_intent_query=_is_correction_intent_query,
                is_explicit_reengage_query=_is_explicit_reengage_query,
                extract_numeric_correction=_extract_numeric_correction,
                resolve_old_from_prior_answer=_resolve_old_from_prior_answer,
                fallback_contextual_correction=_fallback_contextual_correction,
                strip_got_it_prefix=_strip_got_it_prefix,
                last_assistant_text=_last_assistant_text,
                last_user_text_before=_last_user_text_before,
                last_user_text=_last_user_text,
                last_non_correction_user_text=_last_non_correction_user_text,
                to_km=_to_km,
                maybe_apply_consistency_verifier=(
                    lambda **kw: _maybe_apply_consistency_verifier(
                        **kw,
                        call_model_messages_fn=call_model_messages,
                    )
                ),
                finalize_chat_response=_finalize_chat_response,
                correction_bind_active=correction_bind,
            )
            if correction_resp is not None:
                _emit_kaioken_telemetry(
                    state=state,
                    session_id=session_id,
                    user_text=user_text_raw,
                    route_class="model_chat",
                    outcome={
                        "stage": "correction_bind",
                        "correction_signal_strength": (
                            "structural"
                            if correction_bind_structural
                            else ("strong" if correction_is_strong else "ambiguous")
                        ),
                        "correction_structural_prior_claim": bool(structural_corr.get("prior_claim_exists", False)),
                        "correction_structural_references_prior": bool(structural_corr.get("references_prior", False)),
                        "correction_structural_replacement_payload": bool(structural_corr.get("has_replacement_payload", False)),
                    },
                )
                return correction_resp
        if correction_bind and not skip_correction_bind:
            correction_hint = (
                "Correction-binding rule:\n"
                "- Treat this user turn as a correction to your immediately previous answer.\n"
                "- Stay on that immediate prior topic unless the user explicitly asks to switch.\n"
                "- Apply corrected values directly; do not revert to older deterministic decision templates."
            )
            constraints_block = (
                f"{constraints_block}\n\n{correction_hint}".strip()
                if constraints_block
                else correction_hint
            )
    except Exception:
        pass

    serious_max_tokens = _rr_serious_max_tokens_for_query(user_text, int(cfg_get("serious.max_tokens", 384)))
    text = (await _run_sync(
        run_serious,
        session_id=session_id,
        user_text=user_text,
        history=history_text_only,
        vodka=_NoOpVodka(),
        call_model=call_model_prompt,
        facts_block=facts_block,
        constraints_block=constraints_block,
        thinker_role="thinker",
        max_tokens=serious_max_tokens,
    )).strip()
    text = _maybe_apply_consistency_verifier(
        user_text=user_text,
        draft_text=text,
        query_family=query_family,
        lock_active=lock_active,
        state_solver_used=state_solver_used,
        prior_user_text=str(getattr(state, "last_user_text", "") or "").strip(),
        call_model_messages_fn=call_model_messages,
    )
    pre_guard_text = text
    text = _apply_closed_topic_suppression(
        state=state,
        user_text=user_text,
        text=text,
    )
    text = _maybe_apply_kaioken_output_guard(
        state=state,
        user_text=user_text,
        text=text,
        call_model_fn=call_model_prompt,
    )
    kaioken_guard_applied = str(pre_guard_text or "") != str(text or "")
    # Serious/model answer path usually supersedes deterministic follow-up context.
    # Exception: preserve an explicitly disengaged decision lane frame so users can
    # re-engage deterministically with an explicit cue in later turns.
    keep_decision_lane_frame = False
    try:
        fr = dict(getattr(state, "deterministic_last_frame", {}) or {})
        keep_decision_lane_frame = (
            str(getattr(state, "deterministic_last_family", "") or "") == "constraint_decision"
            and str(fr.get("kind") or "") == "option_feasibility"
            and bool(fr.get("decision_lane_disengaged", False))
        )
    except Exception:
        keep_decision_lane_frame = False
    if not keep_decision_lane_frame:
        state.deterministic_last_family = ""
        state.deterministic_last_reason = ""
        state.deterministic_last_answer = ""
        state.deterministic_last_frame = {}
        state.deterministic_last_query_norm = ""

    _emit_kaioken_telemetry(
        state=state,
        session_id=session_id,
        user_text=user_text_raw,
        route_class="model_chat",
        outcome={
            "stage": "serious",
            "kaioken_hint_applied": bool(kaioken_hint),
            "kaioken_guard_applied": bool(kaioken_guard_applied),
            "kaioken_literal_followup": bool(kaioken_literal_followup),
            "kaioken_mode": str(getattr(state, "kaioken_mode", "log_only") or "log_only"),
        },
    )

    return _finalize_chat_response(
        text=text,
        user_text=user_text,
        state=state,
        facts_block=facts_block,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        scratchpad_quotes=scratchpad_quotes,
        has_facts_block=bool((facts_block or "").strip()),
        stream=stream,
        mode="serious",
        sensitive_override_once=sensitive_override_once,
        scratchpad_lock_miss=scratchpad_lock_miss,
        scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
    )


# Convenience for `python router_fastapi.py`
if __name__ == "__main__":
    import uvicorn

    host = str(cfg_get("server.host", "0.0.0.0"))
    port = int(cfg_get("server.port", 9000))
    uvicorn.run("router_fastapi:app", host=host, port=port, reload=False)

