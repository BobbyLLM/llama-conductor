"""FastAPI orchestration layer for llama-conductor.

Responsibilities:
- expose API routes
- route requests across command/selectors/pipelines
- apply shared post-processing and response normalization
"""

from __future__ import annotations
import json
import re
import traceback
from datetime import datetime, timezone
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
from .session_state import get_state, SessionState
from .helpers import (
    has_mentats_in_recent_history,
    normalize_history,
    last_user_message,
    is_command as _is_command,
    split_selector as _split_selector,
    _has_image_blocks,
)
from .model_calls import call_model_prompt, call_model_messages
from .streaming import make_openai_response as _make_openai_response, stream_sse as _stream_sse
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
from .vodka_filter import Filter as VodkaFilter, purge_session_memory_jsonl
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
    )
except Exception:
    maybe_capture_command_output = None  # type: ignore
    build_scratchpad_facts_block = None  # type: ignore
    wants_exhaustive_query = None  # type: ignore
    build_scratchpad_dump_text = None  # type: ignore

try:
    from .sources_footer import normalize_sources_footer  # type: ignore
except Exception:
    normalize_sources_footer = None  # type: ignore

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
from .state_solver_flow import maybe_handle_state_solver_early as _maybe_handle_state_solver_early
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
from .router_correction_utils import (
    is_correction_intent_query as _is_correction_intent_query,
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
    from .state_reasoning import (  # type: ignore
        classify_query_family,
        solve_state_transition_query,
        is_followup_consistency_query,
        solve_constraint_followup,
    )
except Exception:
    classify_query_family = None  # type: ignore
    solve_state_transition_query = None  # type: ignore
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


_SERIOUS_TASK_FORWARD_FALLBACK = (
    "Understood. No more meta. Give me the exact task and desired output format, and I will answer directly."
)


def _finalize_chat_response(
    *,
    text: str,
    user_text: str,
    state: SessionState,
    lock_active: bool,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    has_facts_block: bool,
    stream: bool,
    mode: str = "serious",
    sensitive_override_once: bool = False,
    bypass_serious_anti_loop: bool = False,
):
    return _chat_finalize_response(
        text=text,
        user_text=user_text,
        state=state,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        scratchpad_quotes=scratchpad_quotes,
        has_facts_block=has_facts_block,
        stream=stream,
        mode=mode,
        sensitive_override_once=sensitive_override_once,
        bypass_serious_anti_loop=bypass_serious_anti_loop,
        serious_task_forward_fallback=_SERIOUS_TASK_FORWARD_FALLBACK,
        make_stream_response=lambda t: StreamingResponse(_stream_sse(t), media_type="text/event-stream"),
        make_json_response=lambda t: JSONResponse(_make_openai_response(t)),
        sanitize_scratchpad_grounded_output_fn=_pp_sanitize_scratchpad_grounded_output,
        append_scratchpad_provenance_fn=_pp_append_scratchpad_provenance,
        apply_locked_output_policy_fn=_pp_apply_locked_output_policy,
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
async def _startup_cleanup() -> None:
    try:
        base = str(cfg_get("vodka.storage_dir", "") or "").strip()
        purged = purge_session_memory_jsonl(base)
        _dbg(f"[DEBUG] startup session-memory purge deleted={purged}")
    except Exception:
        pass


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

    user_text_raw, _ = last_user_message(raw_messages)
    if not user_text_raw:
        return JSONResponse(_make_openai_response("[router error: no user message]"))

    # ---------------------------------------------------------------------
    # OpenWebUI meta-prompts (auto-title generation)
    # Short-circuit this task locally with a cheap heuristic
    # ---------------------------------------------------------------------
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

    if _is_openwebui_title_task(user_text_raw):
        if ROUTER_DEBUG:
            _dbg("[DEBUG] openwebui title task bypass")
        title_json = json.dumps({"title": _openwebui_title_for(user_text_raw)}, ensure_ascii=False)
        return JSONResponse(_make_openai_response(title_json))

    # CRITICAL CHECK: Auto-vision detection
    user_idx = None
    for i in range(len(raw_messages) - 1, -1, -1):
        if raw_messages[i].get("role") == "user":
            user_idx = i
            break
    
    has_images = False
    if user_idx is not None:
        content = raw_messages[user_idx].get("content", "")
        if isinstance(content, list):
            has_images = _has_image_blocks(content)
    
    is_session_command = _is_command(user_text_raw)
    has_user_text = bool((user_text_raw or "").strip()) and user_text_raw.strip() != "[image]"
    
    if has_images and has_user_text and not is_session_command:
        # Images + text (no command): force vision mode; disable sticky Fun/FR
        if state.fun_sticky or state.fun_rewrite_sticky:
            state.fun_sticky = False
            state.fun_rewrite_sticky = False
        # Auto-route to vision pipeline
        text = await _run_sync(
            call_model_messages,
            role="vision",
            messages=raw_messages,
            max_tokens=700,
            temperature=0.2,
            top_p=0.9,
        )
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Check for per-turn selectors (##) FIRST
    selector, user_text = _split_selector(user_text_raw)
    sensitive_override_once = False
    _update_blocked_nicknames(state, user_text_raw)
    
    # CRITICAL: Check if user is responding to pending trust recommendations (A/B/C/D/E)
    if state.pending_trust_recommendations:
        user_choice = user_text_raw.strip().upper()
        if user_choice in ['A', 'B', 'C', 'D', 'E'] and len(user_text_raw.strip()) == 1:
            # Find the chosen recommendation
            chosen_rec = None
            for rec in state.pending_trust_recommendations:
                if rec['rank'] == user_choice:
                    chosen_rec = rec
                    break
            
            if chosen_rec:
                command = chosen_rec['command']
                original_query = state.pending_trust_query
                
                # Clear pending state FIRST
                state.pending_trust_query = ""
                state.pending_trust_recommendations = []
                
                # Special handling for >>attach all - auto-run query afterward
                _cmd_norm = (command or "").lstrip()
                if _cmd_norm.startswith("Â»"):
                    _cmd_norm = ">>" + _cmd_norm[1:]
                if _cmd_norm.lower() == '>>attach all' and original_query:
                    state.auto_query_after_attach = original_query
                    state.auto_detach_after_response = True
                
                # Execute the chosen command
                if _is_command(command):
                    try:
                        cmd_reply = handle_command(command, state=state, session_id=session_id)
                        if cmd_reply is not None:
                            # Check if we should auto-run a query after this command
                            if state.auto_query_after_attach:
                                auto_query = state.auto_query_after_attach
                                state.auto_query_after_attach = ""
                                
                                # Inject auto query and continue processing
                                user_text_raw = auto_query
                                selector, user_text = _split_selector(user_text_raw)
                                
                                # Update raw_messages
                                for i in range(len(raw_messages) - 1, -1, -1):
                                    if raw_messages[i].get("role") == "user":
                                        raw_messages[i]["content"] = auto_query
                                        break
                            else:
                                # No auto query - return the command result
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
                                if stream:
                                    return StreamingResponse(_stream_sse(cmd_reply), media_type="text/event-stream")
                                return JSONResponse(_make_openai_response(cmd_reply))
                    except Exception as e:
                        text = f"[router error: {e.__class__.__name__}: {e}]"
                        if stream:
                            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                        return JSONResponse(_make_openai_response(text))
                elif command.startswith('##'):
                    # It's a per-turn selector
                    user_text_raw = command
                    selector, user_text = _split_selector(user_text_raw)
                    for i in range(len(raw_messages) - 1, -1, -1):
                        if raw_messages[i].get("role") == "user":
                            raw_messages[i]["content"] = command
                            break
                else:
                    # It's a regular query
                    user_text_raw = command
                    selector, user_text = _split_selector(user_text_raw)
                    for i in range(len(raw_messages) - 1, -1, -1):
                        if raw_messages[i].get("role") == "user":
                            raw_messages[i]["content"] = command
                            break
            else:
                # Invalid choice
                valid_choices = ', '.join(r['rank'] for r in state.pending_trust_recommendations)
                text = f"[router] Invalid choice. Valid options: {valid_choices}"
                if stream:
                    return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                return JSONResponse(_make_openai_response(text))

    # Pending lock confirmation (Y/N) for partial lock suggestions.
    if selector == "" and state.pending_lock_candidate:
        yn = (user_text_raw or "").strip().lower()
        if yn in ("y", "yes"):
            lock_target = state.pending_lock_candidate
            state.pending_lock_candidate = ""
            try:
                cmd_reply = handle_command(f">>lock {lock_target}", state=state, session_id=session_id)
            except Exception as e:
                text = f"[router error: lock confirm crashed: {e.__class__.__name__}: {e}]"
                if stream:
                    return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                return JSONResponse(_make_openai_response(text))
            text = cmd_reply or f"[router] lock target not found in attached KBs: {lock_target}"
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        if yn in ("n", "no"):
            state.pending_lock_candidate = ""
            text = "[router] lock suggestion cancelled"
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        # Non-Y/N input clears stale confirmation and proceeds normally.
        state.pending_lock_candidate = ""

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
        return preflight_resp
    
    _dbg(f"[DEBUG] selector={selector!r}, user_text={_debug_user_fragment(user_text)}")

    vodka, raw_messages, _NoOpVodka = _apply_vodka_runtime(
        state=state,
        raw_messages=raw_messages,
        session_id=session_id,
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
            if stream:
                return StreamingResponse(_stream_sse(out_text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(out_text))
    
    history_text_only = normalize_history(raw_messages)

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
    early_state = _maybe_handle_state_solver_early(
        state=state,
        user_text=user_text,
        fun_mode=fun_mode,
        lock_active_now=lock_active_now,
        stream=stream,
        sensitive_override_once=sensitive_override_once,
        cfg_get=cfg_get,
        classify_query_family=classify_query_family,
        solve_state_transition_query=solve_state_transition_query,
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
        return early_state

    # Default: serious reasoning
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
    lock_active = lock_active_now
    scratchpad_quotes: List[str] = []
    scratchpad_grounded = False
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
    scratchpad_exhaustive = False
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
            )
            if sp_block:
                scratchpad_grounded = True
                scratchpad_quotes = _pp_scratchpad_quote_lines(sp_block, query=user_text)
                facts_block = f"{facts_block}\n\n{sp_block}".strip() if facts_block else sp_block
                scratchpad_constraints = (
                    "Grounding mode: SCRATCHPAD_ONLY.\n"
                    "- Use only FACTS provided in this turn (scratchpad facts).\n"
                    "- If FACTS are insufficient, say so explicitly.\n"
                    "- Do not use pretrained/background knowledge.\n"
                    "- Keep claims constrained to provided facts."
                )
                constraints_block = (
                    f"{scratchpad_constraints}\n\n{constraints_block}".strip()
                    if constraints_block
                    else scratchpad_constraints
                )
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
            dump_text = build_scratchpad_dump_text(session_id=session_id, query=user_text)
            text = dump_text.strip() if dump_text else "[scratchpad] empty"
            text = _pp_append_scratchpad_provenance(text)
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        except Exception:
            pass

    # Universal control pattern (phase 1): deterministic state-transition routing.
    query_family = "other"
    state_solver_used = False
    state_solver_fail_loud = False
    state_solver_answer = ""
    state_solver_reason = ""
    if classify_query_family is not None and solve_state_transition_query is not None:
        try:
            query_family = classify_query_family(user_text)
            if (
                bool(cfg_get("state_solver.enabled", True))
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
                        state_solver_reason = str(getattr(sr, "reason", "") or "")
                        state_solver_answer = str(getattr(sr, "answer", "") or "").strip()
        except Exception:
            pass

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
        finalize_chat_response=_finalize_chat_response,
    )
    if mode_resp is not None:
        return mode_resp

    # Normal serious
    # Defensive re-check: if initial state-solver gate was missed upstream, apply deterministic
    # state transition solve here before falling back to model serious path.
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
                    state_solver_reason = str(getattr(sr2, "reason", "") or "")
                    state_solver_answer = str(getattr(sr2, "answer", "") or "").strip()
        except Exception:
            pass

    if state_solver_used and state_solver_answer:
        state.deterministic_last_family = str(query_family or "state_transition")
        state.deterministic_last_reason = str(state_solver_reason or "")
        state.deterministic_last_answer = str(state_solver_answer or "")
        return _finalize_chat_response(
            text=state_solver_answer,
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
        )
        if correction_resp is not None:
            return correction_resp
        correction_bind = _is_correction_intent_query(user_text) and not _is_explicit_reengage_query(user_text)
        if correction_bind:
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

    return _finalize_chat_response(
        text=text,
        user_text=user_text,
        state=state,
        lock_active=lock_active,
        scratchpad_grounded=scratchpad_grounded,
        scratchpad_quotes=scratchpad_quotes,
        has_facts_block=bool((facts_block or "").strip()),
        stream=stream,
        mode="serious",
        sensitive_override_once=sensitive_override_once,
    )


# Convenience for `python router_fastapi.py`
if __name__ == "__main__":
    import uvicorn

    host = str(cfg_get("server.host", "0.0.0.0"))
    port = int(cfg_get("server.port", 9000))
    uvicorn.run("router_fastapi:app", host=host, port=port, reload=False)


