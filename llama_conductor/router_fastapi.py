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
from .session_state import get_state, SessionState
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


_SERIOUS_TASK_FORWARD_FALLBACK = (
    "Understood. No more meta. Give me the exact task and desired output format, and I will answer directly."
)
_DELEGATE_DECISION_RE = re.compile(
    r"\b(you tell me|you decide|you choose|your call|pick for me|choose for me|idk|i dont know|i don't know|not sure)\b",
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
    scratchpad_lock_miss: bool | None = None,
    scratchpad_lock_miss_indices: List[int] | None = None,
):
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
        scratchpad_lock_miss=scratchpad_lock_miss,
        scratchpad_lock_miss_indices=scratchpad_lock_miss_indices,
        serious_task_forward_fallback=_SERIOUS_TASK_FORWARD_FALLBACK,
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
    if _cmd_norm.startswith("Â»"):
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
        return trust_judge_resp

    lock_handled, lock_resp = _stage_pending_lock_confirmation(
        state=state,
        session_id=session_id,
        stream=stream,
        selector=selector,
        user_text_raw=user_text_raw,
    )
    if lock_handled:
        return lock_resp

    vodka_comment_handled, vodka_comment_resp = await _stage_pending_vodka_comment(
        state=state,
        user_text_raw=user_text_raw,
        stream=stream,
    )
    if vodka_comment_handled:
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
        return preflight_resp
    
    _dbg(f"[DEBUG] selector={selector!r}, user_text={_debug_user_fragment(user_text)}")

    vodka, raw_messages, _NoOpVodka, vodka_meta = _apply_vodka_runtime(
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
    )
    if mode_resp is not None:
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
        state.deterministic_last_query_norm = ""

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



