# router_fastapi.py
# version 1.2.0-refactored
"""MoA Router (FastAPI) - Refactored

CHANGES IN v1.2.0-refactored:
- MAJOR REFACTOR: Split 2,373 lines into focused modules:
  - config.py: Configuration management
  - session_state.py: Session state dataclass
  - helpers.py: Message parsing utilities
  - model_calls.py: LLM API calls
  - quotes.py: Quote loading and tone inference
  - streaming.py: SSE streaming responses
  - vault_ops.py: Vault and SUMM operations
  - commands/: Command handlers (cliniko, handlers)
- router_fastapi.py now ~700 lines (was 2,373)
- Zero breaking changes - all functionality preserved
- All imports lazy-loaded for optional dependencies

This file is now the thin orchestration layer:
- FastAPI app and routes
- Main /v1/chat/completions endpoint
- Pipeline routing logic
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Core modules (always required)
from .config import cfg_get, ROLES, KB_PATHS, VAULT_KB_NAME, FS_TOP_K, FS_MAX_CHARS
from .session_state import get_state, SessionState
from .helpers import (
    has_images_in_messages,
    has_mentats_in_recent_history,
    normalize_history,
    last_user_message,
    is_command as _is_command,
    split_selector as _split_selector,
    _has_image_blocks,
)
from .model_calls import call_model_prompt, call_model_messages
from .quotes import infer_tone as _infer_tone, pick_quote_for_tone as _pick_quote_for_tone, quotes_by_tag as _quotes_by_tag
from .streaming import make_openai_response as _make_openai_response, stream_sse as _stream_sse
from .commands import handle_command

# Required modules
from .vodka_filter import Filter as VodkaFilter
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


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

def build_fs_facts(query: str, state: SessionState) -> str:
    """Build facts block from filesystem KBs."""
    q = " ".join((query or "").split()).strip()
    state.rag_last_query = q
    state.rag_last_hits = 0
    state.locked_last_fact_lines = 0

    if not q:
        return ""

    # Strict lock mode: deterministic grounding from one SUMM file.
    if state.locked_summ_path:
        if not build_locked_summ_facts_block:
            return ""
        max_chars = int(cfg_get("lock.max_chars", max(FS_MAX_CHARS, 12000)))
        txt, n = build_locked_summ_facts_block(
            query=q,
            kb=state.locked_summ_kb or "locked",
            file=state.locked_summ_file or "SUMM_locked.md",
            rel_path=state.locked_summ_rel_path or "",
            abs_path=state.locked_summ_path,
            max_chars=max_chars,
        )
        state.rag_last_hits = int(n)
        state.locked_last_fact_lines = int(n)
        return txt or ""

    if not build_fs_facts_block:
        return ""

    # Only filesystem KBs (exclude vault)
    kbs = {k for k in state.attached_kbs if k != VAULT_KB_NAME}
    if not kbs:
        return ""

    txt = build_fs_facts_block(q, kbs, KB_PATHS, top_k=FS_TOP_K, max_chars=FS_MAX_CHARS)
    if txt:
        # cheap hit count approximation
        state.rag_last_hits = txt.count("[kb=")
    return txt or ""


def build_vault_facts(query: str, state: SessionState) -> str:
    """Build facts block from Qdrant vault."""
    q = " ".join((query or "").split()).strip()
    state.vault_last_query = q
    state.vault_last_hits = 0

    if not q:
        return ""

    # lazy import
    try:
        from .rag import build_rag_block
    except Exception:
        return ""

    try:
        txt = build_rag_block(q, attached_kbs={VAULT_KB_NAME})
    except Exception as e:
        print(f"[router] build_rag_block failed: {e}")
        return ""
    
    if txt:
        # approximate hits
        state.vault_last_hits = max(1, txt.count("\n\n"))
    return txt or ""


def run_fun_rewrite_fallback(*, session_id: str, user_text: str, history: List[Dict[str, Any]], vodka: VodkaFilter, facts_block: str) -> str:
    """Two-pass Fun Rewrite implemented in-router (only used if fun.py doesn't provide it)."""

    # pass 1: serious answer (not shown)
    base = run_serious(
        session_id=session_id,
        user_text=user_text,
        history=history,
        vodka=vodka,
        call_model=call_model_prompt,
        facts_block=facts_block,
        constraints_block="",
        thinker_role="thinker",
    ).strip()

    tone = _infer_tone(user_text, base)
    quote = _pick_quote_for_tone(tone) or ""

    # pass 2: rewrite
    rewrite_prompt = (
        "You are rewriting an answer in a pop-culture character voice.\n"
        "You are given a SEED_QUOTE which anchors tone/voice.\n\n"
        "Rules:\n"
        "- Style may bend grammar, tone, and voice, but never semantics.\n"
        "- Attitudinal worldview may be emulated, but epistemic claims may not be altered.\n"
        "- Do NOT add new facts. Do NOT remove key facts.\n"
        "- Output ONLY the rewritten answer (no preamble, no analysis).\n\n"
        f"SEED_QUOTE: {quote}\n\n"
        f"ORIGINAL_ANSWER:\n{base}\n\n"
        "REWRITE:" 
    )

    rewritten = call_model_prompt(role="thinker", prompt=rewrite_prompt, max_tokens=420, temperature=0.85, top_p=0.95).strip()

    if not rewritten:
        rewritten = base

    # required visible tag
    qline = f'"{quote}"' if quote else '""'
    return f"[FUN REWRITE] {qline}\n\n{rewritten}"


_QUOTE_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_QUOTE_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "should", "would", "from", "about", "tell", "me", "more", "all", "any",
}


def _quote_query_tokens(text: str) -> set[str]:
    toks = {m.group(0).lower() for m in _QUOTE_TOKEN_RE.finditer(text or "")}
    return {t for t in toks if len(t) > 2 and t not in _QUOTE_STOPWORDS}


def _scratchpad_quote_lines(
    sp_block: str,
    *,
    query: str = "",
    max_quotes: int = 2,
    max_len: int = 160,
) -> List[str]:
    """Extract deterministic short quote lines from scratchpad facts block."""
    out: List[str] = []
    q_tokens = _quote_query_tokens(query)
    for ln in (sp_block or "").splitlines():
        s = (ln or "").strip()
        if not s.startswith("- [scratchpad "):
            continue
        if "] " not in s:
            continue
        snippet = s.split("] ", 1)[1].strip()
        if not snippet:
            continue
        if q_tokens:
            s_tokens = _quote_query_tokens(snippet)
            if not (q_tokens & s_tokens):
                continue
        if len(snippet) > max_len:
            snippet = snippet[: max_len - 1].rstrip() + "..."
        out.append(snippet)
        if len(out) >= max_quotes:
            break
    return out


def _sanitize_scratchpad_grounded_output(text: str) -> str:
    """Remove non-scratchpad provenance markers from grounded outputs."""
    if not text:
        return ""
    cleaned: List[str] = []
    for ln in text.splitlines():
        low = ln.lower()
        if "source: docs" in low:
            continue
        if "source: model" in low:
            continue
        if "source: facts" in low:
            # Normalize fact-grounded confidence lines to scratchpad provenance.
            ln = ln.replace("Source: Facts", "Source: Scratchpad")
            ln = ln.replace("source: facts", "source: scratchpad")
        if "source: contextual" in low:
            ln = ln.replace("Source: Contextual", "Source: Scratchpad")
            ln = ln.replace("source: contextual", "source: scratchpad")
        if "source: mixed" in low:
            ln = ln.replace("Source: Mixed", "Source: Scratchpad")
            ln = ln.replace("source: mixed", "source: scratchpad")
        cleaned.append(ln)

    # Collapse runs of 3+ blank lines to 2 to keep formatting tidy.
    out: List[str] = []
    blank_run = 0
    for ln in cleaned:
        if not ln.strip():
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        out.append(ln)
    return "\n".join(out).strip()


def _append_scratchpad_provenance(text: str) -> str:
    """Append one canonical provenance line only if no source line exists."""
    t = (text or "").strip()
    if not t:
        return "Source: Scratchpad"
    for ln in t.splitlines():
        if "source:" in ln.lower():
            return t
    return t + "\n\nSource: Scratchpad"


def _rewrite_source_line(text: str, source_line: str) -> str:
    """Replace first Source: line; append if missing."""
    t = (text or "").strip()
    if not t:
        return source_line
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        if "source:" in ln.lower():
            lines[i] = source_line
            return "\n".join(lines).strip()
    return t.rstrip() + "\n\n" + source_line


def _lock_constraints_block(locked_file: str) -> str:
    lf = (locked_file or "SUMM_locked.md").strip()
    return (
        "Grounding mode: LOCKED_SUMM_FILE.\n"
        f"- Locked file: {lf}\n"
        "- Prefer facts from the locked file for answers.\n"
        f"- If the answer is not present in {lf}, you may answer from pre-trained knowledge.\n"
        f"- In that fallback case, include this exact note line first: "
        f"[Not found in locked source {lf}. Answer based on pre-trained data.]\n"
        "- In that fallback case, final source line must be exactly: Source: Model (not in locked file)"
    )


def _apply_locked_output_policy(text: str, state: SessionState) -> str:
    """Normalize provenance for lock mode while preserving Mentats isolation."""
    if not state.locked_summ_path:
        return text
    locked_file = state.locked_summ_file or os.path.basename(state.locked_summ_path) or "SUMM_locked.md"
    t = (text or "").strip()
    if not t:
        return _rewrite_source_line("", f"Source: Locked file ({locked_file})")

    low = t.lower()
    if "source: model" in low:
        note = f"[Not found in locked source {locked_file}. Answer based on pre-trained data.]"
        if note.lower() not in low:
            t = note + "\n\n" + t
        return _rewrite_source_line(t, "Source: Model (not in locked file)")

    return _rewrite_source_line(t, f"Source: Locked file ({locked_file})")


def _soft_alias_command(text: str, state: SessionState) -> Optional[str]:
    """Optional bare-text aliases for scratchpad ergonomics.

    Guardrails:
    - `status` alias is always available.
    - Scratchpad aliases are active only when scratchpad is attached.
    - Exact patterns only.
    - Never touches explicit command prefixes (>>/Â»/??/!!/##).
    """
    t = (text or "").strip()
    if not t:
        return None
    if _is_command(t):
        return None
    if t.startswith("??") or t.startswith("!!") or t.startswith("##"):
        return None
    # Global exact alias (strict match only).
    if t.lower() == "status":
        return ">>status"

    attached = set(getattr(state, "attached_kbs", set()) or set())
    fs_attached = {k for k in attached if k in KB_PATHS and k != VAULT_KB_NAME}

    # Lock aliases (guarded: only when a filesystem KB is attached or a lock is active).
    if (fs_attached or getattr(state, "locked_summ_path", "")):
        if low := t.lower():
            if low.startswith("lock summ_") and low.endswith(".md"):
                return ">>" + t
            if low == "unlock":
                return ">>unlock"
            if low == "list files" and fs_attached:
                return ">>list_files"

    if "scratchpad" not in attached:
        return None
    low = t.lower()
    if low.startswith("scratchpad show "):
        q = t[len("scratchpad show ") :].strip()
        if q:
            return f">>scratchpad show {q}"
    if low.startswith("scratch show "):
        q = t[len("scratch show ") :].strip()
        if q:
            return f">>scratchpad show {q}"
    if low in ("list", "list scratchpad"):
        return ">>scratchpad list"
    if low.startswith("delete "):
        q = t[7:].strip()
        if q:
            return f">>scratchpad delete {q}"
    return None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "1.2.0-refactored"}


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
    # Prefer explicit header (OWUI can set it)
    sid = req.headers.get("x-session-id") or req.headers.get("x-chat-id")
    if sid:
        return sid.strip()

    # Fallback to OpenAI user field
    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()

    # Last resort: client addr
    try:
        host = req.client.host if req.client else "local"
    except Exception:
        host = "local"
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
        if "cliniko" in t_l:
            return "Cliniko Pipeline"
        if "router" in t_l or "fastapi" in t_l:
            return "Router Debugging"
        return "Chat Summary"

    if _is_openwebui_title_task(user_text_raw):
        CLINIKO_DEBUG = cfg_get("cliniko.debug", True)
        if CLINIKO_DEBUG:
            print("[DEBUG] openwebui title task bypass", flush=True)
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
        text = call_model_messages(role="vision", messages=raw_messages, max_tokens=700, temperature=0.2, top_p=0.9)
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Check for per-turn selectors (##) FIRST
    selector, user_text = _split_selector(user_text_raw)
    
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
    
    # Only treat as session command if NOT a per-turn selector
    if selector == "":
        try:
            cmd_reply = handle_command(user_text_raw, state=state, session_id=session_id)
        except Exception as e:
            text = f"[router error: command handler crashed: {e.__class__.__name__}: {e}]"
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))

        if cmd_reply is not None:
            text = cmd_reply
            if maybe_capture_command_output is not None:
                try:
                    maybe_capture_command_output(
                        session_id=session_id,
                        state=state,
                        cmd_text=user_text_raw,
                        reply_text=text,
                    )
                except Exception:
                    pass
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))

        # Soft alias fallback (only after explicit command parsing returns None).
        alias_cmd = _soft_alias_command(user_text_raw, state)
        if alias_cmd:
            try:
                alias_reply = handle_command(alias_cmd, state=state, session_id=session_id)
            except Exception as e:
                text = f"[router error: soft alias crashed: {e.__class__.__name__}: {e}]"
                if stream:
                    return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                return JSONResponse(_make_openai_response(text))
            if alias_reply is not None:
                text = alias_reply
                if maybe_capture_command_output is not None:
                    try:
                        maybe_capture_command_output(
                            session_id=session_id,
                            state=state,
                            cmd_text=alias_cmd,
                            reply_text=text,
                        )
                    except Exception:
                        pass
                if stream:
                    return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                return JSONResponse(_make_openai_response(text))
    
    print(f"[DEBUG] selector={selector!r}, user_text={user_text!r}", flush=True)

    # Build history for non-vision pipelines
    
    # Vodka instance: create BEFORE using it
    if state.vodka is None:
        state.vodka = VodkaFilter()
    vodka = state.vodka
    
    # Apply Vodka config
    vodka_cfg = dict(cfg_get("vodka", {}))
    try:
        vodka.valves.storage_dir = str(vodka_cfg.get("storage_dir", vodka.valves.storage_dir) or vodka.valves.storage_dir)
        vodka.valves.base_ttl_days = int(vodka_cfg.get("base_ttl_days", vodka.valves.base_ttl_days))
        vodka.valves.touch_extension_days = int(vodka_cfg.get("touch_extension_days", vodka.valves.touch_extension_days))
        user_max_touches = int(vodka_cfg.get("max_touches", vodka.valves.max_touches))
        vodka.valves.max_touches = min(max(0, user_max_touches), 3)
        vodka.valves.debug = bool(vodka_cfg.get("debug", vodka.valves.debug))
        vodka.valves.debug_dir = str(vodka_cfg.get("debug_dir", vodka.valves.debug_dir) or vodka.valves.debug_dir)
        vodka.valves.n_last_messages = int(vodka_cfg.get("n_last_messages", vodka.valves.n_last_messages))
        vodka.valves.keep_first = bool(vodka_cfg.get("keep_first", vodka.valves.keep_first))
        vodka.valves.max_chars = int(vodka_cfg.get("max_chars", vodka.valves.max_chars))
    except Exception:
        pass
    
    # NOW apply Vodka filtering (CTC, FR, !!, ??)
    vodka_body = {"messages": raw_messages}
    try:
        vodka_body = vodka.inlet(vodka_body)
        vodka_body = vodka.outlet(vodka_body)
        raw_messages = vodka_body.get("messages", raw_messages)
    except Exception as e:
        pass  # Fail-open
    
    # Check if Vodka already answered hard commands (?? list, !! nuke)
    if raw_messages and raw_messages[-1].get("role") == "assistant":
        last_msg_content = raw_messages[-1].get("content", "")
        if isinstance(last_msg_content, str) and (
            "[vodka]" in last_msg_content.lower() or 
            "[vodka memory store]" in last_msg_content.lower()
        ):
            # This is a Vodka hard command answer - return it directly
            if stream:
                return StreamingResponse(_stream_sse(last_msg_content), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(last_msg_content))
    
    history_text_only = normalize_history(raw_messages)
    
    # Vision / OCR selectors (explicit ##vision or ##ocr)
    if selector in ("vision", "ocr"):
        role = "vision"
        text = call_model_messages(role=role, messages=raw_messages, max_tokens=700, temperature=0.2, top_p=0.9)
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Mentats selector
    if selector == "mentats":
        print(f"[DEBUG] Mentats selector triggered, user_text={user_text!r}", flush=True)
        
        # Mentats MUST be isolated: drop fun modes
        state.fun_sticky = False
        state.fun_rewrite_sticky = False

        if not run_mentats:
            text = "[router] mentats.py not available"
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))

        vault_facts = build_vault_facts(user_text, state)
        if not (vault_facts or "").strip():
            text = (
                "[ZARDOZ HATH SPOKEN]\n\n"
                "The Vault contains no relevant knowledge for this query. I cannot reason without authoritative facts.\n\n"
                "Sources: Vault (empty)"
            )
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))

        # run mentats
        def _build_rag_block(query: str, collection: str = "vault") -> str:
            return build_vault_facts(query, state)

        def _build_constraints_block(query: str) -> str:
            return ""

        # Wrapper to enforce temperature=0.1 for critic role
        def _call_model_with_critic_temp(*, role: str, prompt: str, max_tokens: int = 256, temperature: float = 0.3, top_p: float = 0.9) -> str:
            if role == "critic":
                temperature = 0.1
            return call_model_prompt(role=role, prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        # No-op Vodka for Mentats
        class _NoOpVodka:
            def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
                return body
            def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
                return body

        print(f"[DEBUG] About to call run_mentats with vault_facts length={len(vault_facts)}", flush=True)
        try:
            text = run_mentats(
                session_id,
                user_text,
                [],
                vodka=_NoOpVodka(),
                call_model=_call_model_with_critic_temp,
                build_rag_block=_build_rag_block,
                build_constraints_block=_build_constraints_block,
                facts_collection=VAULT_KB_NAME,
                thinker_role="thinker",
                critic_role="critic",
            ).strip()
            print(f"[DEBUG] run_mentats returned {len(text)} chars", flush=True)
        except Exception as e:
            print(f"[DEBUG] run_mentats CRASHED: {e}", flush=True)
            text = f"[DEBUG] Mentats crashed: {e}"

        if "[ZARDOZ HATH SPOKEN]" not in text:
            text = text.rstrip() + "\n\n[ZARDOZ HATH SPOKEN]"
        if "Sources:" not in text:
            text = text.rstrip() + "\nSources: Vault"

        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # FUN selector
    if selector == "fun":
        state.fun_sticky = False
        state.fun_rewrite_sticky = False
        fun_mode = "fun"
    else:
        fun_mode = ""

    # Apply sticky fun modes for default pipeline
    if not fun_mode and (state.fun_rewrite_sticky or state.fun_sticky):
        if has_mentats_in_recent_history(history_text_only, last_n=5):
            # Disable Fun/FR and warn user
            state.fun_sticky = False
            state.fun_rewrite_sticky = False
            text = "[router] Fun/FR auto-disabled: Mentats output in recent history. Start new topic or re-enable with >>f"
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
    
    if not fun_mode and state.fun_rewrite_sticky:
        fun_mode = "fun_rewrite"
    elif not fun_mode and state.fun_sticky:
        fun_mode = "fun"

    # Default: serious reasoning
    facts_block = build_fs_facts(user_text, state)
    lock_active = bool(state.locked_summ_path)
    scratchpad_quotes: List[str] = []
    scratchpad_grounded = False
    constraints_block = ""
    if lock_active:
        constraints_block = _lock_constraints_block(state.locked_summ_file)
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
                scratchpad_quotes = _scratchpad_quote_lines(sp_block, query=user_text)
                facts_block = f"{facts_block}\n\n{sp_block}".strip() if facts_block else sp_block
                constraints_block = (
                    "Grounding mode: SCRATCHPAD_ONLY.\n"
                    "- Use only FACTS provided in this turn (scratchpad facts).\n"
                    "- If FACTS are insufficient, say so explicitly.\n"
                    "- Do not use pretrained/background knowledge.\n"
                    "- Keep claims constrained to provided facts."
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
            text = _append_scratchpad_provenance(text)
            if stream:
                return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
            return JSONResponse(_make_openai_response(text))
        except Exception:
            pass

    if fun_mode == "fun":
        if run_fun is None:
            # fallback: do serious then rewrite in-router
            base = run_serious(
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                vodka=vodka,
                call_model=call_model_prompt,
                facts_block=facts_block,
                constraints_block=constraints_block,
                thinker_role="thinker",
            ).strip()
            tone = _infer_tone(user_text, base)
            quote = _pick_quote_for_tone(tone)
            out = f'[FUN] "{quote}"\n\n{base}' if quote else f"[FUN]\n\n{base}"
            text = out
        else:
            # Build a tone-matched pool and let fun.py randomize within it
            base_preview = run_serious(
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                vodka=vodka,
                call_model=call_model_prompt,
                facts_block=facts_block,
                constraints_block=constraints_block,
                thinker_role="thinker",
            ).strip()
            tone = _infer_tone(user_text, base_preview)
            qb = _quotes_by_tag()
            pool = [q for quotes_list in qb.values() for q in quotes_list] if qb else []

            styled = run_fun(
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                facts_block=facts_block,
                quote_pool=pool,
                vodka=vodka,
                call_model=call_model_prompt,
                thinker_role="thinker",
            ).strip()

            # Ensure required top line is explicit and quoted
            lines = styled.splitlines()
            if lines:
                q = lines[0].strip()
                if q and not (q.startswith('"') and q.endswith('"')):
                    q_clean = q.strip('"')
                    q = '"' + q_clean + '"'
                lines[0] = f"[FUN] {q}" if q else "[FUN]"
                text = "\n".join(lines)
            else:
                text = "[FUN]"

        if scratchpad_grounded:
            text = _sanitize_scratchpad_grounded_output(text)
            if scratchpad_quotes:
                text = (
                    text.rstrip()
                    + "\n\nScratchpad Quotes:\n"
                    + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
                )
            text = _append_scratchpad_provenance(text)

        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text and not scratchpad_grounded and not lock_active:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        if lock_active:
            text = _apply_locked_output_policy(text, state)
        
        # Auto-detach if this was a trust >>attach all operation
        if state.auto_detach_after_response:
            state.attached_kbs.clear()
            state.auto_detach_after_response = False

        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    if fun_mode == "fun_rewrite":
        text = run_fun_rewrite_fallback(
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=vodka,
            facts_block=facts_block,
        ).strip()

        if scratchpad_grounded:
            text = _sanitize_scratchpad_grounded_output(text)
            if scratchpad_quotes:
                text = (
                    text.rstrip()
                    + "\n\nScratchpad Quotes:\n"
                    + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
                )
            text = _append_scratchpad_provenance(text)
        
        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text and not scratchpad_grounded and not lock_active:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        if lock_active:
            text = _apply_locked_output_policy(text, state)
        
        # Auto-detach if this was a trust >>attach all operation
        if state.auto_detach_after_response:
            state.attached_kbs.clear()
            state.auto_detach_after_response = False
        
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # RAW mode (bypass Serious formatting, keep CTC + KB grounding)
    if state.raw_sticky and run_raw:
        text = run_raw(
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=vodka,
            call_model=call_model_prompt,
            facts_block=facts_block,
            constraints_block=constraints_block,
            thinker_role="thinker",
        ).strip()

        if scratchpad_grounded:
            text = _sanitize_scratchpad_grounded_output(text)
            if scratchpad_quotes:
                text = (
                    text.rstrip()
                    + "\n\nScratchpad Quotes:\n"
                    + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
                )
            text = _append_scratchpad_provenance(text)
        
        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text and not scratchpad_grounded and not lock_active:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        if lock_active:
            text = _apply_locked_output_policy(text, state)
        
        # Auto-detach if this was a trust >>attach all operation
        if state.auto_detach_after_response:
            state.attached_kbs.clear()
            state.auto_detach_after_response = False
        
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Normal serious
    text = run_serious(
        session_id=session_id,
        user_text=user_text,
        history=history_text_only,
        vodka=vodka,
        call_model=call_model_prompt,
        facts_block=facts_block,
        constraints_block=constraints_block,
        thinker_role="thinker",
    ).strip()

    if scratchpad_grounded:
        text = _sanitize_scratchpad_grounded_output(text)
        if scratchpad_quotes:
            text = (
                text.rstrip()
                + "\n\nScratchpad Quotes:\n"
                + "\n".join(f'- "{q}"' for q in scratchpad_quotes)
            )
        text = _append_scratchpad_provenance(text)

    # Add disclaimer if KBs were attached but model used training data
    if state.attached_kbs and "Source: Model" in text and not scratchpad_grounded and not lock_active:
        kb_list = ', '.join(sorted(state.attached_kbs))
        disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
        text = disclaimer + text
    if lock_active:
        text = _apply_locked_output_policy(text, state)
    
    # Auto-detach AFTER disclaimer check
    if state.auto_detach_after_response:
        state.attached_kbs.clear()
        state.auto_detach_after_response = False

    if stream:
        return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
    return JSONResponse(_make_openai_response(text))


# Convenience for `python router_fastapi.py`
if __name__ == "__main__":
    import uvicorn

    host = str(cfg_get("server.host", "0.0.0.0"))
    port = int(cfg_get("server.port", 9000))
    uvicorn.run("router_fastapi:app", host=host, port=port, reload=False)

