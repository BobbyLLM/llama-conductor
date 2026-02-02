# router_fastapi.py
# version 1.1.5
"""MoA Router (FastAPI) 

CHANGES IN v1.1.5:
- Added >>trust mode - tool recommendation sidecar (invariants-compliant)
- >>trust analyzes queries and suggests appropriate tools (calc, mentats, wiki, etc.)
- User chooses recommendation (A, B, C) explicitly - no auto-execution
- Preserves all invariants: explicit control, router stays dumb, no auto-escalation

CHANGES IN v1.1.4 (CRITICAL FIX):
- SECURITY FIX: >>attach vault now properly rejected with helpful error
- Previously allowed attaching "vault" (Qdrant) as filesystem KB, causing silent failures
- Serious mode would filter out vault, resulting in empty FACTS_BLOCK and hallucinations
- Now vault can only be accessed via ##mentats (as designed)

CHANGES IN v1.1.3:
- Removed legacy code for fun.py and reduced duplicate calls

CHANGES IN v1.1.2:
- Added keepalive code to prevent streaming time outs

CHANGES IN v1.1.1:
- Added >>wiki <topic> sidecar (Wikipedia summary via free JSON API)
- Added >>exchange <query> sidecar (Currency conversion via Frankfurter API)
- Added >>weather <location> sidecar (Current weather via wttr.in)
- All three are context-light, no-auth deterministic tools

CHANGES IN v1.0.9-debug:
- Fix: Fun/FR blocks after Mentats (checks only recent 5 turns, not all history)
- Fix: Images drop Fun/FR sticky modes and auto-route to vision
- Fix: Mentats selector actually works (was being skipped)
- Fix: Clear errors when conflicts occur


CHANGES IN v1.0.4:
- Auto-vision detection: Images automatically trigger vision pipeline
- Session commands (>>) skip vision even if image is present (prevents UI attachment bugs)
- All other behavior unchanged

This file is the orchestration layer:
- OpenAI-compatible /v1/chat/completions (optionally streaming)
- Session commands (>>...)
- Per-turn selectors (##...)

Key behaviors:
- KBs are filesystem folders containing SUMM_*.md files.
- /serious answers use filesystem KB retrieval ONLY from attached KBs.
- `>>summ` creates SUMM_*.md from new raw files (non-SUMM) and moves originals to /original.
- `>>move to vault` promotes SUMM_*.md into Qdrant under kb="vault" (or configured vault_kb_name).
- `##mentats` runs 3-pass Thinker→Critic→Thinker against Vault (Qdrant) only.
- Fun / Fun Rewrite are post-reasoning style transforms and never run on Mentats.

Notes:
- PDFs require `pypdf` installed (user responsibility). If missing, PDFs are skipped with a clear note.
- This router is designed to be robust: it should start even if some optional deps are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

# Local modules (must exist)
from .vodka_filter import Filter as VodkaFilter
from .serious import run_serious

# Optional local modules (feature-detected)
try:
    from .fun import run_fun  # type: ignore
except Exception:
    run_fun = None  # type: ignore

try:
    from .mentats import run_mentats  # type: ignore
except Exception:
    run_mentats = None  # type: ignore

try:
    from .fs_rag import build_fs_facts_block, search_fs  # type: ignore
except Exception:
    build_fs_facts_block = None  # type: ignore
    search_fs = None  # type: ignore

try:
    from .sidecars import (  # type: ignore
        parse_and_eval_calc,
        format_calc_result,
        list_vodka_memories,
        format_memory_list,
        find_quote_in_kbs,
        format_quote_result,
        flush_ctc_cache,
        handle_wiki_query,
        handle_exchange_query,
        handle_weather_query,
    )
except Exception:
    parse_and_eval_calc = None  # type: ignore
    format_calc_result = None  # type: ignore
    list_vodka_memories = None  # type: ignore
    format_memory_list = None  # type: ignore
    find_quote_in_kbs = None  # type: ignore
    format_quote_result = None  # type: ignore
    flush_ctc_cache = None  # type: ignore
    handle_wiki_query = None  # type: ignore
    handle_exchange_query = None  # type: ignore
    handle_weather_query = None  # type: ignore

try:
    from .pipelines import run_raw  # type: ignore
except Exception:
    run_raw = None  # type: ignore

try:
    from .trust_pipeline import (  # type: ignore
        handle_trust_command,
        generate_recommendations,
    )
except Exception:
    handle_trust_command = None  # type: ignore
    generate_recommendations = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "router_config.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG: Dict[str, Any] = {}


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    global CFG
    try:
        CFG = _load_yaml(path)
    except Exception as e:
        CFG = {}
        print(f"[router] failed to load config '{path}': {e}")
    return CFG


def _cfg_get(path: str, default: Any) -> Any:
    cur: Any = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


load_config(DEFAULT_CONFIG_PATH)

LLAMA_SWAP_URL: str = str(_cfg_get("llama_swap_url", "http://127.0.0.1:8011/v1/chat/completions"))
ROLES: Dict[str, str] = dict(_cfg_get("roles", {}))
KB_PATHS: Dict[str, str] = dict(_cfg_get("kb_paths", {}))
VAULT_KB_NAME: str = str(_cfg_get("vault_kb_name", "vault"))

# Files for SUMM + quotes
SUMM_PROMPT_PATH = os.path.join(HERE, "SUMM.md")
QUOTES_MD_PATH = os.path.join(HERE, "quotes.md")
CHEAT_SHEET_PATH = os.path.join(HERE, "command_cheat_sheet.md")

# fs-rag defaults
FS_TOP_K = int(_cfg_get("fs_rag.top_k", 8))
FS_MAX_CHARS = int(_cfg_get("fs_rag.max_chars", 2400))

# vault promotion defaults (chunking)
VAULT_CHUNK_WORDS = int(_cfg_get("vault.chunk_words", 600))
VAULT_OVERLAP_WORDS = int(_cfg_get("vault.chunk_overlap_words", 175))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    attached_kbs: Set[str] = field(default_factory=set)
    fun_sticky: bool = False
    fun_rewrite_sticky: bool = False
    raw_sticky: bool = False

    rag_last_query: str = ""
    rag_last_hits: int = 0

    vault_last_query: str = ""
    vault_last_hits: int = 0

    # One Vodka per session (reduces noise + preserves stateful debug counters)
    vodka: Optional[VodkaFilter] = None

    # Trust mode: pending recommendations for A/B/C response
    pending_trust_query: str = ""
    pending_trust_recommendations: List[Dict[str, str]] = field(default_factory=list)
    
    # Auto-execute query after >>attach all from trust
    auto_query_after_attach: str = ""
    auto_detach_after_response: bool = False


_SESSIONS: Dict[str, SessionState] = {}


def get_state(session_id: str) -> SessionState:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
    return _SESSIONS[session_id]


# ---------------------------------------------------------------------------
# Helpers: OpenAI-ish parsing
# ---------------------------------------------------------------------------

_IMAGE_BLOCK_TYPES = {"image_url", "input_image"}


def _extract_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("text", "input_text"):
            t = b.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts).strip()


def _has_image_blocks(blocks: List[Dict[str, Any]]) -> bool:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") in _IMAGE_BLOCK_TYPES:
            return True
    return False


def has_images_in_messages(messages: List[Dict[str, Any]]) -> bool:
    """Check if any message in the list contains images."""
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, list) and _has_image_blocks(c):
            return True
    return False



def has_mentats_in_recent_history(messages: List[Dict[str, Any]], last_n: int = 5) -> bool:
    """Check if any of the last N messages contains Mentats output markers."""
    recent = messages[-last_n:] if len(messages) > last_n else messages
    for m in recent:
        c = m.get("content", "")
        if isinstance(c, str) and ("[ZARDOZ HATH SPOKEN]" in c or "Sources: Vault" in c):
            return True
    return False


def normalize_history(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert possibly-multimodal content into text-only messages for non-vision pipelines."""
    out: List[Dict[str, Any]] = []
    for m in raw_messages or []:
        role = m.get("role", "")
        c = m.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = _extract_text_from_blocks(c)
            if _has_image_blocks(c):
                text = (text + "\n[image]") if text else "[image]"
        else:
            text = ""
        out.append({"role": role, "content": text})
    return out


def last_user_message(messages: List[Dict[str, Any]]) -> Tuple[str, int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            c = messages[i].get("content", "")
            if isinstance(c, str):
                return c, i
            if isinstance(c, list):
                return _extract_text_from_blocks(c), i
            return "", i
    return "", -1


# ---------------------------------------------------------------------------
# llama-swap calls
# ---------------------------------------------------------------------------


def _resolve_model(role: str) -> str:
    return str(ROLES.get(role, "")).strip()


def call_model_prompt(
    *,
    role: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    model_name = _resolve_model(role)
    if not model_name:
        return f"[router error: no model configured for role '{role}']"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        resp = requests.post(LLAMA_SWAP_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json() or {}
        choices = data.get("choices", []) or []
        if not choices:
            return "[router error: no choices from model]"
        msg = choices[0].get("message", {}) or {}
        return str(msg.get("content", "") or "").strip()
    except Exception as e:
        return f"[model '{model_name}' unavailable: {e}]"


def call_model_messages(
    *,
    role: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    model_name = _resolve_model(role)
    if not model_name:
        return f"[router error: no model configured for role '{role}']"

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    try:
        resp = requests.post(LLAMA_SWAP_URL, json=payload, timeout=240)
        resp.raise_for_status()
        data = resp.json() or {}
        choices = data.get("choices", []) or []
        if not choices:
            return "[router error: no choices from model]"
        msg = choices[0].get("message", {}) or {}
        return str(msg.get("content", "") or "").strip()
    except Exception as e:
        return f"[model '{model_name}' unavailable: {e}]"


# ---------------------------------------------------------------------------
# Quotes (file-based)
# ---------------------------------------------------------------------------


def _load_quotes_md(path: str) -> Dict[str, List[str]]:
    """Parse quotes.md into {tag: [quotes...]}. Supports headings like: ## futurama snark sarcastic"""
    if not os.path.isfile(path):
        return {}

    tag_to_quotes: Dict[str, List[str]] = {}
    current_tags: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith("##"):
                tags = line.strip().lstrip("#").strip()
                current_tags = [t.strip().lower() for t in tags.split() if t.strip()]
                continue

            q = line.strip()
            if not q:
                continue

            # ignore markdown bullet markers
            if q.startswith("-"):
                q = q.lstrip("-").strip()
            if not q:
                continue

            for t in current_tags or ["default"]:
                tag_to_quotes.setdefault(t, []).append(q)

    return tag_to_quotes


_QUOTES_CACHE: Optional[Dict[str, List[str]]] = None


def _quotes_by_tag() -> Dict[str, List[str]]:
    global _QUOTES_CACHE
    if _QUOTES_CACHE is None:
        _QUOTES_CACHE = _load_quotes_md(QUOTES_MD_PATH)
        # Debug: log if quotes failed to load
        if not _QUOTES_CACHE:
            print(f"[router WARNING] quotes.md not found or empty at {QUOTES_MD_PATH}")
    return _QUOTES_CACHE


def _pick_quote_for_tone(tone: str) -> str:
    import random

    tone = (tone or "").strip().lower()
    qb = _quotes_by_tag()

    pool = qb.get(tone) or qb.get("default") or []
    if not pool:
        return ""

    return random.choice(pool)


def _infer_tone(user_text: str, answer_text: str) -> str:
    """Ask the model for a single tone tag that exists in quotes.md."""
    tags = sorted(set(_quotes_by_tag().keys()))
    if not tags:
        return "default"

    prompt = (
        "You are selecting a single tone tag for a pop-culture quote seed.\n"
        "Return EXACTLY ONE tag from the allowed list. No extra text.\n\n"
        f"ALLOWED_TAGS: {', '.join(tags[:120])}\n\n"
        f"USER: {user_text.strip()}\n\n"
        f"ANSWER: {answer_text.strip()}\n\n"
        "TAG:"
    )

    raw = call_model_prompt(role="thinker", prompt=prompt, max_tokens=10, temperature=0.1, top_p=0.9)
    tag = (raw or "").strip().lower().split()[0] if raw else ""
    return tag if tag in set(tags) else "default"


# ---------------------------------------------------------------------------
# Filesystem SUMM pipeline
# ---------------------------------------------------------------------------

_SUPPORTED_RAW_EXTS = {".md", ".txt", ".pdf", ".html", ".htm"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pypdf not installed ({e})")

    reader = PdfReader(path)
    pieces: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pieces.append(t.strip())
    return "\n\n".join(pieces).strip()


def _read_html_text(path: str) -> str:
    # very cheap HTML strip (good enough for SUMM stage)
    raw = _read_text_file(path)
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_raw_to_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext in (".html", ".htm"):
        return _read_html_text(path)
    return _read_text_file(path)


def _load_summ_prompt() -> str:
    with open(SUMM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _summ_one(src_path: str) -> str:
    """Return SUMM markdown content for src_path."""
    prompt = _load_summ_prompt()
    body = _read_raw_to_text(src_path)

    # Guard: avoid sending insane payloads
    max_chars = int(_cfg_get("summ.max_input_chars", 120_000))
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars] + "\n\n...[truncated]...\n"

    full_prompt = (
        f"{prompt.rstrip()}\n\n"
        "---\n"
        "INPUT_TEXT:\n"
        f"{body}\n"
        "---\n"
    )

    return call_model_prompt(
        role="thinker",
        prompt=full_prompt,
        max_tokens=int(_cfg_get("summ.max_tokens", 900)),
        temperature=float(_cfg_get("summ.temperature", 0.2)),
        top_p=float(_cfg_get("summ.top_p", 0.9)),
    )


def summ_new_in_kb(kb_name: str, folder: str) -> Dict[str, Any]:
    """Create SUMM_*.md for each supported doc in kb folder, moving originals to /original/."""
    kb = (kb_name or "").strip()
    if not kb:
        return {"summ_created": 0, "summ_skipped": 0, "notes": ["kb name missing"]}

    if not folder or not os.path.isdir(folder):
        return {"summ_created": 0, "summ_skipped": 0, "notes": [f"folder missing: {folder}"]}

    notes: List[str] = []
    created = 0
    skipped = 0

    original_dir = os.path.join(folder, "original")
    os.makedirs(original_dir, exist_ok=True)

    for root, _, files in os.walk(folder):
        # skip /original
        if os.path.abspath(root).lower().startswith(os.path.abspath(original_dir).lower()):
            continue

        for fn in files:
            path = os.path.join(root, fn)
            if not os.path.isfile(path):
                continue

            # Only summarize direct raw docs
            if fn.startswith("SUMM_") and fn.lower().endswith(".md"):
                continue

            ext = os.path.splitext(fn)[1].lower()
            if ext not in _SUPPORTED_RAW_EXTS:
                continue

            base = os.path.splitext(fn)[0]
            summ_name = f"SUMM_{base}.md"
            summ_path = os.path.join(root, summ_name)

            if os.path.exists(summ_path):
                skipped += 1
                continue

            try:
                sha = _sha256_file(path)
            except Exception as e:
                notes.append(f"sha fail: {fn}: {e}")
                skipped += 1
                continue

            try:
                summ_md = _summ_one(path)
            except Exception as e:
                notes.append(f"SUMM failed: {fn}: {e}")
                skipped += 1
                continue

            header = (
                "<!--\n"
                f"source_file: {fn}\n"
                f"source_rel_path: {os.path.relpath(path, folder)}\n"
                f"source_sha256: {sha}\n"
                f"summ_created_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                "pipeline: SUMM\n"
                "-->\n\n"
            )

            try:
                with open(summ_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write((summ_md or "").strip() + "\n")
            except Exception as e:
                notes.append(f"write fail: {summ_name}: {e}")
                skipped += 1
                continue

            # move original to /original
            try:
                dest = os.path.join(original_dir, fn)
                # avoid overwrite
                if os.path.exists(dest):
                    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                    dest = os.path.join(original_dir, f"{base}_{ts}{ext}")
                shutil.move(path, dest)
            except Exception as e:
                notes.append(f"move original fail: {fn}: {e}")

            created += 1

    return {"summ_created": created, "summ_skipped": skipped, "notes": notes}


# ---------------------------------------------------------------------------
# Vault promotion (Qdrant)
# ---------------------------------------------------------------------------


def _word_chunks(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []

    chunk_words = max(50, int(chunk_words or 600))
    overlap_words = max(0, int(overlap_words or 175))
    if overlap_words >= chunk_words:
        overlap_words = max(0, chunk_words // 4)

    out: List[str] = []
    i = 0
    step = max(1, chunk_words - overlap_words)
    while i < len(words):
        out.append(" ".join(words[i : i + chunk_words]))
        i += step
    return out


def _ensure_qdrant_ready() -> Tuple[Any, Any, str, str]:
    """Return (client, embed_model, collection, vector_name)."""
    from rag import (
        QDRANT_COLLECTION,
        VECTOR_NAME,
        ensure_collection,
        get_embed_model,
        get_qdrant_client,
    )

    ensure_collection()
    return get_qdrant_client(), get_embed_model(), QDRANT_COLLECTION, VECTOR_NAME


def _qdrant_delete_for_vault_file(client: Any, *, vault_kb: str, source_kb: str, rel_path: str) -> int:
    """Best-effort delete of existing vault points for a given source_kb+rel_path."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        flt = Filter(
            must=[
                FieldCondition(key="kb", match=MatchValue(value=vault_kb)),
                FieldCondition(key="source_kb", match=MatchValue(value=source_kb)),
                FieldCondition(key="source_rel_path", match=MatchValue(value=rel_path)),
            ]
        )
        res = client.delete(
            collection_name=_cfg_get("rag.collection", None) or "moa_kb_docs",
            points_selector=flt,
            wait=True,
        )
        # qdrant returns status only; we can't know count reliably
        return 0
    except Exception:
        return 0


def move_summ_to_vault(source_kbs: Set[str]) -> Dict[str, Any]:
    """Promote SUMM_*.md from source_kbs into Qdrant under kb=VAULT_KB_NAME."""
    source_kbs = {k.strip() for k in (source_kbs or set()) if k and k.strip()}
    if not source_kbs:
        return {"files": 0, "chunks": 0, "notes": ["no source KBs provided"]}

    # Ensure SUMM.md exists (so users have it next to router)
    if not os.path.isfile(SUMM_PROMPT_PATH):
        return {"files": 0, "chunks": 0, "notes": [f"SUMM.md missing at {SUMM_PROMPT_PATH}"]}

    notes: List[str] = []
    files_done = 0
    chunks_done = 0

    try:
        client, embed_model, collection, vector_name = _ensure_qdrant_ready()
    except Exception as e:
        return {"files": 0, "chunks": 0, "notes": [f"Qdrant not ready: {e}"]}

    # import here to avoid hard dependency on qdrant types at startup
    try:
        from qdrant_client.models import PointStruct  # type: ignore
    except Exception as e:
        return {"files": 0, "chunks": 0, "notes": [f"qdrant_client missing: {e}"]}

    for kb in sorted(source_kbs):
        folder = KB_PATHS.get(kb)
        if not folder or not os.path.isdir(folder):
            notes.append(f"KB folder missing: {kb} -> {folder}")
            continue

        # find SUMM_*.md files
        summ_files: List[str] = []
        for root, _, files in os.walk(folder):
            if "original" in {p.lower() for p in root.split(os.sep)}:
                continue
            for fn in files:
                if fn.startswith("SUMM_") and fn.lower().endswith(".md"):
                    summ_files.append(os.path.join(root, fn))

        for summ_path in sorted(summ_files):
            rel_path = os.path.relpath(summ_path, folder)

            # replace-by-file: delete existing vault points for this source file
            _qdrant_delete_for_vault_file(client, vault_kb=VAULT_KB_NAME, source_kb=kb, rel_path=rel_path)

            try:
                text = _read_text_file(summ_path)
            except Exception as e:
                notes.append(f"read fail: {kb}:{rel_path}: {e}")
                continue

            # Strip HTML comment header if present
            if text.lstrip().startswith("<!--"):
                end = text.find("-->")
                if end != -1:
                    text = text[end + 3 :]

            text = (text or "").strip()
            if not text:
                continue

            chunks = _word_chunks(text, VAULT_CHUNK_WORDS, VAULT_OVERLAP_WORDS)
            if not chunks:
                continue

            points: List[Any] = []
            # deterministic ids so re-promote is stable
            ns = uuid.UUID("12345678-1234-5678-1234-567812345678")

            for idx, ch in enumerate(chunks):
                try:
                    vec = embed_model.encode(["passage: " + ch], normalize_embeddings=True)[0]
                    vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                except Exception as e:
                    notes.append(f"embed fail: {kb}:{rel_path}#{idx}: {e}")
                    continue

                # id is uuid5(kb|rel_path|idx|hashprefix)
                key = f"{kb}|{rel_path}|{idx}|{hashlib.sha1(ch[:120].encode('utf-8', errors='ignore')).hexdigest()}"
                try:
                    pid = str(uuid.uuid5(ns, key)) if uuid else key
                except Exception:
                    pid = key

                payload = {
                    "kb": VAULT_KB_NAME,
                    "text": ch,
                    "source_kb": kb,
                    "source_rel_path": rel_path,
                    "file": os.path.basename(summ_path),
                    "path": summ_path,
                }

                points.append(
                    PointStruct(
                        id=pid,
                        vector={vector_name: vec_list},
                        payload=payload,
                    )
                )

            if points:
                try:
                    client.upsert(collection_name=collection, points=points, wait=True)
                    files_done += 1
                    chunks_done += len(points)
                except Exception as e:
                    notes.append(f"upsert fail: {kb}:{rel_path}: {e}")

    return {"files": files_done, "chunks": chunks_done, "notes": notes}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _is_command(s: str) -> bool:
    """Check if string is a session command (>>). Does NOT match ?? (Vodka recall)."""
    # DEFENSIVE: Normalize broken UTF-8 sequences
    s = (s or "").replace("Â»", ">>").replace("Â»", ">>").replace("Â¿", "??").lstrip()
    # Explicitly exclude ?? (Vodka recall)
    if s.startswith("??"):
        return False
    return s.startswith(">>") or s.startswith("»")
def _strip_command_prefix(s: str) -> str:
    # DEFENSIVE: Normalize broken UTF-8 sequences
    s = (s or "").replace("Â»", ">>").replace("Â»", ">>").lstrip()
    if s.startswith("»"):
        return s[1:].strip()
    if s.startswith(">>"):
        return s[2:].strip()
    return s.strip()


def _split_selector(user_text: str) -> Tuple[str, str]:
    """Return (selector, text). selector is one of: '', 'mentats','fun','vision','ocr'."""
    t = (user_text or "").lstrip()
    if not t.startswith("##"):
        return "", user_text

    # allow: ##m, ##mentats, ##fun, ##vision, ##ocr
    m = re.match(r"^##\s*([A-Za-z_]+)\b(.*)$", t, flags=re.I | re.S)
    if not m:
        return "", user_text

    sel = (m.group(1) or "").strip().lower()
    rest = (m.group(2) or "").lstrip()

    if sel in ("m", "mentats"):
        return "mentats", rest
    if sel in ("fun",):
        return "fun", rest
    if sel in ("vision",):
        return "vision", rest
    if sel in ("ocr",):
        return "ocr", rest

    return "", user_text


def _help_text() -> str:
    try:
        with open(CHEAT_SHEET_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[help missing: {e}]"


def _parse_args(cmd: str) -> List[str]:
    return [p for p in (cmd or "").split() if p]


def handle_command(cmd_text: str, *, state: SessionState, session_id: str) -> Optional[str]:
    """Return immediate reply if handled, else None."""
    if not _is_command(cmd_text):
        return None

    cmd = _strip_command_prefix(cmd_text)
    if not cmd:
        return "[router] empty command"

    low = cmd.strip().lower()
    low = low.replace("-", "_")
    parts = _parse_args(cmd)

    # status/help
    if low == "status":
        return (
            "[status]\n"
            f"session_id={session_id}\n"
            f"attached_kbs={sorted(state.attached_kbs)}\n"
            f"fun_sticky={state.fun_sticky}\n"
            f"fun_rewrite_sticky={state.fun_rewrite_sticky}\n"
            f"rag_last_query={state.rag_last_query!r}\n"
            f"rag_last_hits={state.rag_last_hits}\n"
            f"vault_last_query={state.vault_last_query!r}\n"
            f"vault_last_hits={state.vault_last_hits}\n"
        )

    if low == "help":
        return _help_text()

    # trust mode (tool recommendation)
    if parts and parts[0].lower() == "trust":
        if not handle_trust_command:
            return "[router] trust_pipeline not available"
        
        query = " ".join(parts[1:]).strip()
        if not query:
            return "[router] usage: >>trust <query>"
        
        recommendations = generate_recommendations(
            query=query,
            attached_kbs=state.attached_kbs,
            kb_paths=KB_PATHS,
            vault_kb_name=VAULT_KB_NAME
        )
        
        # Store recommendations for A/B/C response handling
        state.pending_trust_query = query
        state.pending_trust_recommendations = recommendations
        
        # Format and return recommendations
        from .trust_pipeline import format_recommendations
        return "[trust] " + format_recommendations(recommendations, query=query)

    # attach/detach/list
    if parts and parts[0].lower() in ("attach", "a"):
        if len(parts) < 2:
            return "[router] usage: >>attach <kb|all>"

        target = parts[1].strip().lower()

        if target == "all":
            # attach all known non-empty kb_paths
            for k in sorted(KB_PATHS.keys()):
                if k:
                    state.attached_kbs.add(k)
            return f"[router] attached ALL: {sorted(state.attached_kbs)}"

        # Special error for vault (Qdrant collection, not filesystem KB)
        if target == VAULT_KB_NAME:
            return (
                f"[error] '{VAULT_KB_NAME}' is the Qdrant collection, not a filesystem KB.\n"
                f"\n"
                f"To search Vault:\n"
                f"  • Use ##mentats <query> (automatically searches Qdrant)\n"
                f"\n"
                f"Available filesystem KBs:\n"
                f"  • {', '.join(sorted(KB_PATHS.keys()))}\n"
                f"\n"
                f"Use >>attach <kb> to attach a filesystem KB."
            )

        # Only allow filesystem KBs
        if target in KB_PATHS:
            state.attached_kbs.add(target)
            return f"[router] attached: {sorted(state.attached_kbs)}"

        return f"[router] unknown kb: '{target}' (known: {sorted(KB_PATHS.keys())})"

    if parts and parts[0].lower() in ("detach", "d"):
        if len(parts) < 2:
            return "[router] usage: >>detach <kb|all>"
        target = parts[1].strip().lower()
        if target == "all":
            state.attached_kbs.clear()
            return "[router] detached ALL"
        if target in state.attached_kbs:
            state.attached_kbs.remove(target)
            return f"[router] detached '{target}'"
        return f"[router] kb not attached: '{target}'"

    if low in ("list_kb", "list", "kbs"):
        return "[router] known KBs: " + ", ".join(sorted(KB_PATHS.keys()))

    # fun toggles
    if low in ("fun", "f"):
        state.fun_sticky = True
        state.fun_rewrite_sticky = False
        return "[router] fun mode ON (sticky)"

    if low in ("fun off", "f off"):
        state.fun_sticky = False
        return "[router] fun mode OFF"

    if low in ("fun_rewrite", "fr"):
        state.fun_rewrite_sticky = True
        state.fun_sticky = False
        return "[router] fun rewrite ON (sticky)"

    if low in ("fun_rewrite off", "fr off"):
        state.fun_rewrite_sticky = False
        return "[router] fun rewrite OFF"

    # peek
    if parts and parts[0].lower() == "peek":
        q = cmd[len(parts[0]):].strip()
        if not q:
            return "[router] usage: >>peek <query>"
        if not build_fs_facts_block:
            return "[router] fs_rag not available"
        facts = build_fs_facts_block(q, state.attached_kbs, KB_PATHS, top_k=FS_TOP_K, max_chars=FS_MAX_CHARS)
        return "[peek]\n" + (facts or "(no facts)")

    # summ / ingest (alias)
    if parts and parts[0].lower() in ("summ", "summarize", "ingest"):
        if not os.path.isfile(SUMM_PROMPT_PATH):
            return f"[router] SUMM.md missing at {SUMM_PROMPT_PATH}"

        # target: NEW (summarize in currently attached KBs), or a kb name, or ALL
        target = parts[1].lower() if len(parts) >= 2 else "new"

        if target == "new":
            if not state.attached_kbs:
                return "[router] No KBs attached. Attach a KB, then: >>summ new"
            total_created = 0
            total_skipped = 0
            notes: List[str] = []
            for kb in sorted(state.attached_kbs):
                if kb == VAULT_KB_NAME:
                    continue
                folder = KB_PATHS.get(kb)
                res = summ_new_in_kb(kb, folder or "")
                total_created += int(res.get("summ_created", 0))
                total_skipped += int(res.get("summ_skipped", 0))
                notes.extend(res.get("notes", []) or [])
            msg = f"[router] SUMM complete: created={total_created} skipped={total_skipped}"
            if notes:
                msg += "\n" + "\n".join("- " + n for n in notes[:25])
            return msg

        if target == "all":
            # Summ in every kb folder
            total_created = 0
            total_skipped = 0
            notes: List[str] = []
            for kb, folder in sorted(KB_PATHS.items()):
                if kb == VAULT_KB_NAME:
                    continue
                res = summ_new_in_kb(kb, folder)
                total_created += int(res.get("summ_created", 0))
                total_skipped += int(res.get("summ_skipped", 0))
                notes.extend(res.get("notes", []) or [])
            msg = f"[router] SUMM ALL complete: created={total_created} skipped={total_skipped}"
            if notes:
                msg += "\n" + "\n".join("- " + n for n in notes[:25])
            return msg

        # single kb
        kb = target
        folder = KB_PATHS.get(kb)
        if not folder:
            return f"[router] unknown kb '{kb}'"
        res = summ_new_in_kb(kb, folder)
        msg = f"[router] SUMM {kb}: created={res.get('summ_created', 0)} skipped={res.get('summ_skipped', 0)}"
        notes = res.get("notes", []) or []
        if notes:
            msg += "\n" + "\n".join("- " + n for n in notes[:25])
        return msg

    # move to vault
    if low.startswith("move to vault") or low.startswith("move_to_vault") or low.startswith("mtv"):
        # parse: >>move to vault [all]
        arg = parts[-1].lower() if parts else ""
        if arg == "all":
            src = set(k for k in KB_PATHS.keys() if k and k != VAULT_KB_NAME)
        else:
            src = set(k for k in state.attached_kbs if k and k != VAULT_KB_NAME)

        if not src:
            return "[router] No source KBs to promote. Attach KB(s) then: >>move to vault"

        res = move_summ_to_vault(src)
        msg = f"[router] move-to-vault complete: files={res.get('files', 0)} chunks={res.get('chunks', 0)}"
        notes = res.get("notes", []) or []
        if notes:
            msg += "\n" + "\n".join("- " + n for n in notes[:25])
        return msg

    # =========================================================================
    # SIDECAR COMMANDS (non-LLM utilities)
    # =========================================================================

    # >>calc <expression>
    if parts and parts[0].lower() == "calc":
        if not parse_and_eval_calc:
            return "[router] calc not available (sidecars.py missing)"
        expr = cmd[len(parts[0]):].strip()
        if not expr:
            return "[calc] usage: >>calc <expression>\nExamples: >>calc 30% of 79.95, >>calc 14*365, >>calc sqrt(16)"
        result = parse_and_eval_calc(expr)
        return f"[calc] {expr} = {format_calc_result(result)}"

    # >>list (Vodka memories)
    if low == "list" and cmd.startswith(">>"):
        # Note: This is >>list (not ??)
        if not list_vodka_memories or not state.vodka:
            return "[list] vodka not available"
        entries = list_vodka_memories(state.vodka)
        return format_memory_list(entries)

    # >>find <query> (search KBs)
    if parts and parts[0].lower() == "find":
        if not find_quote_in_kbs:
            return "[router] find not available (sidecars.py missing)"
        query = cmd[len(parts[0]):].strip()
        if not query:
            return "[find] usage: >>find <text to search for>"
        if not state.attached_kbs:
            return "[find] No KBs attached. >>attach <kb> first"
        result = find_quote_in_kbs(query, state.attached_kbs, KB_PATHS)
        return format_quote_result(result)

    # >>flush (CTC cache)
    if low == "flush":
        if not flush_ctc_cache or not state.vodka:
            return "[router] flush not available"
        return flush_ctc_cache(state.vodka)

    # >>wiki <topic> (Wikipedia summary)
    if parts and parts[0].lower() == "wiki":
        if not handle_wiki_query:
            return "[router] wiki not available (sidecars.py missing)"
        topic = cmd[len(parts[0]):].strip()
        if not topic:
            return "[wiki] usage: >>wiki <topic>\nExample: >>wiki Albert Einstein"
        return handle_wiki_query(topic)

    # >>exchange <query> (Currency conversion)
    if parts and parts[0].lower() == "exchange":
        if not handle_exchange_query:
            return "[router] exchange not available (sidecars.py missing)"
        query = cmd[len(parts[0]):].strip()
        if not query:
            return "[exchange] usage: >>exchange <query>\nExamples: >>exchange 1 USD to EUR, >>exchange GBP to JPY"
        return handle_exchange_query(query)

    # >>weather <location> (Current weather)
    if parts and parts[0].lower() == "weather":
        if not handle_weather_query:
            return "[router] weather not available (sidecars.py missing)"
        location = cmd[len(parts[0]):].strip()
        if not location:
            return "[weather] usage: >>weather <location>\nExample: >>weather Perth"
        return handle_weather_query(location)

    # >>raw (Raw mode toggle)
    if low == "raw":
        state.raw_sticky = True
        return "[router] raw mode ON (sticky, no Serious formatting)"

    if low == "raw off":
        state.raw_sticky = False
        return "[router] raw mode OFF"

    return f"[router] unknown command: {cmd}"


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def build_fs_facts(query: str, state: SessionState) -> str:
    if not build_fs_facts_block:
        return ""

    q = " ".join((query or "").split()).strip()
    state.rag_last_query = q
    state.rag_last_hits = 0

    if not q:
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
    q = " ".join((query or "").split()).strip()
    state.vault_last_query = q
    state.vault_last_hits = 0

    if not q:
        return ""

    # lazy import
    try:
        from rag import build_rag_block
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


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "1.0.7"}


@app.get("/v1/models")
def v1_models():
    # minimal OpenAI-compatible
    models = []
    for role, model in (ROLES or {}).items():
        if model:
            models.append({"id": model, "object": "model"})
    # Advertise router meta-model
    models.append({"id": "moa-router", "object": "model"})
    return {"object": "list", "data": models}


def _make_openai_response(text: str, model: str = "moa-router") -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _stream_sse(text: str, model: str = "moa-router") -> Iterable[bytes]:
    """OpenAI-style SSE streamer with keepalive pings.

    Open WebUI expects JSON payloads per `data:` line and a terminal `data: [DONE]`.
    Keepalive comments prevent OWUI from timing out on slow responses.
    """
    chunk = 48
    for i in range(0, len(text), chunk):
        part = text[i : i + chunk]
        payload = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        yield ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")
        yield b": keepalive\n\n"
    payload = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")
    yield b"data: [DONE]\n\n"

def _session_id_from_request(req: Request, body: Dict[str, Any]) -> str:
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

    # CRITICAL CHECK: Auto-vision detection
    # If images are present AND user wrote something (not just image), route to vision
    # IMPORTANT: Check ONLY the latest user message, not entire history!
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
        # Images + text (no command) → force vision mode, disable sticky Fun/FR
        if state.fun_sticky or state.fun_rewrite_sticky:
            state.fun_sticky = False
            state.fun_rewrite_sticky = False
        # Auto-route to vision pipeline
        text = call_model_messages(role="vision", messages=raw_messages, max_tokens=700, temperature=0.2, top_p=0.9)
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Commands are handled against raw user text.
    # BUT: Check for per-turn selectors (##) FIRST, before session commands (>>)
    # This prevents ##vision from being treated as >>vision command
    selector, user_text = _split_selector(user_text_raw)
    
    # CRITICAL: Check if user is responding to pending trust recommendations (A/B/C/D/E)
    # This must happen BEFORE normal command processing
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
                if command == '>>attach all' and original_query:
                    state.auto_query_after_attach = original_query
                    state.auto_detach_after_response = True  # Clean up after response
                
                # Execute the chosen command
                if command.startswith('>>'):
                    # It's a session command - execute via handle_command
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
                                
                                # Don't return - continue to normal pipeline processing
                                # First, send the attach confirmation as a message
                                # (We can't do this in streaming mode cleanly, so we'll just skip to the query)
                            else:
                                # No auto query - return the command result
                                if stream:
                                    return StreamingResponse(_stream_sse(cmd_reply), media_type="text/event-stream")
                                return JSONResponse(_make_openai_response(cmd_reply))
                    except Exception as e:
                        text = f"[router error: {e.__class__.__name__}: {e}]"
                        if stream:
                            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
                        return JSONResponse(_make_openai_response(text))
                elif command.startswith('##'):
                    # It's a per-turn selector - override user_text_raw and continue processing
                    user_text_raw = command
                    selector, user_text = _split_selector(user_text_raw)
                    # Update raw_messages so model sees the actual command
                    for i in range(len(raw_messages) - 1, -1, -1):
                        if raw_messages[i].get("role") == "user":
                            raw_messages[i]["content"] = command
                            break
                else:
                    # It's a regular query - override user_text_raw and continue processing
                    user_text_raw = command
                    selector, user_text = _split_selector(user_text_raw)
                    # Update raw_messages so model sees the actual query
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
    vodka_cfg = dict(_cfg_get("vodka", {}))
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
    
    # ⚠️  CRITICAL: Check if Vodka has already answered (hard commands like ?? list, !! nuke)
    # If the last message is an assistant reply with Vodka marker, return it directly (no LLM)
    if raw_messages and raw_messages[-1].get("role") == "assistant":
        last_msg_content = raw_messages[-1].get("content", "")
        # Check if this looks like a Vodka hard command response
        # Hard command indicators: [vodka], [Vodka Memory Store], etc.
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
        # Use multimodal history as-is for vision calls
        text = call_model_messages(role=role, messages=raw_messages, max_tokens=700, temperature=0.2, top_p=0.9)
        if stream:
            return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
        return JSONResponse(_make_openai_response(text))

    # Mentats selector
    if selector == "mentats":
        # DEBUG: Mentats selector triggered
        print(f"[DEBUG] Mentats selector triggered, user_text={user_text!r}", flush=True)
        
        # Mentats MUST be isolated: drop fun modes and ignore attached KBs.
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

        # run mentats (router provides RAG+constraints hooks)
        def _build_rag_block(query: str, collection: str = "vault") -> str:
            return build_vault_facts(query, state)

        def _build_constraints_block(query: str) -> str:
            return ""

        # Wrapper to enforce temperature=0.1 for critic role (step 2 fact-checking)
        def _call_model_with_critic_temp(*, role: str, prompt: str, max_tokens: int = 256, temperature: float = 0.3, top_p: float = 0.9) -> str:
            if role == "critic":
                temperature = 0.1  # Force low temperature for critic to prevent hallucinations
            return call_model_prompt(role=role, prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        # No chat history / Vodka into Mentats as truth: pass empty history and NoOp vodka.
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
        # per-turn fun (not sticky)
        state.fun_sticky = False
        state.fun_rewrite_sticky = False
        fun_mode = "fun"
    else:
        fun_mode = ""

    # Apply sticky fun modes for default pipeline
    # BUT: Check if recent history has Mentats (Fun cannot run on Mentats context)
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
                constraints_block="",
                thinker_role="thinker",
            ).strip()
            tone = _infer_tone(user_text, base)
            quote = _pick_quote_for_tone(tone)
            out = f'[FUN] "{quote}"\n\n{base}' if quote else f"[FUN]\n\n{base}"
            text = out
        else:
            # Build a tone-matched pool and let fun.py randomize within it.
            base_preview = run_serious(
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                vodka=vodka,
                call_model=call_model_prompt,
                facts_block=facts_block,
                constraints_block="",
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

        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        
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
        
        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        
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
            constraints_block="",
            thinker_role="thinker",
        ).strip()
        
        # Add disclaimer if KBs were attached but model used training data
        if state.attached_kbs and "Source: Model" in text:
            kb_list = ', '.join(sorted(state.attached_kbs))
            disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
            text = disclaimer + text
        
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
        constraints_block="",
        thinker_role="thinker",
    ).strip()

    # Add disclaimer if KBs were attached but model used training data
    # Check for "Source: Model" marker (serious.py adds this when facts aren't useful)
    if state.attached_kbs and "Source: Model" in text:
        kb_list = ', '.join(sorted(state.attached_kbs))
        disclaimer = f"[Note: No relevant information found in attached KBs ({kb_list}). Answer based on pre-trained data.]\n\n"
        text = disclaimer + text
    
    # Auto-detach AFTER disclaimer check (so attached_kbs is still available for the check)
    if state.auto_detach_after_response:
        state.attached_kbs.clear()
        state.auto_detach_after_response = False

    if stream:
        return StreamingResponse(_stream_sse(text), media_type="text/event-stream")
    return JSONResponse(_make_openai_response(text))


# Convenience for `python router_fastapi.py`
if __name__ == "__main__":
    import uvicorn

    host = str(_cfg_get("server.host", "0.0.0.0"))
    port = int(_cfg_get("server.port", 9000))
    uvicorn.run("router_fastapi:app", host=host, port=port, reload=False)
