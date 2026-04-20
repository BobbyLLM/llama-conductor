from __future__ import annotations

import base64
import hmac
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


@dataclass
class ShimConfig:
    host: str
    port: int
    llama_server_url: str
    router_url: str
    auth_user: str
    auth_password: str
    force_vision: bool
    render_mode: str
    inject_ui: bool
    ui_model_alias: str
    router_model_id: str
    single_instance: bool
    takeover_existing: bool
    takeover_foreign: bool
    takeover_timeout_s: float


def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_render_mode(name: str, default: str = "stream") -> str:
    raw = (os.getenv(name) or default).strip().lower()
    if raw in {"stream", "buffered"}:
        return raw
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _basename_from_url(url: str) -> str:
    path = urlparse(url).path
    if not path:
        return url
    bits = [b for b in path.split("/") if b]
    return bits[-1] if bits else url


def load_config() -> ShimConfig:
    router_model_id = str(os.getenv("MOA_CHAT_ROUTER_MODEL_ID", "moa-router")).strip()
    if not router_model_id:
        router_model_id = "moa-router"
    return ShimConfig(
        host=os.getenv("MOA_CHAT_HOST", "0.0.0.0"),
        port=int(os.getenv("MOA_CHAT_PORT", "8088")),
        llama_server_url=_strip_trailing_slash(
            os.getenv("MOA_CHAT_LLAMASERVER_URL", "http://127.0.0.1:8010")
        ),
        router_url=_strip_trailing_slash(
            os.getenv("MOA_CHAT_ROUTER_URL", "http://127.0.0.1:9000")
        ),
        auth_user=os.getenv("MOA_CHAT_AUTH_USER", "moa"),
        auth_password=os.getenv("MOA_CHAT_PASSWORD", ""),
        force_vision=_env_bool("MOA_CHAT_FORCE_VISION", default=False),
        render_mode=_env_render_mode("MOA_CHAT_RENDER_MODE", default="stream"),
        inject_ui=_env_bool("MOA_CHAT_INJECT_UI", default=True),
        ui_model_alias=str(os.getenv("MOA_CHAT_UI_MODEL_ALIAS", "")).strip(),
        router_model_id=router_model_id,
        single_instance=_env_bool("MOA_CHAT_SINGLE_INSTANCE", default=True),
        takeover_existing=_env_bool("MOA_CHAT_TAKEOVER_EXISTING", default=True),
        takeover_foreign=_env_bool("MOA_CHAT_TAKEOVER_FOREIGN", default=True),
        takeover_timeout_s=_env_float("MOA_CHAT_TAKEOVER_TIMEOUT_S", default=5.0),
    )


CONFIG = load_config()
APP = FastAPI(title="MoA Chat UI Shim", version="0.2.0")

_MODEL_CACHE_TTL_S = 20.0
_MODEL_CACHE: Dict[str, Any] = {
    "at": 0.0,
    "entries": [],
}


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}


def _safe_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}


_SHIM_IMAGE_EMITTER_ENABLED = _env_bool("MOA_CHAT_IMAGE_EMITTER_ENABLED", default=False)
_SHIM_IMAGE_EMITTER_MARKER = "MOA_SHIM_IMAGE_EMITTER_V1"
_SHIM_IMAGE_EMITTER_PATH = str(
    os.getenv(
        "MOA_CHAT_IMAGE_EMITTER_PATH",
        os.path.join(os.getcwd(), "TEST_ARTIFACTS_VALIDATION", "image-emitter-shim.jsonl"),
    )
)


def _shim_has_image_signal(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        s = value.strip().lower()
        return s.startswith("data:image/") or s.startswith("blob:") or s.startswith("file:")
    if isinstance(value, list):
        return any(_shim_has_image_signal(v) for v in value)
    if not isinstance(value, dict):
        return False

    typ = str(value.get("type", "")).strip().lower()
    if typ in {"image", "image_url", "input_image"}:
        return True

    mime = str(
        value.get("mime_type")
        or value.get("content_type")
        or value.get("mimetype")
        or ""
    ).strip().lower()
    if mime.startswith("image/"):
        return True

    image_url = value.get("image_url")
    if isinstance(image_url, str) and _shim_has_image_signal(image_url):
        return True
    if isinstance(image_url, dict) and _shim_has_image_signal(image_url.get("url")):
        return True

    for key in ("url", "uri", "src", "data", "file", "files", "image", "images", "attachments", "content", "parts"):
        if key in value and _shim_has_image_signal(value.get(key)):
            return True
    return False


def _shim_summarize_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {
            "kind": "str",
            "len": len(content),
            "preview": content[:120],
            "has_image_signal": bool(_shim_has_image_signal(content)),
        }
    if isinstance(content, list):
        type_counts: Dict[str, int] = {}
        image_blocks = 0
        for b in content:
            if isinstance(b, dict):
                t = str(b.get("type", "")).strip().lower() or "dict"
                type_counts[t] = int(type_counts.get(t, 0)) + 1
                if _shim_has_image_signal(b):
                    image_blocks += 1
            elif isinstance(b, str):
                type_counts["str"] = int(type_counts.get("str", 0)) + 1
        return {
            "kind": "list",
            "len": len(content),
            "types": type_counts,
            "image_blocks": image_blocks,
            "has_image_signal": bool(_shim_has_image_signal(content)),
        }
    if isinstance(content, dict):
        return {
            "kind": "dict",
            "keys": sorted([str(k) for k in list(content.keys())[:20]]),
            "has_image_signal": bool(_shim_has_image_signal(content)),
        }
    return {
        "kind": type(content).__name__,
        "has_image_signal": bool(_shim_has_image_signal(content)),
    }


def _shim_last_user_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return {
                "keys": sorted([str(k) for k in m.keys()]),
                "content": _shim_summarize_content(m.get("content", "")),
                "files_has_image": bool(_shim_has_image_signal(m.get("files"))),
                "attachments_has_image": bool(_shim_has_image_signal(m.get("attachments"))),
            }
    return {"kind": "none"}


def _shim_collect_probe_paths(value: Any, limit: int = 60) -> List[str]:
    out: List[str] = []

    def rec(v: Any, prefix: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(v, dict):
            for k, nv in v.items():
                ks = str(k)
                path = f"{prefix}.{ks}" if prefix else ks
                kl = ks.lower()
                if any(tok in kl for tok in ("image", "file", "attach", "blob", "mime", "audio")):
                    out.append(path)
                    if len(out) >= limit:
                        return
                rec(nv, path)
        elif isinstance(v, list):
            for idx, nv in enumerate(v):
                rec(nv, f"{prefix}[{idx}]")
                if len(out) >= limit:
                    return

    rec(value, "")
    return out


def _shim_emit_image_trace(stage: str, payload: Dict[str, Any]) -> None:
    if not _SHIM_IMAGE_EMITTER_ENABLED:
        return
    event = {
        "marker": _SHIM_IMAGE_EMITTER_MARKER,
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        **(payload or {}),
    }
    try:
        print(f"[SHIM-IMAGE-EMITTER] {json.dumps(event, ensure_ascii=False)}", flush=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(_SHIM_IMAGE_EMITTER_PATH), exist_ok=True)
        with open(_SHIM_IMAGE_EMITTER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _parse_requested_model(request_body: bytes) -> str:
    if not request_body:
        return ""
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    model = payload.get("model")
    return model.strip() if isinstance(model, str) else ""


def _is_router_model(model_name: str) -> bool:
    norm = (model_name or "").strip().lower()
    if not norm:
        return True
    if norm == CONFIG.router_model_id.lower():
        return True
    if norm == "moa-router":
        return True
    alias = (CONFIG.ui_model_alias or "").strip().lower()
    return bool(alias) and norm == alias


def _target_base(path: str, request_body: bytes = b"") -> str:
    if path.startswith("/v1/models"):
        return CONFIG.router_url
    if path == "/v1/chat/completions":
        requested_model = _parse_requested_model(request_body)
        if _is_router_model(requested_model):
            return CONFIG.router_url
        return CONFIG.llama_server_url
    if path.startswith("/v1/"):
        return CONFIG.router_url
    return CONFIG.llama_server_url


def _is_authorized(request: Request) -> bool:
    if not CONFIG.auth_password:
        return True

    raw_auth = request.headers.get("authorization", "")
    if not raw_auth.lower().startswith("basic "):
        return False

    try:
        encoded = raw_auth.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode("utf-8", errors="strict")
        user, sep, password = decoded.partition(":")
        if not sep:
            return False
        return hmac.compare_digest(user, CONFIG.auth_user) and hmac.compare_digest(
            password, CONFIG.auth_password
        )
    except Exception:
        return False


@APP.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path == "/shim/healthz":
        return await call_next(request)

    if not _is_authorized(request):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="moa-chat-ui"'},
        )

    return await call_next(request)


@APP.get("/shim/healthz")
async def healthz():
    return {
        "ok": True,
        "host": CONFIG.host,
        "port": CONFIG.port,
        "llama_server_url": CONFIG.llama_server_url,
        "router_url": CONFIG.router_url,
        "auth_enabled": bool(CONFIG.auth_password),
        "force_vision": CONFIG.force_vision,
        "render_mode": CONFIG.render_mode,
        "inject_ui": CONFIG.inject_ui,
        "ui_model_alias": CONFIG.ui_model_alias,
        "router_model_id": CONFIG.router_model_id,
        "single_instance": CONFIG.single_instance,
        "takeover_existing": CONFIG.takeover_existing,
        "takeover_foreign": CONFIG.takeover_foreign,
        "takeover_timeout_s": CONFIG.takeover_timeout_s,
    }


def _build_target_url(request: Request, path: str, request_body: bytes = b"") -> str:
    target = _target_base(path, request_body) + path
    if request.url.query:
        target += "?" + request.url.query
    return target


def _mark_media_capable(model_obj: Dict[str, Any]) -> None:
    # Advertise multimodal support so the UI enables image/audio affordances.
    model_obj["modalities"] = {"vision": True, "audio": True}
    model_obj["input_modalities"] = ["text", "image", "audio"]
    model_obj["output_modalities"] = ["text", "audio"]

    caps = model_obj.get("capabilities")
    if not isinstance(caps, dict):
        caps = {}
    caps["vision"] = True
    caps["images"] = True
    caps["audio"] = True
    caps["input_audio"] = True
    caps["output_audio"] = True
    model_obj["capabilities"] = caps


def _capability_list(raw: Any) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        token = str(name or "").strip().lower()
        if not token or token in seen:
            return
        seen.add(token)
        out.append(token)

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                add(item)
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if not value:
                continue
            k = str(key or "").strip().lower()
            if not k:
                continue
            if k in {"images", "input_image", "output_image"}:
                add("vision")
                continue
            if k in {"input_audio", "output_audio"}:
                add("audio")
                continue
            add(k)

    return out


def _build_models_meta(payload: Dict[str, Any], data: List[Any]) -> List[Dict[str, Any]]:
    source_models = payload.get("models")
    if not isinstance(source_models, list):
        source_models = []

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue

        src = source_models[idx] if idx < len(source_models) and isinstance(source_models[idx], dict) else {}
        name = str(src.get("name") or model_id).strip() or model_id
        model_name = str(src.get("model") or model_id).strip() or model_id
        description = src.get("description")
        details = src.get("details")

        caps = _capability_list(src.get("capabilities"))
        if not caps:
            caps = _capability_list(item.get("capabilities"))
        if _is_router_model_id(model_id):
            for required in ("vision", "audio"):
                if required not in caps:
                    caps.append(required)

        model_meta: Dict[str, Any] = {
            "name": name,
            "model": model_name,
            "capabilities": caps,
        }
        if isinstance(description, str) and description.strip():
            model_meta["description"] = description
        if details is not None:
            model_meta["details"] = details
        out.append(model_meta)

    return out


def _is_router_model_id(model_id: str) -> bool:
    mid = str(model_id or "").strip().lower()
    if not mid:
        return False
    if mid == "moa-router":
        return True
    if mid == str(CONFIG.router_model_id or "").strip().lower():
        return True
    alias = str(CONFIG.ui_model_alias or "").strip().lower()
    return bool(alias) and mid == alias


def _patch_json_response(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if path == "/props":
        if CONFIG.force_vision:
            modalities = payload.get("modalities")
            if not isinstance(modalities, dict):
                modalities = {}
            modalities["vision"] = True
            modalities["audio"] = True
            payload["modalities"] = modalities
            payload["input_modalities"] = ["text", "image", "audio"]
            payload["output_modalities"] = ["text", "audio"]

            caps = payload.get("capabilities")
            if not isinstance(caps, dict):
                caps = {}
            caps["vision"] = True
            caps["images"] = True
            caps["audio"] = True
            payload["capabilities"] = caps

        if CONFIG.ui_model_alias:
            payload["model_alias"] = CONFIG.ui_model_alias
        return payload

    if path == "/v1/models":
        data = payload.get("data")
        if isinstance(data, list):
            router_template: Dict[str, Any] | None = None
            existing_ids = {
                str(m.get("id") or "").strip().lower()
                for m in data
                if isinstance(m, dict) and str(m.get("id") or "").strip()
            }
            for model in data:
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("id") or "")
                if _is_router_model_id(model_id):
                    router_template = dict(model)
                if (
                    CONFIG.router_model_id
                    and CONFIG.router_model_id != "moa-router"
                    and model_id == "moa-router"
                ):
                    model["id"] = CONFIG.router_model_id
                if CONFIG.force_vision or _is_router_model_id(model_id):
                    _mark_media_capable(model)

            if router_template is None:
                for model in data:
                    if isinstance(model, dict) and str(model.get("id") or "").strip().lower() == "moa-router":
                        router_template = dict(model)
                        break

            if isinstance(router_template, dict):
                _mark_media_capable(router_template)
                for alias_id in [str(CONFIG.router_model_id or "").strip(), str(CONFIG.ui_model_alias or "").strip()]:
                    aid = alias_id.strip()
                    if not aid:
                        continue
                    if aid.lower() in existing_ids:
                        continue
                    aliased = dict(router_template)
                    aliased["id"] = aid
                    data.append(aliased)
                    existing_ids.add(aid.lower())
            payload["models"] = _build_models_meta(payload, data)
        return payload

    return payload


def _should_stream(path: str, request_body: bytes, accept_header: str) -> bool:
    if "text/event-stream" in accept_header.lower():
        return True

    if path in {"/completion", "/completions", "/v1/chat/completions"}:
        body_l = request_body.lower()
        return b'"stream":true' in body_l or b'"stream": true' in body_l

    return False


def _is_chat_path(path: str) -> bool:
    return path in {"/v1/chat/completions", "/completions", "/completion"}


def _decode_json_body(request_body: bytes) -> Dict[str, Any]:
    if not request_body:
        return {}
    try:
        payload = json.loads(request_body.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _encode_json_body(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _normalize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokenize_model_name(value: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(value or "").lower()) if len(t) >= 2]


def _score_model_candidate(requested: str, candidate: str) -> int:
    req = str(requested or "").strip()
    cand = str(candidate or "").strip()
    if not req or not cand:
        return -1
    if req.lower() == cand.lower():
        return 10000

    req_norm = _normalize_model_name(req)
    cand_norm = _normalize_model_name(cand)
    if not req_norm or not cand_norm:
        return -1

    score = 0
    if req_norm == cand_norm:
        score += 7000
    if cand_norm.startswith(req_norm):
        score += 2200
    elif req_norm in cand_norm:
        score += 1600

    req_tokens = _tokenize_model_name(req)
    if req_tokens:
        hits = sum(1 for tok in req_tokens if tok in cand_norm)
        score += int((hits / len(req_tokens)) * 1200)
        if hits == len(req_tokens):
            score += 800

    length_gap = abs(len(cand_norm) - len(req_norm))
    score += max(0, 250 - length_gap)
    return score


async def _get_llama_model_entries(timeout: httpx.Timeout) -> List[Dict[str, Any]]:
    now = time.monotonic()
    cached_at = float(_MODEL_CACHE.get("at") or 0.0)
    cached_entries = _MODEL_CACHE.get("entries") or []
    if (now - cached_at) <= _MODEL_CACHE_TTL_S and isinstance(cached_entries, list):
        return cached_entries

    entries: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            resp = await client.get(f"{CONFIG.llama_server_url}/v1/models")
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return cached_entries if isinstance(cached_entries, list) else []

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = str(item.get("id") or "").strip()
            if not mid:
                continue
            aliases_raw = item.get("aliases")
            aliases: List[str] = []
            if isinstance(aliases_raw, list):
                for alias in aliases_raw:
                    if isinstance(alias, str) and alias.strip():
                        aliases.append(alias.strip())
            entries.append({"id": mid, "aliases": aliases})

    _MODEL_CACHE["at"] = now
    _MODEL_CACHE["entries"] = entries
    return entries


async def _resolve_llama_model_name(requested_model: str, timeout: httpx.Timeout) -> str:
    requested = str(requested_model or "").strip()
    if not requested:
        return requested

    entries = await _get_llama_model_entries(timeout)
    if not entries:
        return requested

    canonical_ids = [str(e.get("id") or "").strip() for e in entries]
    if any(requested.lower() == mid.lower() for mid in canonical_ids if mid):
        return requested

    best_id = ""
    best_score = -1
    req_norm = _normalize_model_name(requested)

    for entry in entries:
        model_id = str(entry.get("id") or "").strip()
        if not model_id:
            continue

        candidate_names = [model_id]
        aliases = entry.get("aliases")
        if isinstance(aliases, list):
            candidate_names.extend([str(a).strip() for a in aliases if str(a).strip()])

        entry_score = -1
        for name in candidate_names:
            s = _score_model_candidate(requested, name)
            if s > entry_score:
                entry_score = s

        if entry_score > best_score:
            best_score = entry_score
            best_id = model_id
        elif entry_score == best_score and best_id:
            prev_norm = _normalize_model_name(best_id)
            cand_norm = _normalize_model_name(model_id)
            if req_norm and req_norm in cand_norm and req_norm not in prev_norm:
                best_id = model_id

    if best_score < 2150:
        return requested
    return best_id or requested


async def _rewrite_chat_model_for_forwarding(
    path: str,
    request_body: bytes,
    timeout: httpx.Timeout,
) -> bytes:
    if path != "/v1/chat/completions":
        return request_body

    payload = _decode_json_body(request_body)
    if not payload:
        return request_body

    requested = str(payload.get("model") or "").strip()
    if _is_router_model(requested):
        if payload.get("model") != "moa-router":
            payload["model"] = "moa-router"
            return _encode_json_body(payload)
        return request_body

    resolved = await _resolve_llama_model_name(requested, timeout=timeout)
    if resolved and resolved != requested:
        payload["model"] = resolved
        return _encode_json_body(payload)

    return request_body


def _sse_to_buffered_sse(raw_text: str) -> str:
    # Convert token stream (SSE delta chunks) into one final content chunk + done.
    content_parts: List[str] = []
    first_obj: Dict[str, Any] = {}

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        if not first_obj and isinstance(obj, dict):
            first_obj = obj

        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content") is not None:
                    content_parts.append(str(delta.get("content")))
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    content_parts.append(str(message.get("content")))
                    continue
                if choice.get("text") is not None:
                    content_parts.append(str(choice.get("text")))
        elif isinstance(obj, dict) and obj.get("content") is not None:
            content_parts.append(str(obj.get("content")))

    final_content = "".join(content_parts)
    if not final_content:
        return raw_text

    cid = str(first_obj.get("id") or f"chatcmpl-{int(datetime.now().timestamp())}")
    created = int(first_obj.get("created") or int(datetime.now().timestamp()))
    model = str(first_obj.get("model") or "moa-router")

    chunk_1 = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": final_content}, "finish_reason": None}],
    }
    chunk_2 = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk_1)}\n\ndata: {json.dumps(chunk_2)}\n\ndata: [DONE]\n\n"


def _inject_toolbar(html: str) -> str:
    if not CONFIG.inject_ui:
        return html

    # Hard fallback: force multimodal capability getters on the shipped WebUI bundle.
    # Some builds still resolve supportsVision/supportsAudio to false despite patched /props.
    html = re.sub(
        r"get supportsVision\(\)\{return this\._serverProps\?\.modalities\?\.vision\?\?!1\}",
        "get supportsVision(){return!0}",
        html,
    )
    html = re.sub(
        r"get supportsAudio\(\)\{return this\._serverProps\?\.modalities\?\.audio\?\?!1\}",
        "get supportsAudio(){return!0}",
        html,
    )

    # Newer bundled webui builds gate uploads via minified hasVision/hasAudio flow.
    html = html.replace(
        "const t=[],r=[],a={},{hasVision:i,hasAudio:s}=e;",
        "const t=[],r=[],a={};const i=!0,s=!0;",
    )
    html = html.replace(
        "get hasAudioModality(){return p(g)},get hasVisionModality(){return p(_)}",
        "get hasAudioModality(){return!0},get hasVisionModality(){return!0}",
    )
    html = html.replace(
        "const dt={hasVision:p(D),hasAudio:p(R)}",
        "const dt={hasVision:!0,hasAudio:!0}",
    )
    html = html.replace(
        'p(D)&&pt.push("images"),p(R)&&pt.push("audio files")',
        'pt.push("images"),pt.push("audio files")',
    )

    preload = r"""
<script id="moa-shim-preload">
(function () {
  function patchPropsPayload(payload) {
    if (!payload || typeof payload !== "object") return payload;
    payload.modalities = { vision: true, audio: true };
    payload.input_modalities = ["text", "image", "audio"];
    payload.output_modalities = ["text", "audio"];
    var caps = payload.capabilities;
    if (!caps || typeof caps !== "object" || Array.isArray(caps)) caps = {};
    caps.vision = true;
    caps.images = true;
    caps.audio = true;
    payload.capabilities = caps;
    return payload;
  }

  function patchModelsPayload(payload) {
    if (!payload || typeof payload !== "object") return payload;
    if (!Array.isArray(payload.data)) return payload;
    for (var i = 0; i < payload.data.length; i++) {
      var row = payload.data[i];
      if (!row || typeof row !== "object") continue;
      row.modalities = { vision: true, audio: true };
      row.input_modalities = ["text", "image", "audio"];
      row.output_modalities = ["text", "audio"];
      var rowCaps = row.capabilities;
      if (!rowCaps || typeof rowCaps !== "object" || Array.isArray(rowCaps)) rowCaps = {};
      rowCaps.vision = true;
      rowCaps.images = true;
      rowCaps.audio = true;
      rowCaps.input_audio = true;
      rowCaps.output_audio = true;
      row.capabilities = rowCaps;
    }

    var meta = payload.models;
    if (!Array.isArray(meta)) {
      meta = [];
      for (var j = 0; j < payload.data.length; j++) {
        var d = payload.data[j] || {};
        var id = String(d.id || "").trim();
        if (!id) continue;
        meta.push({ name: id, model: id, capabilities: ["vision", "audio"] });
      }
    } else {
      for (var k = 0; k < meta.length; k++) {
        var m = meta[k];
        if (!m || typeof m !== "object") continue;
        var outCaps = [];
        if (Array.isArray(m.capabilities)) {
          for (var c = 0; c < m.capabilities.length; c++) {
            var tok = String(m.capabilities[c] || "").toLowerCase().trim();
            if (tok && outCaps.indexOf(tok) < 0) outCaps.push(tok);
          }
        }
        if (outCaps.indexOf("vision") < 0) outCaps.push("vision");
        if (outCaps.indexOf("audio") < 0) outCaps.push("audio");
        m.capabilities = outCaps;
      }
    }
    payload.models = meta;
    return payload;
  }

  // Seed llama.cpp WebUI cached server props before app bundle boots.
  try {
    var k = "LlamaCppWebui.serverProps";
    var raw = localStorage.getItem(k);
    var obj = {};
    if (raw) {
      try { obj = JSON.parse(raw) || {}; } catch (_err) { obj = {}; }
    }
    obj.modalities = { vision: true, audio: true };
    obj.input_modalities = ["text", "image", "audio"];
    obj.output_modalities = ["text", "audio"];
    obj.capabilities = Object.assign({}, obj.capabilities || {}, {
      vision: true,
      images: true,
      audio: true
    });
    localStorage.setItem(k, JSON.stringify(obj));
  } catch (_err) {}

  // Force-capability patch at fetch-time so UI cannot regress to text/PDF-only mode.
  try {
    var origFetch = window.fetch ? window.fetch.bind(window) : null;
    if (origFetch && !window.__moaShimForceCapsFetch) {
      window.__moaShimForceCapsFetch = true;
      window.fetch = function (input, init) {
        var url = "";
        try { url = typeof input === "string" ? input : ((input && input.url) || ""); } catch (_err) {}
        var isProps = /\/props(\?|$)/.test(url);
        var isModels = /\/v1\/models(\?|$)/.test(url);
        if (!isProps && !isModels) return origFetch(input, init);

        return origFetch(input, init).then(function (resp) {
          try {
            var ct = String(resp.headers.get("content-type") || "").toLowerCase();
            if (ct.indexOf("application/json") < 0) return resp;
            return resp.clone().text().then(function (txt) {
              var parsed = null;
              try { parsed = JSON.parse(txt); } catch (_err) { return resp; }
              if (!parsed || typeof parsed !== "object") return resp;
              var patched = isProps ? patchPropsPayload(parsed) : patchModelsPayload(parsed);
              var h = new Headers(resp.headers);
              h.set("content-type", "application/json; charset=utf-8");
              return new Response(JSON.stringify(patched), {
                status: resp.status,
                statusText: resp.statusText,
                headers: h
              });
            }).catch(function () { return resp; });
          } catch (_err) {
            return resp;
          }
        });
      };
    }
  } catch (_err) {}
})();
</script>
"""

    if "id=\"moa-shim-preload\"" not in html:
        if "<script" in html:
            html = html.replace("<script", preload + "\n<script", 1)
        elif "</head>" in html:
            html = html.replace("</head>", preload + "\n</head>")
        else:
            html = preload + html

    if "id=\"moa-shim-toolbar\"" in html:
        return html

    script = r"""
<script id="moa-shim-toolbar">
(function () {
  if (window.__moaShimToolbarLoaded) return;
  window.__moaShimToolbarLoaded = true;

  // Self-heal stale llama.cpp WebUI capability cache so image/audio toggles recover automatically.
  (function healCapabilityCache() {
    var healFlag = "__moaShimCapHealV1";
    try {
      if (sessionStorage.getItem(healFlag) === "1") return;
    } catch (_err) {}

    var raw = null;
    try { raw = localStorage.getItem("LlamaCppWebui.serverProps"); } catch (_err) {}

    var parsed = null;
    if (raw) {
      try { parsed = JSON.parse(raw); } catch (_err) {}
    }

    var hasVision = !!(parsed && parsed.modalities && parsed.modalities.vision);
    var hasAudio = !!(parsed && parsed.modalities && parsed.modalities.audio);
    if (hasVision && hasAudio) return;

    try {
      var next = (parsed && typeof parsed === "object") ? parsed : {};
      next.modalities = { vision: true, audio: true };
      next.input_modalities = ["text", "image", "audio"];
      next.output_modalities = ["text", "audio"];
      next.capabilities = Object.assign({}, next.capabilities || {}, {
        vision: true,
        images: true,
        audio: true
      });
      localStorage.setItem("LlamaCppWebui.serverProps", JSON.stringify(next));
      localStorage.removeItem("LlamaCppWebui.selectedModel");
      sessionStorage.setItem(healFlag, "1");
      location.reload();
      return;
    } catch (_err) {}
  })();

  function mk(tag, attrs, text) {
    var el = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    if (text != null) el.textContent = text;
    return el;
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = mk("a", { href: url, download: filename || "conversation.md" });
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(url);
      a.remove();
    }, 0);
  }

  function parseContentDispositionFilename(cd) {
    if (!cd) return "";
    var m = /filename=\"?([^\";]+)\"?/i.exec(cd);
    return m ? m[1] : "";
  }

  function tryParseJson(text) {
    try { return JSON.parse(text); } catch (_err) { return null; }
  }

  function textFromContent(content) {
    if (content == null) return "";
    if (typeof content === "string") return content.trim();
    if (Array.isArray(content)) {
      return content.map(function (x) {
        if (typeof x === "string") return x;
        if (x && typeof x === "object") {
          if (typeof x.text === "string") return x.text;
          if (typeof x.content === "string") return x.content;
        }
        return "";
      }).join("\n").trim();
    }
    if (typeof content === "object") {
      if (typeof content.text === "string") return content.text.trim();
      if (typeof content.content === "string") return content.content.trim();
    }
    return String(content || "").trim();
  }

  function currentChatKey() {
    var p = "";
    var h = "";
    var t = "";
    try { p = String(location.pathname || ""); } catch (_err) {}
    try { h = String(location.hash || ""); } catch (_err) {}
    try { t = String(document.title || ""); } catch (_err) {}
    return [p, h, t].join("|");
  }

  function captureState() {
    if (!window.__moaShimCapture || typeof window.__moaShimCapture !== "object") {
      window.__moaShimCapture = {
        chat_key: currentChatKey(),
        name: (document.title || "conversation").trim() || "conversation",
        messages: [],
        updated_at: 0
      };
    }
    if (!Array.isArray(window.__moaShimCapture.messages)) {
      window.__moaShimCapture.messages = [];
    }
    return window.__moaShimCapture;
  }

  function normalizeMessagesFromRequest(payload) {
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return [];
    var out = [];
    for (var i = 0; i < payload.messages.length; i++) {
      var m = payload.messages[i];
      if (!m || typeof m !== "object") continue;
      var role = String(m.role || "").toLowerCase();
      if (role !== "user" && role !== "assistant" && role !== "system" && role !== "tool") continue;
      out.push({
        role: role,
        content: m.content == null ? "" : m.content,
        timestamp: Date.now()
      });
    }
    return out;
  }

  function captureFromRequestPayload(payload) {
    var state = captureState();
    var normalized = normalizeMessagesFromRequest(payload);
    if (!normalized.length) return;
    state.chat_key = currentChatKey();
    state.name = (document.title || "conversation").trim() || "conversation";
    state.messages = normalized;
    state.updated_at = Date.now();
  }

  function appendAssistantToCurrentChat(content) {
    if (content == null) return;
    var text = textFromContent(content);
    if (!text) return;
    var state = captureState();
    var activeKey = currentChatKey();
    if (state.chat_key !== activeKey) {
      // Refuse cross-chat writes; require a request from this chat first.
      return;
    }
    var list = state.messages || [];
    var msg = { role: "assistant", content: content, timestamp: Date.now() };
    var prev = list.length ? list[list.length - 1] : null;
    if (prev && prev.role === "assistant" && textFromContent(prev.content) === text) return;
    list.push(msg);
    state.messages = list;
    state.updated_at = Date.now();
  }

  function extractAssistantContentFromJson(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (Array.isArray(payload.choices) && payload.choices.length > 0) {
      var choice = payload.choices[0] || {};
      if (choice.message && choice.message.content != null) return choice.message.content;
      if (choice.delta && choice.delta.content != null) return choice.delta.content;
      if (choice.text != null) return choice.text;
    }
    if (payload.message && payload.message.content != null) return payload.message.content;
    if (payload.content != null) return payload.content;
    return null;
  }

  function parseSseAssistantText(raw) {
    if (typeof raw !== "string" || raw.length === 0) return "";
    var out = [];
    var lines = raw.split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line || line.indexOf("data:") !== 0) continue;
      var jsonPart = line.slice(5).trim();
      if (!jsonPart || jsonPart === "[DONE]") continue;
      var obj = tryParseJson(jsonPart);
      if (!obj) continue;
      if (Array.isArray(obj.choices)) {
        for (var j = 0; j < obj.choices.length; j++) {
          var ch = obj.choices[j] || {};
          if (ch.delta && ch.delta.content != null) out.push(String(ch.delta.content));
          else if (ch.message && ch.message.content != null) out.push(String(ch.message.content));
          else if (ch.text != null) out.push(String(ch.text));
        }
      } else if (obj.content != null) {
        out.push(String(obj.content));
      }
    }
    return out.join("");
  }

  function currentPayloadForExport() {
    var state = captureState();
    var activeKey = currentChatKey();
    if (state.chat_key !== activeKey) return null;
    if (!state.messages || state.messages.length === 0) return null;
    return { name: state.name || "conversation", messages: state.messages };
  }

  async function exportToMarkdown(payload) {
    var resp = await fetch("/shim/export/md", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error("export failed: " + resp.status);
    var blob = await resp.blob();
    var filename = parseContentDispositionFilename(resp.headers.get("content-disposition")) || "conversation.md";
    downloadBlob(blob, filename);
  }

  function pendingImagesStore() {
    if (!Array.isArray(window.__moaShimPendingImageUrls)) {
      window.__moaShimPendingImageUrls = [];
    }
    return window.__moaShimPendingImageUrls;
  }

  function stashPendingImageUrl(url) {
    var u = String(url || "").trim();
    if (!u) return;
    var low = u.toLowerCase();
    if (!(low.indexOf("data:image/") === 0 || low.indexOf("http://") === 0 || low.indexOf("https://") === 0)) return;
    var arr = pendingImagesStore();
    if (arr.indexOf(u) < 0) arr.push(u);
    if (arr.length > 4) arr.splice(0, arr.length - 4);
  }

  function sentImageUrlsStore() {
    if (!window.__moaShimSentImageUrls || typeof window.__moaShimSentImageUrls !== "object") {
      window.__moaShimSentImageUrls = {};
    }
    return window.__moaShimSentImageUrls;
  }

  function markImageUrlAsSent(url) {
    var u = String(url || "").trim();
    if (!u) return;
    sentImageUrlsStore()[u] = true;
  }

  function isImageUrlSent(url) {
    var u = String(url || "").trim();
    if (!u) return false;
    return !!sentImageUrlsStore()[u];
  }

  function payloadHasImageSignal(value) {
    if (value == null) return false;
    if (typeof value === "string") {
      var s = value.trim().toLowerCase();
      return s.indexOf("data:image/") === 0;
    }
    if (Array.isArray(value)) {
      for (var i = 0; i < value.length; i++) {
        if (payloadHasImageSignal(value[i])) return true;
      }
      return false;
    }
    if (typeof value !== "object") return false;
    var typ = String(value.type || "").toLowerCase();
    if (typ === "image" || typ === "image_url" || typ === "input_image") return true;
    var mime = String(value.mime_type || value.content_type || value.mimetype || "").toLowerCase();
    if (mime.indexOf("image/") === 0) return true;
    var iu = value.image_url;
    if (typeof iu === "string" && payloadHasImageSignal(iu)) return true;
    if (iu && typeof iu === "object" && payloadHasImageSignal(iu.url)) return true;
    var keys = ["url", "uri", "src", "data", "file", "files", "image", "images", "attachments", "content", "parts"];
    for (var k = 0; k < keys.length; k++) {
      var key = keys[k];
      if (Object.prototype.hasOwnProperty.call(value, key) && payloadHasImageSignal(value[key])) return true;
    }
    return false;
  }

  function normalizeUserContentToBlocks(content) {
    if (Array.isArray(content)) {
      var out = [];
      for (var i = 0; i < content.length; i++) {
        var b = content[i];
        if (b && typeof b === "object") out.push(b);
        else if (typeof b === "string" && b.trim()) out.push({ type: "text", text: b });
      }
      return out;
    }
    if (typeof content === "string") {
      var t = content.trim();
      return t ? [{ type: "text", text: t }] : [];
    }
    if (content && typeof content === "object") return [content];
    return [];
  }

  function injectPendingImagesIntoPayload(payload) {
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return false;
    if (payloadHasImageSignal(payload)) return false;
    var pending = pendingImagesStore();
    if (!pending.length) return false;

    var idx = -1;
    for (var i = payload.messages.length - 1; i >= 0; i--) {
      var m = payload.messages[i];
      if (m && typeof m === "object" && String(m.role || "").toLowerCase() === "user") { idx = i; break; }
    }
    if (idx < 0) {
      payload.messages.push({ role: "user", content: "" });
      idx = payload.messages.length - 1;
    }

    var msg = payload.messages[idx] || {};
    var blocks = normalizeUserContentToBlocks(msg.content);
    var hasText = false;
    for (var j = 0; j < blocks.length; j++) {
      var bt = String((blocks[j] && blocks[j].type) || "").toLowerCase();
      if ((bt === "text" || bt === "input_text") && String((blocks[j] && blocks[j].text) || "").trim()) { hasText = true; break; }
    }
    if (!hasText) blocks.push({ type: "text", text: "What is in this image?" });

    for (var p = 0; p < pending.length; p++) {
      blocks.push({ type: "image_url", image_url: { url: pending[p] } });
    }

    msg.content = blocks;
    payload.messages[idx] = msg;
    window.__moaShimPendingImageUrls = [];
    return true;
  }

  function readFileAsDataUrl(file) {
    return new Promise(function (resolve) {
      try {
        var reader = new FileReader();
        reader.onload = function () { resolve(String(reader.result || "")); };
        reader.onerror = function () { resolve(""); };
        reader.readAsDataURL(file);
      } catch (_err) {
        resolve("");
      }
    });
  }

  async function captureImagesFromFormData(fd) {
    if (!fd || typeof fd.forEach !== "function") return;
    var tasks = [];
    fd.forEach(function (val) {
      if (typeof val === "string") {
        stashPendingImageUrl(val);
        return;
      }
      if (typeof File !== "undefined" && val instanceof File) {
        var mime = String(val.type || "").toLowerCase();
        if (mime.indexOf("image/") === 0) tasks.push(readFileAsDataUrl(val));
      }
    });
    if (!tasks.length) return;
    var urls = await Promise.all(tasks);
    for (var i = 0; i < urls.length; i++) stashPendingImageUrl(urls[i]);
  }

  function blobToDataUrl(blob) {
    return new Promise(function (resolve) {
      try {
        var reader = new FileReader();
        reader.onload = function () { resolve(String(reader.result || "")); };
        reader.onerror = function () { resolve(""); };
        reader.readAsDataURL(blob);
      } catch (_err) {
        resolve("");
      }
    });
  }

  async function collectImagesFromDomFallback() {
    try {
      var pending = pendingImagesStore();
      if (pending.length > 0) return;
      if (!document || !document.querySelectorAll) return;

      var imgs = document.querySelectorAll('img[src^="data:image/"], img[src^="blob:"]');
      if (!imgs || !imgs.length) return;

      var seen = {};
      for (var i = 0; i < imgs.length; i++) {
        var src = String((imgs[i] && imgs[i].getAttribute("src")) || "").trim();
        if (!src || seen[src]) continue;
        seen[src] = true;

        var low = src.toLowerCase();
        if (low.indexOf("data:image/") === 0) {
          if (isImageUrlSent(src)) continue;
          stashPendingImageUrl(src);
          continue;
        }

        if (low.indexOf("blob:") === 0) {
          try {
            var r = await originalFetch(src);
            if (!r || !r.ok) continue;
            var b = await r.blob();
            var mt = String((b && b.type) || "").toLowerCase();
            if (mt.indexOf("image/") !== 0) continue;
            var durl = await blobToDataUrl(b);
            if (isImageUrlSent(durl)) continue;
            stashPendingImageUrl(durl);
          } catch (_err) {}
        }
      }
    } catch (_err) {}
  }

  (function patchXhrForImageCapture() {
    try {
      if (window.__moaShimXhrPatched) return;
      window.__moaShimXhrPatched = true;
      if (typeof XMLHttpRequest === "undefined" || !XMLHttpRequest.prototype) return;
      var origSend = XMLHttpRequest.prototype.send;
      if (!origSend) return;
      XMLHttpRequest.prototype.send = function (body) {
        try {
          if (typeof FormData !== "undefined" && body instanceof FormData) {
            captureImagesFromFormData(body);
          }
        } catch (_err) {}
        return origSend.call(this, body);
      };
    } catch (_err) {}
  })();

  var originalFetch = window.fetch ? window.fetch.bind(window) : null;
  if (originalFetch && !window.__moaShimFetchPatched) {
    window.__moaShimFetchPatched = true;
    window.fetch = async function (input, init) {
      var url = "";
      var method = "GET";
      try {
        url = typeof input === "string" ? input : (input && input.url) || "";
        method = ((init && init.method) || (typeof input !== "string" && input && input.method) || "GET").toUpperCase();
      } catch (_err) {}

      var isChatCall = /\/v1\/chat\/completions(\?|$)/.test(url) ||
        /\/completions(\?|$)/.test(url) ||
        /\/completion(\?|$)/.test(url);
      var reqPayload = null;
      async function readRequestPayload() {
        try {
          if (init && typeof init.body === "string") return tryParseJson(init.body);
        } catch (_err) {}
        try {
          if (typeof input !== "string" && input && typeof input.clone === "function") {
            var txt = await input.clone().text();
            if (txt && txt.length) return tryParseJson(txt);
          }
        } catch (_err) {}
        return null;
      }

      try {
        if (init && init.body && typeof FormData !== "undefined" && init.body instanceof FormData) {
          await captureImagesFromFormData(init.body);
        } else if (typeof input !== "string" && input && typeof input.clone === "function") {
          var maybeReq = input.clone();
          var reqCt = String((maybeReq.headers && maybeReq.headers.get && maybeReq.headers.get("content-type")) || "").toLowerCase();
          if (reqCt.indexOf("multipart/form-data") >= 0 && typeof maybeReq.formData === "function") {
            var fd = await maybeReq.formData();
            await captureImagesFromFormData(fd);
          }
        }
      } catch (_err) {}

      if (isChatCall && method === "POST") {
        try { await collectImagesFromDomFallback(); } catch (_err) {}
      }

      if (isChatCall && method === "POST") {
        try {
          reqPayload = await readRequestPayload();
          if (!init || typeof init !== "object") init = {};
          var dbgHdr = null;
          try {
            dbgHdr = new Headers((init && init.headers) || ((typeof input !== "string" && input && input.headers) || {}));
          } catch (_err) {
            dbgHdr = new Headers();
          }
          var preInjectUrls = pendingImagesStore().slice();
          dbgHdr.set("x-moa-shim-image-injected", "0");
          dbgHdr.set("x-moa-shim-pending-images", String(preInjectUrls.length));
          if (reqPayload && injectPendingImagesIntoPayload(reqPayload)) {
            init.method = "POST";
            init.body = JSON.stringify(reqPayload);
            dbgHdr.set("content-type", "application/json");
            dbgHdr.set("x-moa-shim-image-injected", "1");
            dbgHdr.set("x-moa-shim-pending-images", String(pendingImagesStore().length));
            for (var si = 0; si < preInjectUrls.length; si++) {
              markImageUrlAsSent(preInjectUrls[si]);
            }
          }
          init.headers = dbgHdr;
          captureFromRequestPayload(reqPayload);
        } catch (_err) {}
      }

      var response = await originalFetch(input, init);
      if (isChatCall && method === "POST" && response && response.ok) {
        (async function (resp) {
          try {
            var clone = resp.clone();
            var contentType = String(clone.headers.get("content-type") || "").toLowerCase();
            var assistant = null;

            if (contentType.indexOf("text/event-stream") >= 0) {
              var sseText = await clone.text();
              assistant = parseSseAssistantText(sseText);
            } else {
              var rawText = await clone.text();
              var parsed = tryParseJson(rawText);
              assistant = extractAssistantContentFromJson(parsed);
              if ((assistant == null || assistant === "") && rawText && rawText.indexOf("data:") >= 0) {
                assistant = parseSseAssistantText(rawText);
              }
            }

            if (assistant != null && String(assistant).trim() !== "") appendAssistantToCurrentChat(assistant);
          } catch (_err) {}
        })(response);
      }
      return response;
    };
  }

  var bar = mk("div", { id: "moa-shim-toolbar" });
  bar.style.position = "fixed";
  bar.style.right = "12px";
  bar.style.bottom = "12px";
  bar.style.zIndex = "2147483647";
  bar.style.display = "flex";
  bar.style.alignItems = "center";
  bar.style.gap = "8px";
  bar.style.padding = "8px 10px";
  bar.style.borderRadius = "12px";
  bar.style.border = "1px solid rgba(255,255,255,0.18)";
  bar.style.background = "rgba(16,18,20,0.85)";
  bar.style.color = "#e7ebf0";
  bar.style.fontFamily = "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
  bar.style.fontSize = "12px";
  bar.style.backdropFilter = "blur(6px)";

  var btn = mk("button", { type: "button" }, "EXPORT MD");
  btn.style.cursor = "pointer";
  btn.style.border = "1px solid rgba(255,255,255,0.22)";
  btn.style.borderRadius = "10px";
  btn.style.padding = "4px 9px";
  btn.style.background = "rgba(255,255,255,0.06)";
  btn.style.color = "#fff";
  btn.style.fontWeight = "600";

  bar.appendChild(btn);
  document.body.appendChild(bar);

  btn.addEventListener("click", async function () {
    btn.disabled = true;
    btn.style.opacity = "0.7";
    try {
      var payload = currentPayloadForExport();
      if (!payload || !Array.isArray(payload.messages) || payload.messages.length === 0) {
        alert("No current-chat capture found. Send one message in this chat, then try EXPORT MD again.");
        return;
      }
      await exportToMarkdown(payload);
    } catch (err) {
      console.error("EXPORT MD failed:", err);
      alert("EXPORT MD failed. Open console for details.");
    } finally {
      btn.disabled = false;
      btn.style.opacity = "1";
    }
  });

})();
</script>
"""
    if "</body>" in html:
        return html.replace("</body>", script + "\n</body>")
    return html + script


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = str(item.get("type", "")).lower()
                if typ in {"text", "input_text"}:
                    parts.append(str(item.get("text", item.get("content", ""))))
                elif typ in {"image", "image_url", "input_image"}:
                    parts.append("[image]")
                elif typ in {"code", "codeblock", "code_block"}:
                    lang = str(item.get("language", item.get("lang", ""))).strip()
                    code = str(item.get("text", item.get("content", item.get("code", ""))))
                    parts.append(f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```")
                else:
                    parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p is not None]).strip()
    if isinstance(content, dict):
        typ = str(content.get("type", "")).lower()
        if typ in {"image", "image_url", "input_image"}:
            return "[image]"
        if typ in {"code", "codeblock", "code_block"}:
            lang = str(content.get("language", content.get("lang", ""))).strip()
            code = str(content.get("text", content.get("content", content.get("code", ""))))
            return f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"
        return str(content.get("text", content.get("content", "")))
    return str(content)


def _extract_current_branch(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []

    conv = payload.get("conv")
    curr_node = conv.get("currNode") if isinstance(conv, dict) else None
    if not curr_node:
        return [m for m in messages if isinstance(m, dict)]

    by_id: Dict[str, Dict[str, Any]] = {}
    for m in messages:
        if isinstance(m, dict):
            mid = str(m.get("id", "")).strip()
            if mid:
                by_id[mid] = m

    if curr_node not in by_id:
        return [m for m in messages if isinstance(m, dict)]

    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    cur = curr_node
    while cur and cur in by_id and cur not in seen:
        seen.add(cur)
        msg = by_id[cur]
        ordered.append(msg)
        parent = str(msg.get("parent", "")).strip()
        cur = parent if parent else ""

    ordered.reverse()
    return ordered


def _coerce_messages(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return _extract_current_branch(payload)
        if isinstance(payload.get("data"), list):
            return [m for m in payload["data"] if isinstance(m, dict)]
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    return []


def _pick_title(payload: Any) -> str:
    if isinstance(payload, dict):
        conv = payload.get("conv")
        if isinstance(conv, dict):
            name = str(conv.get("name", "")).strip()
            if name:
                return name
        name = str(payload.get("name", "")).strip()
        if name:
            return name
    return "conversation"


def _slug(name: str, fallback: str = "conversation") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "-", str(name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    cleaned = re.sub(r"\.+", ".", cleaned)
    if not cleaned:
        cleaned = fallback
    return cleaned[:96]


def _messages_to_markdown(payload: Any) -> tuple[str, str]:
    messages = _coerce_messages(payload)
    title = _pick_title(payload)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = [f"# {title}", "", f"*{stamp}*", "", "---", ""]

    for m in messages:
        role = str(m.get("role", "assistant")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "assistant"
        role_h = role.capitalize()
        content = _flatten_content(m.get("content"))
        content = content.strip() or "(empty)"

        lines.append(f"## {role_h}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    if len(messages) == 0:
        lines.append("(No messages found in payload.)")
        lines.append("")

    filename = _slug(title) + ".md"
    return "\n".join(lines).rstrip() + "\n", filename


@APP.post("/shim/export/md")
async def export_markdown(request: Request):
    payload = await request.json()
    md_text, filename = _messages_to_markdown(payload)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=md_text, media_type="text/markdown; charset=utf-8", headers=headers)


@APP.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    path = "/" + path
    request_body = await request.body()
    request_headers = _safe_headers(dict(request.headers))
    method_u = request.method.upper()
    content_type_l = str(request.headers.get("content-type", "") or "").lower()

    if _SHIM_IMAGE_EMITTER_ENABLED and method_u in {"POST", "PUT", "PATCH"} and not _is_chat_path(path):
        payload_probe = _decode_json_body(request_body)
        _shim_emit_image_trace(
            "shim_probe_nonchat",
            {
                "path": path,
                "method": method_u,
                "content_type": request.headers.get("content-type", ""),
                "body_len": len(request_body or b""),
                "json_keys": sorted([str(k) for k in payload_probe.keys()]) if isinstance(payload_probe, dict) else [],
                "has_image_signal": bool(_shim_has_image_signal(payload_probe)) if isinstance(payload_probe, dict) else False,
                "body_prefix": (request_body[:180].decode("utf-8", errors="replace") if request_body else ""),
            },
        )

    if method_u == "POST" and _is_chat_path(path):
        payload_in = _decode_json_body(request_body)
        if payload_in:
            msgs_in = payload_in.get("messages")
            if not isinstance(msgs_in, list):
                msgs_in = []
            sid = (
                request.headers.get("x-chat-id")
                or request.headers.get("x-session-id")
                or request.headers.get("x-openwebui-chat-id")
                or str(payload_in.get("chat_id") or payload_in.get("conversation_id") or payload_in.get("session_id") or payload_in.get("user") or "").strip()
            )
            _shim_emit_image_trace(
                "shim_ingress",
                {
                    "path": path,
                    "session_id": sid,
                    "stream": bool(payload_in.get("stream")),
                    "content_type": request.headers.get("content-type", ""),
                    "client_image_injected": request.headers.get("x-moa-shim-image-injected", ""),
                    "client_pending_images": request.headers.get("x-moa-shim-pending-images", ""),
                    "body_keys": sorted([str(k) for k in payload_in.keys()]),
                    "request_has_image_signal": bool(_shim_has_image_signal(payload_in)),
                    "top_level_counts": {
                        "messages": len(msgs_in),
                        "files": len(payload_in.get("files") or []) if isinstance(payload_in.get("files"), list) else int(bool(payload_in.get("files"))),
                        "attachments": len(payload_in.get("attachments") or []) if isinstance(payload_in.get("attachments"), list) else int(bool(payload_in.get("attachments"))),
                        "images": len(payload_in.get("images") or []) if isinstance(payload_in.get("images"), list) else int(bool(payload_in.get("images"))),
                    },
                    "last_user": _shim_last_user_summary(msgs_in),
                    "probe_paths": _shim_collect_probe_paths(payload_in),
                },
            )
        else:
            _shim_emit_image_trace(
                "shim_ingress_nonjson",
                {
                    "path": path,
                    "content_type": request.headers.get("content-type", ""),
                    "body_len": len(request_body or b""),
                    "body_prefix": (request_body[:160].decode("utf-8", errors="replace") if request_body else ""),
                },
            )

    # Some llama.cpp WebUI builds call /models/load and /models/unload,
    # but current llama-server may not expose these routes. Return success
    # so UI model-switch flow does not break on 404.
    if request.method.upper() == "POST" and path in {"/models/load", "/models/unload"}:
        payload = _decode_json_body(request_body)
        selected = str(
            payload.get("model")
            or payload.get("id")
            or CONFIG.ui_model_alias
            or CONFIG.router_model_id
            or "moa-router"
        ).strip()
        status = "loaded" if path == "/models/load" else "unloaded"
        return JSONResponse(
            content={
                "ok": True,
                "status": status,
                "id": selected,
                "model": selected,
                "object": "model",
            },
            status_code=200,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    timeout = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=120.0)
    request_body = await _rewrite_chat_model_for_forwarding(path, request_body, timeout)
    target_url = _build_target_url(request, path, request_body)
    if method_u == "POST" and _is_chat_path(path):
        payload_out = _decode_json_body(request_body)
        if payload_out:
            msgs_out = payload_out.get("messages")
            if not isinstance(msgs_out, list):
                msgs_out = []
            _shim_emit_image_trace(
                "shim_post_rewrite",
                {
                    "path": path,
                    "target_url": target_url,
                    "model": str(payload_out.get("model") or ""),
                    "request_has_image_signal": bool(_shim_has_image_signal(payload_out)),
                    "top_level_counts": {
                        "messages": len(msgs_out),
                        "files": len(payload_out.get("files") or []) if isinstance(payload_out.get("files"), list) else int(bool(payload_out.get("files"))),
                        "attachments": len(payload_out.get("attachments") or []) if isinstance(payload_out.get("attachments"), list) else int(bool(payload_out.get("attachments"))),
                        "images": len(payload_out.get("images") or []) if isinstance(payload_out.get("images"), list) else int(bool(payload_out.get("images"))),
                    },
                    "last_user": _shim_last_user_summary(msgs_out),
                },
            )
        else:
            _shim_emit_image_trace(
                "shim_post_rewrite_nonjson",
                {
                    "path": path,
                    "target_url": target_url,
                    "body_len": len(request_body or b""),
                },
            )
    accept_header = request.headers.get("accept", "")
    stream_mode = _should_stream(path, request_body, accept_header)

    try:
        if stream_mode and CONFIG.render_mode == "buffered" and _is_chat_path(path):
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                upstream = await client.request(
                    request.method,
                    target_url,
                    headers=request_headers,
                    content=request_body,
                )
                response_headers = _safe_headers(dict(upstream.headers))
                ctype = (upstream.headers.get("content-type") or "").lower()
                if "text/event-stream" in ctype:
                    buffered_sse = _sse_to_buffered_sse(upstream.text)
                    return Response(
                        content=buffered_sse.encode("utf-8"),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type="text/event-stream; charset=utf-8",
                    )
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type=upstream.headers.get("content-type"),
                )

        if stream_mode:
            client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
            stream_ctx = client.stream(
                request.method,
                target_url,
                headers=request_headers,
                content=request_body,
            )
            upstream = await stream_ctx.__aenter__()
            response_headers = _safe_headers(dict(upstream.headers))

            async def iterator():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()

            return StreamingResponse(
                iterator(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
            )

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            upstream = await client.request(
                request.method,
                target_url,
                headers=request_headers,
                content=request_body,
            )

            content_type = (upstream.headers.get("content-type") or "").lower()

            if path in {"/props", "/v1/models"} and "application/json" in content_type:
                try:
                    patched = _patch_json_response(path, upstream.json())
                    response_headers = _safe_headers(dict(upstream.headers))
                    # Prevent stale capability metadata from being cached by the browser.
                    response_headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                    response_headers["Pragma"] = "no-cache"
                    response_headers["Expires"] = "0"
                    return JSONResponse(
                        content=patched,
                        status_code=upstream.status_code,
                        headers=response_headers,
                    )
                except ValueError:
                    pass

            if path in {"/", "/index.html"} and "text/html" in content_type:
                try:
                    raw = upstream.content.decode("utf-8", errors="replace")
                    injected = _inject_toolbar(raw)
                    response_headers = _safe_headers(dict(upstream.headers))
                    response_headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                    response_headers["Pragma"] = "no-cache"
                    response_headers["Expires"] = "0"
                    return Response(
                        content=injected.encode("utf-8"),
                        status_code=upstream.status_code,
                        headers=response_headers,
                        media_type=upstream.headers.get("content-type"),
                    )
                except Exception:
                    pass

            response_headers = _safe_headers(dict(upstream.headers))
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
            )
    except httpx.RequestError as exc:
        help_text = (
            "Check shim upstream settings and confirm llama-server is reachable "
            f"at {CONFIG.upstream_base.rstrip('/') or 'http://127.0.0.1:8010'}"
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": True,
                "message": "Upstream request failed.",
                "target_url": target_url,
                "details": str(exc),
                "hint": help_text,
            },
        )


if __name__ == "__main__":
    import uvicorn

    def _find_listener_pids(port: int) -> List[int]:
        found: List[int] = []
        port_i = int(port)

        # Preferred path: psutil gives cross-platform listener->pid mapping.
        try:
            import psutil  # type: ignore

            pids = set()
            for conn in psutil.net_connections(kind="inet"):
                laddr = getattr(conn, "laddr", None)
                cpid = getattr(conn, "pid", None)
                status = str(getattr(conn, "status", "")).upper()
                if not laddr or not cpid:
                    continue
                if int(getattr(laddr, "port", -1)) != port_i:
                    continue
                if status != "LISTEN":
                    continue
                pids.add(int(cpid))
            found = sorted(pids)
            if found:
                return found
        except Exception:
            pass

        # Fallback path for environments without psutil.
        if os.name == "nt":
            try:
                out = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                pids = set()
                for line in out.splitlines():
                    if f":{port_i}" not in line:
                        continue
                    if "LISTENING" not in line.upper():
                        continue
                    cols = line.split()
                    if not cols:
                        continue
                    pid_raw = cols[-1].strip()
                    if pid_raw.isdigit():
                        pids.add(int(pid_raw))
                found = sorted(pids)
            except Exception:
                found = []
        else:
            try:
                out = subprocess.run(
                    ["lsof", "-nP", f"-iTCP:{port_i}", "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                pids = set()
                for line in out.splitlines():
                    raw = line.strip()
                    if raw.isdigit():
                        pids.add(int(raw))
                found = sorted(pids)
                if found:
                    return found
            except Exception:
                pass
            try:
                out = subprocess.run(
                    ["ss", "-ltnp", f"sport = :{port_i}"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                pids = set()
                for match in re.finditer(r"pid=(\d+)", out or ""):
                    pids.add(int(match.group(1)))
                found = sorted(pids)
            except Exception:
                found = []
        return found

    def _local_port_is_listening(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.35)
            return sock.connect_ex(("127.0.0.1", int(port))) == 0
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _existing_is_this_shim(port: int) -> bool:
        try:
            with httpx.Client(timeout=0.6, follow_redirects=False) as client:
                resp = client.get(f"http://127.0.0.1:{int(port)}/shim/healthz")
                if resp.status_code != 200:
                    return False
                payload = resp.json() if "application/json" in (resp.headers.get("content-type") or "").lower() else {}
                return bool(isinstance(payload, dict) and payload.get("ok") is True)
        except Exception:
            return False

    def _terminate_pid(pid: int) -> bool:
        if os.name == "nt":
            try:
                proc = subprocess.run(
                    ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return proc.returncode == 0
            except Exception:
                return False
        try:
            os.kill(int(pid), signal.SIGTERM)
            return True
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except Exception:
            return False

    def _take_over_existing_instance(port: int) -> bool:
        pids = _find_listener_pids(port)
        if not pids:
            return not _local_port_is_listening(port)

        for pid in pids:
            _terminate_pid(pid)

        deadline = time.monotonic() + max(0.5, float(CONFIG.takeover_timeout_s or 5.0))
        while time.monotonic() < deadline:
            if not _local_port_is_listening(port):
                return True
            time.sleep(0.1)
        return not _local_port_is_listening(port)

    def _format_pid_list(port: int) -> str:
        pids = _find_listener_pids(port)
        if not pids:
            return "(unknown)"
        return ", ".join(str(pid) for pid in pids)

    def _print_port_blocked_help(port: int) -> None:
        pids = _format_pid_list(port)
        print(f"[moa-shim] Port {port} is busy (PID(s): {pids}).")
        if os.name == "nt":
            print(f"[moa-shim] Fix: run `netstat -ano | findstr :{port}` then `taskkill /PID <pid> /T /F`.")
        else:
            print(f"[moa-shim] Fix: run `lsof -nP -iTCP:{port} -sTCP:LISTEN` then kill stale PID(s).")

    if CONFIG.single_instance and _local_port_is_listening(CONFIG.port):
        if _existing_is_this_shim(CONFIG.port):
            if CONFIG.takeover_existing:
                print(f"[moa-shim] found existing instance on 127.0.0.1:{CONFIG.port}; taking over...")
                if not _take_over_existing_instance(CONFIG.port):
                    print(f"[moa-shim] takeover failed: port {CONFIG.port} still busy.")
                    _print_port_blocked_help(CONFIG.port)
                    sys.exit(1)
                print(f"[moa-shim] takeover complete; starting fresh instance.")
            else:
                print(f"[moa-shim] already running on 127.0.0.1:{CONFIG.port}; exiting duplicate launch.")
                sys.exit(0)
        elif _local_port_is_listening(CONFIG.port):
            if CONFIG.takeover_existing and CONFIG.takeover_foreign:
                print(f"[moa-shim] port {CONFIG.port} is in use by non-responsive process; forcing takeover...")
                if not _take_over_existing_instance(CONFIG.port):
                    print(f"[moa-shim] forced takeover failed: port {CONFIG.port} still busy.")
                    _print_port_blocked_help(CONFIG.port)
                    sys.exit(1)
                print(f"[moa-shim] forced takeover complete; starting fresh instance.")
            else:
                print(f"[moa-shim] port {CONFIG.port} is already in use by another process; aborting.")
                _print_port_blocked_help(CONFIG.port)
                sys.exit(1)

    try:
        uvicorn.run("shim_server:APP", host=CONFIG.host, port=CONFIG.port, reload=False)
    except OSError as exc:
        if "10048" in str(exc) or "Address already in use" in str(exc):
            _print_port_blocked_help(CONFIG.port)
        raise

