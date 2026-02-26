from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

VODKA_PRESET_MAP: Dict[str, Dict[str, Any]] = {
    "fast": {
        "n_last_messages": 8,
        "keep_first": False,
        "max_chars": 4000,
        "enable_summary": True,
        "summary_every_n_user_msgs": 6,
        "summary_inject_max_units": 2,
        "summary_inject_max_chars": 380,
        "summary_memory_max_units": 72,
        "summary_segments_per_message": 5,
    },
    "balanced": {
        "n_last_messages": 12,
        "keep_first": False,
        "max_chars": 6000,
        "enable_summary": True,
        "summary_every_n_user_msgs": 4,
        "summary_inject_max_units": 3,
        "summary_inject_max_chars": 500,
        "summary_memory_max_units": 96,
        "summary_segments_per_message": 8,
    },
    "max-recall": {
        "n_last_messages": 16,
        "keep_first": True,
        "max_chars": 8000,
        "enable_summary": True,
        "summary_every_n_user_msgs": 3,
        "summary_inject_max_units": 5,
        "summary_inject_max_chars": 900,
        "summary_memory_max_units": 160,
        "summary_segments_per_message": 10,
    },
}


def normalize_vodka_preset_name(name: str, preset_map: Dict[str, Dict[str, Any]]) -> str:
    n = (name or "").strip().lower().replace("_", "-")
    if n in ("maxrecall", "max recall"):
        return "max-recall"
    if n not in preset_map:
        return "balanced"
    return n


def preset_for_session(state: Any, vodka_cfg: Dict[str, Any], preset_map: Dict[str, Dict[str, Any]]) -> str:
    session_override = str(getattr(state, "vodka_preset_override", "") or "").strip()
    cfg_default = str(vodka_cfg.get("preset", "balanced") or "balanced").strip()
    return normalize_vodka_preset_name(session_override or cfg_default, preset_map)


def apply_vodka_runtime(
    *,
    state: Any,
    raw_messages: List[Dict[str, Any]],
    session_id: str,
    cfg_get: Callable[[str, Any], Any],
    VodkaFilter: Any,
    preset_for_session: Callable[[Any, Dict[str, Any]], str],
    preset_map: Dict[str, Dict[str, Any]],
    debug_fn: Callable[[str], None],
) -> Tuple[Any, List[Dict[str, Any]], str]:
    """Initialize/configure vodka, apply inlet/outlet once, return noop wrapper."""
    if state.vodka is None:
        state.vodka = VodkaFilter()
    vodka = state.vodka

    vodka_cfg = dict(cfg_get("vodka", {}))
    try:
        session_override = str(getattr(state, "vodka_preset_override", "") or "").strip()
        preset_name = preset_for_session(state, vodka_cfg)
        preset_cfg = dict(preset_map.get(preset_name, preset_map["balanced"]))
        if session_override:
            merged_vodka_cfg = dict(vodka_cfg)
            merged_vodka_cfg.update(preset_cfg)
            merged_vodka_cfg["preset"] = preset_name
        else:
            merged_vodka_cfg = dict(preset_cfg)
            merged_vodka_cfg.update(vodka_cfg)

        vodka.valves.storage_dir = str(merged_vodka_cfg.get("storage_dir", vodka.valves.storage_dir) or vodka.valves.storage_dir)
        vodka.valves.base_ttl_days = int(merged_vodka_cfg.get("base_ttl_days", vodka.valves.base_ttl_days))
        vodka.valves.touch_extension_days = int(merged_vodka_cfg.get("touch_extension_days", vodka.valves.touch_extension_days))
        user_max_touches = int(merged_vodka_cfg.get("max_touches", vodka.valves.max_touches))
        vodka.valves.max_touches = min(max(0, user_max_touches), 3)
        vodka.valves.debug = bool(merged_vodka_cfg.get("debug", vodka.valves.debug))
        vodka.valves.debug_dir = str(merged_vodka_cfg.get("debug_dir", vodka.valves.debug_dir) or vodka.valves.debug_dir)
        vodka.valves.n_last_messages = int(merged_vodka_cfg.get("n_last_messages", vodka.valves.n_last_messages))
        vodka.valves.keep_first = bool(merged_vodka_cfg.get("keep_first", vodka.valves.keep_first))
        vodka.valves.max_chars = int(merged_vodka_cfg.get("max_chars", vodka.valves.max_chars))
        vodka.valves.enable_summary = bool(merged_vodka_cfg.get("enable_summary", vodka.valves.enable_summary))
        vodka.valves.summary_every_n_user_msgs = int(
            merged_vodka_cfg.get("summary_every_n_user_msgs", vodka.valves.summary_every_n_user_msgs)
        )
        vodka.valves.summary_max_words = int(merged_vodka_cfg.get("summary_max_words", vodka.valves.summary_max_words))
        vodka.valves.summary_inject_max_units = int(
            merged_vodka_cfg.get("summary_inject_max_units", vodka.valves.summary_inject_max_units)
        )
        vodka.valves.summary_inject_max_chars = int(
            merged_vodka_cfg.get("summary_inject_max_chars", vodka.valves.summary_inject_max_chars)
        )
        vodka.valves.summary_memory_max_units = int(
            merged_vodka_cfg.get("summary_memory_max_units", vodka.valves.summary_memory_max_units)
        )
        vodka.valves.summary_segments_per_message = int(
            merged_vodka_cfg.get("summary_segments_per_message", vodka.valves.summary_segments_per_message)
        )
        vodka.valves.summary_require_session_id = bool(
            merged_vodka_cfg.get("summary_require_session_id", vodka.valves.summary_require_session_id)
        )
        vodka.valves.summary_include_assistant = bool(
            merged_vodka_cfg.get("summary_include_assistant", vodka.valves.summary_include_assistant)
        )
        vodka.valves.summary_store_pii = bool(merged_vodka_cfg.get("summary_store_pii", vodka.valves.summary_store_pii))
        vodka.valves.summary_pii_redaction_token = str(
            merged_vodka_cfg.get("summary_pii_redaction_token", vodka.valves.summary_pii_redaction_token)
            or vodka.valves.summary_pii_redaction_token
        )
    except Exception:
        pass

    vodka_body = {"messages": raw_messages, "session_id": session_id}
    try:
        vodka_body = vodka.inlet(vodka_body)
        vodka_body = vodka.outlet(vodka_body)
        raw_messages = vodka_body.get("messages", raw_messages)
    except Exception as e:
        debug_fn(f"[DEBUG] Vodka fail-open: {e.__class__.__name__}: {e}")

    class _NoOpVodka:
        def inlet(self, body: dict, user: dict | None = None) -> dict:
            return body

        def outlet(self, body: dict, user: dict | None = None) -> dict:
            return body

    return vodka, raw_messages, _NoOpVodka
