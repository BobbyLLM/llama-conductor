# config.py
"""Configuration management for MoA Router."""

import os
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
INSTALL_ROOT = os.path.abspath(os.path.join(HERE, ".."))
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


def cfg_get(path: str, default: Any) -> Any:
    """Get config value by dot-separated path (e.g., 'server.host')."""
    cur: Any = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# Load config on import
load_config(DEFAULT_CONFIG_PATH)

# Export commonly used config values
UPSTREAM_CHAT_URL: str = str(
    cfg_get(
        "backend.upstream_chat_url",
        cfg_get(
            "upstream_chat_url",
            cfg_get("llama_swap_url", "http://127.0.0.1:8011/v1/chat/completions"),
        ),
    )
)
# Back-compat alias retained for modules not yet migrated.
LLAMA_SWAP_URL: str = UPSTREAM_CHAT_URL
ROLES: Dict[str, str] = dict(cfg_get("roles", {}))


def _resolve_kb_paths(raw: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kb, path in dict(raw or {}).items():
        key = str(kb or "").strip()
        val = str(path or "").strip()
        if not key or not val:
            continue
        norm = val.replace("\\", "/")
        if os.path.isabs(val):
            out[key] = os.path.abspath(val)
            continue
        if norm.lower().startswith("docz/"):
            out[key] = os.path.abspath(os.path.join(INSTALL_ROOT, val))
            continue
        out[key] = os.path.abspath(val)
    return out


KB_PATHS: Dict[str, str] = _resolve_kb_paths(dict(cfg_get("kb_paths", {})))
VAULT_KB_NAME: str = str(cfg_get("vault_kb_name", "vault"))

# File paths
QUOTES_MD_PATH = os.path.join(HERE, "quotes.md")
CHEAT_SHEET_PATH = os.path.join(HERE, "command_cheat_sheet.md")

# fs-rag defaults
FS_TOP_K = int(cfg_get("fs_rag.top_k", 8))
FS_MAX_CHARS = int(cfg_get("fs_rag.max_chars", 2400))

# vault promotion defaults (chunking)
VAULT_CHUNK_WORDS = int(cfg_get("vault.chunk_words", 600))
VAULT_OVERLAP_WORDS = int(cfg_get("vault.chunk_overlap_words", 175))

# Runtime debug/privacy flags
MENTATS_DEBUG = bool(cfg_get("mentats.debug", False))
MENTATS_DEBUG_PAYLOAD = bool(cfg_get("mentats.debug_payload", False))
ROUTER_DEBUG = bool(cfg_get("router.debug", False))
ROUTER_DEBUG_LOG_USER_TEXT = bool(cfg_get("router.debug_log_user_text", False))

