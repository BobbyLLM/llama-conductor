from __future__ import annotations

import re
from typing import Any, Dict, Optional


def soft_alias_command(
    *,
    text: str,
    state: Any,
    is_command_fn,
    kb_paths: Dict[str, str],
    vault_kb_name: str,
) -> Optional[str]:
    """Optional bare-text aliases for ergonomics; behavior-preserving extraction."""
    t = (text or "").strip()
    if not t:
        return None
    if is_command_fn(t):
        return None
    if t.startswith("??") or t.startswith("!!") or t.startswith("##"):
        return None

    low = t.lower()
    if low == "status":
        return ">>status"
    if low == "profile show":
        return ">>profile show"
    if low == "profile reset":
        return ">>profile reset"
    if low == "profile on":
        return ">>profile on"
    if low == "profile off":
        return ">>profile off"
    if low.startswith("profile set "):
        return ">>" + t
    if low == "preset show":
        return ">>preset show"
    if low == "preset reset":
        return ">>preset reset"
    if low in ("preset fast", "preset balanced", "preset max-recall", "preset max recall"):
        return ">>" + t.replace("max recall", "max-recall")
    if low.startswith("preset set "):
        return ">>" + t.replace("max recall", "max-recall")
    if low in ("memory status", "memory show"):
        return ">>memory status"

    attached = set(getattr(state, "attached_kbs", set()) or set())
    fs_attached = {k for k in attached if k in kb_paths and k != vault_kb_name}

    if fs_attached or getattr(state, "locked_summ_path", ""):
        if low.startswith("lock "):
            rest = t[5:].strip()
            if low.startswith("lock summ_") and low.endswith(".md"):
                return ">>" + t
            if rest and " " not in rest:
                return f">>lock {rest}"
        if low == "unlock":
            return ">>unlock"
        if low == "list files" and fs_attached:
            return ">>list_files"

    # Always allow explicit scratch listing aliases, even if scratchpad
    # has not been attached in this turn/session yet.
    if low in ("list scratchpad", "list contents"):
        return ">>scratchpad list"
    if low.startswith("lock scratchpad "):
        idx = t[len("lock scratchpad ") :].strip()
        if idx:
            return f">>scratchpad lock {idx}"
    if low.startswith("lock scratch "):
        idx = t[len("lock scratch ") :].strip()
        if idx:
            return f">>scratchpad lock {idx}"

    # Soft alias: when scratchpad is attached, numeric lock forms target
    # scratch records instead of KB SUMM locks.
    if "scratchpad" in attached and low.startswith("lock "):
        raw = t[5:].strip()
        m = re.fullmatch(r"\[?\s*(\d+(?:\s*,\s*\d+)*)\s*\]?", raw)
        if m:
            idxs = re.sub(r"\s+", "", m.group(1))
            return f">>scratchpad lock {idxs}"
    # Parity alias: when scratchpad is attached, numeric unlock forms
    # (e.g., "unlock 1" / "unlock [1,2]") route to scratch unlock.
    # Current scratch unlock clears scratch lock scope globally.
    if "scratchpad" in attached and low.startswith("unlock "):
        raw = t[7:].strip()
        m = re.fullmatch(r"\[?\s*(\d+(?:\s*,\s*\d+)*)\s*\]?", raw)
        if m:
            return ">>scratchpad unlock"

    if "scratchpad" not in attached:
        return None

    if low.startswith("scratchpad show "):
        q = t[len("scratchpad show ") :].strip()
        if q:
            return f">>scratchpad show {q}"
    if low.startswith("scratch show "):
        q = t[len("scratch show ") :].strip()
        if q:
            return f">>scratchpad show {q}"
    if low in ("list",):
        return ">>scratchpad list"
    if low.startswith("delete "):
        q = t[7:].strip()
        if q:
            return f">>scratchpad delete {q}"
    return None
