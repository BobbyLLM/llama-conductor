"""Kaioken vNext phrase asset loader.

Authoritative spec: docs/planning/KAIOKEN-vNEXT.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_VALID_SLOTS = {"opener", "anchor_line", "boundary_line", "repair_ack", "closer"}
_VALID_REGISTERS = {"casual", "personal", "working", "any"}


def _resolve_path_guard(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    parts = [part.lower() for part in resolved.parts]
    is_inside_data = False
    for idx in range(len(parts) - 1):
        if parts[idx] == "llama_conductor" and parts[idx + 1] == "data":
            is_inside_data = True
            break
    if is_inside_data and "cheatsheet" in resolved.name.lower():
        raise ValueError(f"cheatsheet asset paths are not allowed under llama_conductor/data: {resolved}")
    return resolved


def _normalize_row(row: Dict[str, Any], *, source: str, line_no: int, strict: bool) -> Optional[Dict[str, Any]]:
    required = {"id", "slot", "macro_register", "text", "enabled", "priority"}
    missing = [key for key in required if key not in row]
    if missing:
        if strict:
            raise ValueError(f"{source}:{line_no} missing required fields: {missing}")
        logger.warning("%s:%s missing required fields: %s", source, line_no, missing)
        return None

    normalized = dict(row)
    normalized.setdefault("speech_act", "")
    normalized.setdefault("forbidden_contexts", [])
    normalized.setdefault("cooldown_turns", 6)

    if not isinstance(normalized["id"], str) or not normalized["id"].strip():
        if strict:
            raise ValueError(f"{source}:{line_no} invalid id: {normalized['id']!r}")
        logger.warning("%s:%s invalid id: %r", source, line_no, normalized["id"])
        return None
    if not isinstance(normalized["slot"], str) or normalized["slot"] not in _VALID_SLOTS:
        if strict:
            raise ValueError(f"{source}:{line_no} invalid slot: {normalized['slot']!r}")
        logger.warning("%s:%s invalid slot: %r", source, line_no, normalized["slot"])
        return None
    if not isinstance(normalized["macro_register"], str) or normalized["macro_register"] not in _VALID_REGISTERS:
        if strict:
            raise ValueError(f"{source}:{line_no} invalid macro_register: {normalized['macro_register']!r}")
        logger.warning("%s:%s invalid macro_register: %r", source, line_no, normalized["macro_register"])
        return None
    if not isinstance(normalized["text"], str):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid text: {normalized['text']!r}")
        logger.warning("%s:%s invalid text: %r", source, line_no, normalized["text"])
        return None
    if not isinstance(normalized["enabled"], bool):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid enabled: {normalized['enabled']!r}")
        logger.warning("%s:%s invalid enabled: %r", source, line_no, normalized["enabled"])
        return None
    if not isinstance(normalized["priority"], int):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid priority: {normalized['priority']!r}")
        logger.warning("%s:%s invalid priority: %r", source, line_no, normalized["priority"])
        return None
    if not isinstance(normalized["speech_act"], str):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid speech_act: {normalized['speech_act']!r}")
        logger.warning("%s:%s invalid speech_act: %r", source, line_no, normalized["speech_act"])
        return None
    if not isinstance(normalized["forbidden_contexts"], list):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid forbidden_contexts: {normalized['forbidden_contexts']!r}")
        logger.warning("%s:%s invalid forbidden_contexts: %r", source, line_no, normalized["forbidden_contexts"])
        return None
    if not isinstance(normalized["cooldown_turns"], int):
        if strict:
            raise ValueError(f"{source}:{line_no} invalid cooldown_turns: {normalized['cooldown_turns']!r}")
        logger.warning("%s:%s invalid cooldown_turns: %r", source, line_no, normalized["cooldown_turns"])
        return None

    return normalized


def _load_jsonl(path: Path, *, strict: bool) -> tuple[list[Dict[str, Any]], int]:
    entries: list[Dict[str, Any]] = []
    invalid = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except Exception as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
                logger.warning("%s:%s invalid JSON: %s", path, line_no, exc)
                invalid += 1
                continue
            if not isinstance(row, dict):
                if strict:
                    raise ValueError(f"{path}:{line_no} expected object row, got {type(row).__name__}")
                logger.warning("%s:%s expected object row, got %s", path, line_no, type(row).__name__)
                invalid += 1
                continue
            normalized = _normalize_row(row, source=str(path), line_no=line_no, strict=strict)
            if normalized is None:
                invalid += 1
                continue
            entries.append(normalized)
    return entries, invalid


def load_kaioken_assets(default_path, user_path, strict=True) -> list[dict]:
    default_path = _resolve_path_guard(str(default_path))
    user_path = _resolve_path_guard(str(user_path))

    default_entries, default_invalid = _load_jsonl(default_path, strict=strict)
    user_entries, user_invalid = _load_jsonl(user_path, strict=strict)

    merged: dict[str, Dict[str, Any]] = {}
    shadowed = 0
    for entry in default_entries:
        merged[entry["id"]] = entry
    for entry in user_entries:
        if entry["id"] in merged:
            shadowed += 1
        merged[entry["id"]] = entry

    logger.info(
        "kaioken_assets loaded: defaults=%d user_overrides=%d invalid_rows=%d shadowed_defaults=%d",
        len(default_entries),
        len(user_entries),
        default_invalid + user_invalid,
        shadowed,
    )

    return sorted(merged.values(), key=lambda entry: (entry["priority"], entry["id"]))


def select_phrases(
    entries,
    macro_register,
    speech_act="",
    recent_phrase_ids=None,
    max_per_slot=1,
) -> dict:
    recent = set(recent_phrase_ids or [])
    chosen: dict[str, Dict[str, Any]] = {}
    candidates: dict[str, list[Dict[str, Any]]] = {}

    for entry in entries:
        if not entry.get("enabled", False):
            continue
        if entry.get("slot") not in _VALID_SLOTS:
            continue
        if entry.get("macro_register") not in {macro_register, "any"}:
            continue
        if speech_act and entry.get("speech_act", "") not in {"", speech_act}:
            continue
        if entry.get("id") in recent:
            continue
        candidates.setdefault(entry["slot"], []).append(entry)

    for slot, rows in candidates.items():
        rows_sorted = sorted(rows, key=lambda entry: (entry["priority"], entry["id"]))
        if rows_sorted:
            chosen[slot] = rows_sorted[0]

    return chosen


__all__ = ["load_kaioken_assets", "select_phrases"]
