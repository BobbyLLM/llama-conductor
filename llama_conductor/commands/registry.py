"""Declarative command registry and canonical command resolution.

This module keeps command identity and contract metadata in one place:
- aliases
- preconditions
- side effects/state mutations
- help contract
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CommandMeta:
    key: str
    aliases: Tuple[str, ...]
    preconditions: Tuple[str, ...]
    side_effects: Tuple[str, ...]
    help_contract: str


@lru_cache(maxsize=1)
def command_registry() -> Dict[str, CommandMeta]:
    entries = [
        CommandMeta("status", ("status",), ("none",), ("none",), ">>status"),
        CommandMeta("help", ("help",), ("none",), ("none",), ">>help | >>help advanced"),
        CommandMeta("faq", ("faq",), ("none",), ("none",), ">>faq"),
        CommandMeta("preset", ("preset",), ("none",), ("vodka_preset_override",), ">>preset show|set <fast|balanced|max-recall>|reset"),
        CommandMeta("memory", ("memory",), ("vodka available",), ("none",), ">>memory status"),
        CommandMeta("profile", ("profile",), ("none",), ("interaction_profile/session profile state",), ">>profile show|set|reset|on|off"),
        CommandMeta("scratchpad", ("scratchpad", "scratch"), ("scratchpad sidecar available",), ("scratchpad attach/capture/delete + mode/lock state",), ">>scratch | >>scratch strict|free|lock|unlock | >>scratchpad <subcommand>"),
        CommandMeta("trust", ("trust",), ("trust pipeline available",), ("pending trust recommendation state",), ">>trust <query>"),
        CommandMeta("attach", ("attach", "a"), ("kb known",), ("attached_kbs",), ">>attach <kb|all>"),
        CommandMeta("detach", ("detach", "d"), ("kb attached|all",), ("attached_kbs/lock/profile reset on detach all",), ">>detach <kb|all>"),
        CommandMeta("list_files", ("list_files", "list files"), ("filesystem KB attached",), ("none",), ">>list_files"),
        CommandMeta("list_kb", ("list_kb", "kbs"), ("none",), ("none",), ">>list_kb"),
        CommandMeta("lock", ("lock",), ("filesystem KB attached",), ("locked_summ_* fields",), ">>lock SUMM_<name>.md"),
        CommandMeta("unlock", ("unlock",), ("none",), ("locked_summ_* clear",), ">>unlock"),
        CommandMeta("peek", ("peek",), ("fs_rag available",), ("none",), ">>peek <query>"),
        CommandMeta("summ", ("summ", "summarize", "ingest"), ("kb attached|all",), ("creates SUMM files",), ">>summ new|all|<kb>"),
        CommandMeta("move_to_vault", ("move to vault", "move new to vault", "move_to_vault", "mtv"), ("source KB available",), ("vault chunks created",), ">>move to vault [all|new]"),
        CommandMeta("send_to_vault", ("send to vault", "send_to_vault"), ("operator archive workflow",), ("none",), ">>send to vault"),
        CommandMeta("calc", ("calc",), ("sidecar available",), ("none",), ">>calc <expression>"),
        CommandMeta("list", ("list",), ("vodka available",), ("none",), ">>list"),
        CommandMeta("find", ("find",), ("sidecar available",), ("none",), ">>find <query>"),
        CommandMeta("flush", ("flush",), ("none",), ("profile/style/runtime state reset + cache flush",), ">>flush"),
        CommandMeta("wiki", ("wiki",), ("sidecar available",), ("none",), ">>wiki <topic>"),
        CommandMeta("define", ("define",), ("sidecar available",), ("none",), ">>define <word>"),
        CommandMeta("exchange", ("exchange",), ("sidecar available",), ("none",), ">>exchange <query>"),
        CommandMeta("weather", ("weather",), ("sidecar available",), ("none",), ">>weather <location>"),
        CommandMeta("judge", ("judge",), ("judge worker available",), ("none",), ">>judge [criterion] : item1, item2, item3 [--verbose]"),
        CommandMeta("raw", ("raw",), ("none",), ("raw_sticky=true",), ">>raw"),
        CommandMeta("raw_off", ("raw off",), ("none",), ("raw_sticky=false",), ">>raw off"),
        CommandMeta("fun", ("fun", "f"), ("none",), ("fun_sticky=true, fr_sticky=false",), ">>fun | >>f"),
        CommandMeta("fun_off", ("fun off", "f off"), ("none",), ("fun_sticky=false",), ">>fun off | >>f off"),
        CommandMeta("fun_rewrite", ("fr", "fun_rewrite"), ("none",), ("fr_sticky=true, fun_sticky=false",), ">>fr"),
        CommandMeta("fun_rewrite_off", ("fr off", "fun_rewrite off"), ("none",), ("fr_sticky=false",), ">>fr off"),
    ]
    return {entry.key: entry for entry in entries}


@lru_cache(maxsize=1)
def _alias_to_key() -> Dict[str, str]:
    reg = command_registry()
    alias_map: Dict[str, str] = {}
    for key, meta in reg.items():
        for alias in meta.aliases:
            alias_map[alias.strip().lower().replace("-", "_")] = key
    return alias_map


def resolve_command_key(low: str, parts: List[str]) -> str:
    """Resolve canonical command key for command-group dispatch.

    This is intentionally conservative: only command identity is normalized.
    Behavioral dispatch order remains in handlers.py to prevent drift.
    """
    low_norm = (low or "").strip().lower().replace("-", "_")
    if low_norm in ("raw off", "raw_off"):
        return "raw_off"
    if low_norm in ("fun off", "f off"):
        return "fun_off"
    if low_norm in ("fr off", "fun_rewrite off", "fun_rewrite_off"):
        return "fun_rewrite_off"
    if low_norm in ("list files", "list_files"):
        return "list_files"
    if low_norm.startswith("move to vault") or low_norm.startswith("move_to_vault") or low_norm.startswith("mtv") or low_norm.startswith("move new to vault"):
        return "move_to_vault"
    if low_norm.startswith("send to vault") or low_norm.startswith("send_to_vault"):
        return "send_to_vault"

    if not parts:
        return ""

    first = (parts[0] or "").strip().lower().replace("-", "_")
    return _alias_to_key().get(first, first)

