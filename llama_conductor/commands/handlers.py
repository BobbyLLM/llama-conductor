# commands/handlers.py
"""Main command handling logic."""

import os
from typing import Callable, Dict, List, Optional

from ..config import (
    KB_PATHS,
    VAULT_KB_NAME,
    CHEAT_SHEET_PATH,
    FS_TOP_K,
    FS_MAX_CHARS,
)
from ..session_state import SessionState
from ..interaction_profile import (
    SETTABLE_FIELDS,
    compute_effective_strength,
    profile_set,
    render_profile_show,
    reset_profile,
)
from ..helpers import is_command, strip_cmd_prefix, parse_args
from ..vault_ops import summ_new_in_kb, move_summ_to_vault

# Optional imports (feature-detected)
try:
    from ..fs_rag import build_fs_facts_block, find_summ_file_matches, find_summ_file_candidates  # type: ignore
except Exception:
    build_fs_facts_block = None  # type: ignore
    find_summ_file_matches = None  # type: ignore
    find_summ_file_candidates = None  # type: ignore

try:
    from ..sidecars import (  # type: ignore
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
    from ..trust_pipeline import (  # type: ignore
        handle_trust_command,
        generate_recommendations,
        format_recommendations,
    )
except Exception:
    handle_trust_command = None  # type: ignore
    generate_recommendations = None  # type: ignore
    format_recommendations = None  # type: ignore

try:
    from ..scratchpad_sidecar import (  # type: ignore
        get_session_scratchpad_path,
        clear_scratchpad,
        list_scratchpad_records,
        capture_scratchpad_output,
        delete_scratchpad_by_index,
        delete_scratchpad_by_query,
        build_scratchpad_dump_text,
    )
except Exception:
    get_session_scratchpad_path = None  # type: ignore
    clear_scratchpad = None  # type: ignore
    list_scratchpad_records = None  # type: ignore
    capture_scratchpad_output = None  # type: ignore
    delete_scratchpad_by_index = None  # type: ignore
    delete_scratchpad_by_query = None  # type: ignore
    build_scratchpad_dump_text = None  # type: ignore

try:
    from ..vodka_filter import purge_session_memory_jsonl  # type: ignore
except Exception:
    purge_session_memory_jsonl = None  # type: ignore


def load_help_text() -> str:
    """Load command cheat sheet."""
    try:
        with open(CHEAT_SHEET_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[help missing: {e}]"


def _clear_locked_summ(state: SessionState) -> str:
    """Clear current lock and return previous locked filename (if any)."""
    if not state.locked_summ_path:
        return ""
    prev = state.locked_summ_file or os.path.basename(state.locked_summ_path)
    state.locked_summ_file = ""
    state.locked_summ_kb = ""
    state.locked_summ_path = ""
    state.locked_summ_rel_path = ""
    state.locked_last_fact_lines = 0
    return prev


def _clear_pending_lock(state: SessionState) -> None:
    state.pending_lock_candidate = ""


def _reset_profile_runtime_state(state: SessionState) -> None:
    """Reset session interaction-profile state to defaults."""
    reset_profile(state.interaction_profile)
    state.profile_turn_counter = 0
    state.profile_effective_strength = 0.0
    state.profile_output_compliance = 0.0
    state.profile_blocked_nicknames.clear()
    state.serious_ack_reframe_streak = 0
    state.serious_last_body_signature = ""
    state.serious_repeat_streak = 0
    state.pending_sensitive_confirm_query = ""


def _list_lockable_summ_files(state: SessionState) -> List[str]:
    """Return lockable SUMM files across attached filesystem KBs."""
    rows: List[str] = []
    source_kbs = sorted(
        k for k in state.attached_kbs if k and k in KB_PATHS and k not in (VAULT_KB_NAME, "scratchpad")
    )
    for kb in source_kbs:
        root = KB_PATHS.get(kb)
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "original" in {p.lower() for p in dirpath.split(os.sep)}:
                continue
            for fn in sorted(filenames):
                if not (fn.startswith("SUMM_") and fn.lower().endswith(".md")):
                    continue
                abs_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(abs_path, root)
                mark = ""
                if (
                    state.locked_summ_path
                    and os.path.abspath(abs_path).lower() == os.path.abspath(state.locked_summ_path).lower()
                ):
                    mark = " [LOCKED]"
                rel_norm = rel.replace("\\", "/")
                rel_part = f" rel={rel_norm}" if rel_norm != fn else ""
                rows.append(f"- kb={kb} file={fn}{rel_part}{mark}")
    rows.sort()
    return rows


def _source_filesystem_kbs(state: SessionState) -> set:
    return {
        k
        for k in state.attached_kbs
        if k and k in KB_PATHS and k not in (VAULT_KB_NAME, "scratchpad")
    }


def _render_status(session_id: str, state: SessionState) -> str:
    return (
        "[status]\n"
        f"session_id={session_id}\n"
        f"attached_kbs={sorted(state.attached_kbs)}\n"
        f"locked_summ_file={state.locked_summ_file!r}\n"
        f"locked_summ_kb={state.locked_summ_kb!r}\n"
        f"pending_lock_candidate={state.pending_lock_candidate!r}\n"
        f"pending_sensitive_confirm_query={state.pending_sensitive_confirm_query!r}\n"
        f"fun_sticky={state.fun_sticky}\n"
        f"fun_rewrite_sticky={state.fun_rewrite_sticky}\n"
        f"profile_enabled={state.profile_enabled}\n"
        f"profile_confidence={state.interaction_profile.confidence:.2f}\n"
        f"profile_effective_strength={state.profile_effective_strength:.2f}\n"
        f"profile_output_compliance={state.profile_output_compliance:.2f}\n"
        f"profile_blocked_nicknames={sorted(state.profile_blocked_nicknames)!r}\n"
        f"profile_last_updated_turn={state.interaction_profile.last_updated_turn}\n"
        f"serious_ack_reframe_streak={state.serious_ack_reframe_streak}\n"
        f"serious_repeat_streak={state.serious_repeat_streak}\n"
        f"last_query={state.rag_last_query!r}\n"
        f"last_hits={state.rag_last_hits}\n"
        f"vault_last_query={state.vault_last_query!r}\n"
        f"vault_last_hits={state.vault_last_hits}\n"
    )


def _dispatch_exact_early(low: str, *, state: SessionState, session_id: str) -> Optional[str]:
    handlers: Dict[str, Callable[[], str]] = {
        "status": lambda: _render_status(session_id, state),
        "help": load_help_text,
    }
    fn = handlers.get(low)
    return fn() if fn else None


def _dispatch_exact_fun(low: str, *, state: SessionState) -> Optional[str]:
    def _fun_on() -> str:
        state.fun_sticky = True
        state.fun_rewrite_sticky = False
        return "[router] fun mode ON (sticky)"

    def _fun_off() -> str:
        state.fun_sticky = False
        return "[router] fun mode OFF"

    def _fr_on() -> str:
        state.fun_rewrite_sticky = True
        state.fun_sticky = False
        return "[router] fun rewrite ON (sticky)"

    def _fr_off() -> str:
        state.fun_rewrite_sticky = False
        return "[router] fun rewrite OFF"

    handlers: Dict[str, Callable[[], str]] = {
        "fun": _fun_on,
        "f": _fun_on,
        "fun off": _fun_off,
        "f off": _fun_off,
        "fun_rewrite": _fr_on,
        "fr": _fr_on,
        "fun_rewrite off": _fr_off,
        "fr off": _fr_off,
    }
    fn = handlers.get(low)
    return fn() if fn else None


def _dispatch_exact_listing(low: str, *, state: SessionState) -> Optional[str]:
    if low in ("list_kb", "list", "kbs"):
        known = sorted(set(KB_PATHS.keys()) | {"scratchpad"})
        return "[router] known KBs: " + ", ".join(known)

    return None


def _handle_unlock(parts: List[str], *, state: SessionState) -> Optional[str]:
    if not (parts and parts[0].lower() == "unlock"):
        return None
    _clear_pending_lock(state)
    prev = _clear_locked_summ(state)
    if not prev:
        return "[router] no locked file"
    return f"[router] unlocked file: {prev}"


def _handle_list_files(low: str, *, state: SessionState) -> Optional[str]:
    if low not in ("list_files", "list files"):
        return None
    rows = _list_lockable_summ_files(state)
    if not rows:
        return "[router] No lockable SUMM files found in attached filesystem KBs."
    return "[router] lockable SUMM files:\n" + "\n".join(rows[:300])


def _handle_list_known_kbs(low: str) -> Optional[str]:
    if low not in ("list_kb", "list", "kbs"):
        return None
    known = sorted(set(KB_PATHS.keys()) | {"scratchpad"})
    return "[router] known KBs: " + ", ".join(known)


def handle_command(cmd_text: str, *, state: SessionState, session_id: str) -> Optional[str]:
    """Return immediate reply if handled, else None."""
    if not is_command(cmd_text):
        return None

    cmd = strip_cmd_prefix(cmd_text)
    if not cmd:
        return "[router] empty command"

    low = cmd.strip().lower()
    low = low.replace("-", "_")
    parts = parse_args(cmd)

    # Alias: >>list scratchpad -> >>scratchpad list
    if len(parts) >= 2 and parts[0].lower() == "list" and parts[1].lower() == "scratchpad":
        low = "scratchpad list"
        parts = ["scratchpad", "list"]

    # Scratchpad shorthand aliases when scratchpad is attached
    # - >>add <text>  -> >>scratchpad add <text>
    # - >>list        -> >>scratchpad list (only when scratchpad attached)
    if "scratchpad" in state.attached_kbs:
        if parts and parts[0].lower() == "add":
            if not capture_scratchpad_output:
                return "[scratchpad] add unavailable"
            payload = cmd[len(parts[0]) :].strip()
            if not payload:
                return "[scratchpad] usage: >>add <text>"
            rec = capture_scratchpad_output(
                session_id=session_id,
                source_command=">>add",
                text=payload,
            )
            if not rec:
                return "[scratchpad] add failed"
            return f"[scratchpad] added sha={str(rec.get('sha256', ''))[:12]}"

        if low == "list" and list_scratchpad_records:
            recs = list_scratchpad_records(session_id, limit=20)
            if not recs:
                return "[scratchpad] empty"
            lines = [f"[scratchpad] last {len(recs)} capture(s):"]
            start = max(0, len(recs) - 20)
            for i, rec in enumerate(recs[start:], 1):
                src = str(rec.get("source_command", "unknown"))
                txt = str(rec.get("text", "") or "")
                preview = str(rec.get("preview", "") or "").strip()
                if not preview:
                    compact = " ".join(txt.split()).strip()
                    preview = compact[:219] + "..." if len(compact) > 220 else compact
                lines.append(f"{i}. {src} chars={len(txt)}")
                lines.append(f"   {preview}")
            return "\n".join(lines)

    # status/help (exact dispatch, behavior-preserving)
    out = _dispatch_exact_early(low, state=state, session_id=session_id)
    if out is not None:
        return out

    # profile controls
    if parts and parts[0].lower() == "profile":
        def _norm_level(v: str) -> str:
            low_v = (v or "").strip().lower()
            if low_v in ("med", "mid"):
                return "medium"
            return low_v

        def _as_bool_text(v: str) -> str:
            low_v = (v or "").strip().lower()
            if low_v in ("on", "yes", "y", "1", "true"):
                return "true"
            if low_v in ("off", "no", "n", "0", "false"):
                return "false"
            return low_v

        sub = parts[1].lower() if len(parts) > 1 else "show"
        # User-friendly shorthand aliases.
        if sub in ("direct", "neutral", "softened"):
            ok, msg = profile_set(state.interaction_profile, "correction_style", sub)
            return msg if ok else f"[profile] invalid style: {sub}"
        if sub in ("snark", "sarcasm", "profanity", "verbosity", "sensitive_override", "sensitive"):
            if len(parts) < 3:
                if sub == "snark":
                    return "[profile] usage: >>profile snark <low|medium|high>"
                if sub == "sarcasm":
                    return "[profile] usage: >>profile sarcasm <off|low|medium|high>"
                if sub == "profanity":
                    return "[profile] usage: >>profile profanity <on|off>"
                if sub == "verbosity":
                    return "[profile] usage: >>profile verbosity <compact|standard|expanded>"
                return "[profile] usage: >>profile sensitive <on|off>"
            raw_val = " ".join(parts[2:]).strip()
            if sub == "snark":
                return profile_set(state.interaction_profile, "snark_tolerance", _norm_level(raw_val))[1]
            if sub == "sarcasm":
                return profile_set(state.interaction_profile, "sarcasm_level", _norm_level(raw_val))[1]
            if sub == "profanity":
                return profile_set(state.interaction_profile, "profanity_ok", _as_bool_text(raw_val))[1]
            if sub == "verbosity":
                return profile_set(state.interaction_profile, "verbosity", raw_val)[1]
            return profile_set(state.interaction_profile, "sensitive_override", _as_bool_text(raw_val))[1]
        if sub in ("casual", "feral", "turbo"):
            # Shortcut presets.
            p = state.interaction_profile
            if sub == "casual":
                profile_set(p, "correction_style", "direct")
                profile_set(p, "verbosity", "compact")
                profile_set(p, "snark_tolerance", "high")
                profile_set(p, "sarcasm_level", "medium")
                profile_set(p, "profanity_ok", "false")
                return "[profile] override applied: casual"
            # 'feral' and 'turbo' are equivalent.
            profile_set(p, "correction_style", "direct")
            profile_set(p, "verbosity", "compact")
            profile_set(p, "snark_tolerance", "high")
            profile_set(p, "sarcasm_level", "high")
            profile_set(p, "profanity_ok", "true")
            return "[profile] override applied: feral"

        if sub in ("override", "preset", "turbo"):
            name = "feral" if sub == "turbo" else (parts[2].lower() if len(parts) > 2 else "")
            p = state.interaction_profile
            if name == "direct":
                profile_set(p, "correction_style", "direct")
                profile_set(p, "verbosity", "compact")
                profile_set(p, "snark_tolerance", "medium")
                profile_set(p, "sarcasm_level", "low")
                profile_set(p, "profanity_ok", "false")
                return "[profile] override applied: direct"
            if name == "casual":
                profile_set(p, "correction_style", "direct")
                profile_set(p, "verbosity", "compact")
                profile_set(p, "snark_tolerance", "high")
                profile_set(p, "sarcasm_level", "medium")
                profile_set(p, "profanity_ok", "false")
                return "[profile] override applied: casual"
            if name == "feral":
                profile_set(p, "correction_style", "direct")
                profile_set(p, "verbosity", "compact")
                profile_set(p, "snark_tolerance", "high")
                profile_set(p, "sarcasm_level", "high")
                profile_set(p, "profanity_ok", "true")
                return "[profile] override applied: feral"
            return "[profile] usage: >>profile override <direct|casual|feral> (alias: >>profile turbo)"
        if sub == "show":
            state.profile_effective_strength = compute_effective_strength(
                state.interaction_profile,
                enabled=state.profile_enabled,
                output_compliance=state.profile_output_compliance,
            )
            return render_profile_show(
                state.interaction_profile,
                enabled=state.profile_enabled,
                effective_strength=state.profile_effective_strength,
            )
        if sub == "reset":
            _reset_profile_runtime_state(state)
            return "[profile] reset to defaults"
        if sub == "on":
            state.profile_enabled = True
            return "[profile] enabled"
        if sub == "off":
            state.profile_enabled = False
            return "[profile] disabled"
        if sub == "set":
            if len(parts) < 3 or "=" not in " ".join(parts[2:]):
                return "[profile] usage: >>profile set <field>=<value>"
            rhs = " ".join(parts[2:]).strip()
            field_name, value = rhs.split("=", 1)
            ok, msg = profile_set(state.interaction_profile, field_name.strip(), value.strip())
            if not ok and field_name.strip() in SETTABLE_FIELDS:
                return msg
            return msg
        return (
            "[profile] usage: >>profile show|set|reset|on|off | "
            ">>profile <direct|neutral|softened> | "
            ">>profile snark <low|medium|high> | "
            ">>profile sarcasm <off|low|medium|high> | "
            ">>profile profanity <on|off>"
        )

    # scratchpad controls
    if parts and parts[0].lower() in ("scratchpad", "scratch"):
        if not get_session_scratchpad_path or not clear_scratchpad or not list_scratchpad_records:
            return "[scratchpad] unavailable (scratchpad_sidecar.py missing)"

        # UX: bare >>scratch acts as attach (idempotent), not status.
        if parts[0].lower() == "scratch" and len(parts) == 1:
            state.attached_kbs.add("scratchpad")
            return (
                f"[router] attached: {sorted(state.attached_kbs)}\n"
                "cmd: >>attach scratchpad | >>detach scratchpad | >>scratch status | >>scratch list\n"
                "see >>help for full scratchpad commands"
            )

        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub in ("clear", "flush"):
            ok = clear_scratchpad(session_id)
            return "[scratchpad] cleared" if ok else "[scratchpad] clear failed"

        if sub in ("list", "ls"):
            recs = list_scratchpad_records(session_id, limit=20)
            if not recs:
                return "[scratchpad] empty"
            lines = [f"[scratchpad] last {len(recs)} capture(s):"]
            start = max(0, len(recs) - 20)
            for i, rec in enumerate(recs[start:], 1):
                src = str(rec.get("source_command", "unknown"))
                txt = str(rec.get("text", "") or "")
                preview = str(rec.get("preview", "") or "").strip()
                if not preview:
                    compact = " ".join(txt.split()).strip()
                    preview = compact[:219] + "..." if len(compact) > 220 else compact
                lines.append(f"{i}. {src} chars={len(txt)}")
                lines.append(f"   {preview}")
            return "\n".join(lines)

        if sub in ("show", "dump"):
            if not build_scratchpad_dump_text:
                return "[scratchpad] show unavailable"
            query = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
            if not query:
                query = "all"
            return build_scratchpad_dump_text(session_id=session_id, query=query)

        if sub == "add":
            if not capture_scratchpad_output:
                return "[scratchpad] add unavailable"
            payload = cmd[len(parts[0]) :].strip()
            payload = payload[len(parts[1]) :].strip() if len(parts) > 1 else payload
            if not payload:
                return "[scratchpad] usage: >>scratchpad add <text>"
            rec = capture_scratchpad_output(
                session_id=session_id,
                source_command=">>scratchpad add",
                text=payload,
            )
            if not rec:
                return "[scratchpad] add failed"
            return f"[scratchpad] added sha={str(rec.get('sha256', ''))[:12]}"

        if sub == "delete":
            if len(parts) < 3:
                return "[scratchpad] usage: >>scratchpad delete <index|query>"
            arg = " ".join(parts[2:]).strip()
            if not arg:
                return "[scratchpad] usage: >>scratchpad delete <index|query>"
            if arg.isdigit():
                if not delete_scratchpad_by_index:
                    return "[scratchpad] delete unavailable"
                ok = delete_scratchpad_by_index(session_id, int(arg))
                return "[scratchpad] deleted 1 record" if ok else "[scratchpad] delete failed"
            if not delete_scratchpad_by_query:
                return "[scratchpad] delete unavailable"
            n = delete_scratchpad_by_query(session_id, arg)
            return f"[scratchpad] deleted {n} record(s)"

        recs = list_scratchpad_records(session_id, limit=1000000)
        attached = "scratchpad" in state.attached_kbs
        return (
            "[scratchpad status]\n"
            f"attached={attached}\n"
            f"captures={len(recs)}\n"
            "tip: raw captures are stored in total_recall/session_kb/<session_id>.jsonl\n"
            "cmd: >>attach scratchpad | >>detach scratchpad | >>scratch status | >>scratch list\n"
            "see >>help for full scratchpad commands"
        )

    # trust mode (tool recommendation)
    if parts and parts[0].lower() == "trust":
        if not handle_trust_command or not generate_recommendations or not format_recommendations:
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
        return "[trust] " + format_recommendations(recommendations, query=query)
    # attach/detach/list
    if parts and parts[0].lower() in ("attach", "a"):
        if len(parts) < 2:
            return "[router] usage: >>attach <kb|all>"

        target = parts[1].strip().lower()
        if target == "scratch":
            target = "scratchpad"

        if target == "all":
            # Attach all filesystem KBs that exist on disk (+scratchpad).
            missing: List[str] = []
            for k in sorted(KB_PATHS.keys()):
                if not k:
                    continue
                folder = KB_PATHS.get(k, "")
                if folder and os.path.isdir(folder):
                    state.attached_kbs.add(k)
                else:
                    missing.append(f"- {k} -> {folder}")
            state.attached_kbs.add("scratchpad")
            msg = f"[router] attached: {sorted(state.attached_kbs)}"
            if missing:
                msg += "\n[router] skipped KBs with missing folders (create path or update router_config.yaml):\n"
                msg += "\n".join(missing[:25])
            return msg

        # Scratchpad virtual KB
        if target == "scratchpad":
            state.attached_kbs.add("scratchpad")
            return (
                f"[router] attached: {sorted(state.attached_kbs)}\n"
                "cmd: >>attach scratchpad | >>detach scratchpad | >>scratch status | >>scratch list\n"
                "see >>help for full scratchpad commands"
            )

        # Special error for vault (Qdrant collection, not filesystem KB)
        if target == VAULT_KB_NAME:
            return (
                f"[error] '{VAULT_KB_NAME}' is the Qdrant collection, not a filesystem KB.\n"
                f"\n"
                f"To search Vault:\n"
                f"  - Use ##mentats <query> (automatically searches Qdrant)\n"
                f"\n"
                f"Available filesystem KBs:\n"
                f"  - {', '.join(sorted(KB_PATHS.keys()))}\n"
                f"\n"
                f"Use >>attach <kb> to attach a filesystem KB."
            )

        # Only allow filesystem KBs
        if target in KB_PATHS:
            folder = KB_PATHS.get(target, "")
            if not folder or not os.path.isdir(folder):
                return (
                    f"[router] cannot attach '{target}': folder missing. "
                    f"Please create {folder} or update router_config.yaml."
                )
            state.attached_kbs.add(target)
            return f"[router] attached: {sorted(state.attached_kbs)}"

        return f"[router] unknown kb: '{target}' (known: {sorted(KB_PATHS.keys())})"

    if parts and parts[0].lower() in ("detach", "d"):
        if len(parts) < 2:
            return "[router] usage: >>detach <kb|all>"
        target = parts[1].strip().lower()
        if target == "scratch":
            target = "scratchpad"
        if target == "all":
            scratchpad_deleted = False
            scratchpad_count = 0
            prev_locked = _clear_locked_summ(state)
            _clear_pending_lock(state)
            _reset_profile_runtime_state(state)
            if "scratchpad" in state.attached_kbs and list_scratchpad_records and clear_scratchpad:
                recs = list_scratchpad_records(session_id, limit=1000000)
                scratchpad_count = len(recs)
                scratchpad_deleted = bool(clear_scratchpad(session_id))
            state.attached_kbs.clear()
            unlock_note = f" and unlocked '{prev_locked}'" if prev_locked else ""
            if scratchpad_deleted:
                return (
                    f"[router] detached ALL and deleted scratchpad data ({scratchpad_count} records dumped)"
                    f"{unlock_note}"
                )
            return f"[router] detached ALL{unlock_note}"
        if target in state.attached_kbs:
            unlock_note = ""
            if target in KB_PATHS and state.locked_summ_path and state.locked_summ_kb == target:
                prev_locked = _clear_locked_summ(state)
                if prev_locked:
                    unlock_note = f" and unlocked '{prev_locked}'"
            if target in KB_PATHS:
                _clear_pending_lock(state)
            if target == "scratchpad" and list_scratchpad_records and clear_scratchpad:
                recs = list_scratchpad_records(session_id, limit=1000000)
                n = len(recs)
                deleted = bool(clear_scratchpad(session_id))
                state.attached_kbs.remove(target)
                if not deleted:
                    return f"[router] detached 'scratchpad' (warning: scratchpad delete failed){unlock_note}"
                return f"[router] detached 'scratchpad' and deleted scratchpad data ({n} records dumped){unlock_note}"
            state.attached_kbs.remove(target)
            return f"[router] detached '{target}'{unlock_note}"
        return f"[router] kb not attached: '{target}'"

    out = _handle_list_files(low, state=state)
    if out is not None:
        return out

    out = _handle_list_known_kbs(low)
    if out is not None:
        return out

    out = _dispatch_exact_listing(low, state=state)
    if out is not None:
        return out

    # lock / unlock SUMM source file (filesystem KB grounding scope)
    if parts and parts[0].lower() == "lock":
        if len(parts) < 2:
            return "[router] usage: >>lock SUMM_<name>.md"

        target_file = parts[1].strip()
        low_target = target_file.lower()

        source_kbs = _source_filesystem_kbs(state)
        if not source_kbs:
            return "[router] No filesystem KBs attached. Attach KB(s) first, then: >>lock SUMM_<name>.md"

        if not find_summ_file_matches:
            return "[router] lock unavailable (fs_rag missing)"

        # Full-name lock path: SUMM_*.md
        if low_target.startswith("summ_") and low_target.endswith(".md"):
            matches = find_summ_file_matches(target_file, source_kbs, KB_PATHS)
            if not matches:
                return f"[router] lock target not found in attached KBs: {target_file}"
        else:
            # Partial-name lock path: suggest candidate then require Y/N confirm.
            if not find_summ_file_candidates:
                return "[router] lock unavailable (fs_rag missing)"
            matches = find_summ_file_candidates(target_file, source_kbs, KB_PATHS)
            if not matches:
                return f"[router] lock target not found in attached KBs: {target_file}"
            if len(matches) == 1:
                _kb, _abs_path, _rel_path, fn = matches[0]
                state.pending_lock_candidate = fn
                return f"[router] Did you mean: >>lock {fn} ? [Y/N]"

        if len(matches) > 1:
            lines = [f"[router] lock target is ambiguous: {target_file}"]
            for kb, _abs_path, rel_path, fn in matches[:10]:
                lines.append(f"- kb={kb} file={fn} rel={rel_path}")
            lines.append("Tip: detach unrelated KBs, then retry >>lock.")
            _clear_pending_lock(state)
            return "\n".join(lines)

        kb, abs_path, rel_path, fn = matches[0]
        _clear_pending_lock(state)
        state.locked_summ_file = fn
        state.locked_summ_kb = kb
        state.locked_summ_path = abs_path
        state.locked_summ_rel_path = rel_path
        state.locked_last_fact_lines = 0
        return f"[router] locked file: kb={kb} file={fn}"

    out = _handle_unlock(parts, state=state)
    if out is not None:
        return out

    # fun toggles (exact dispatch, behavior-preserving)
    out = _dispatch_exact_fun(low, state=state)
    if out is not None:
        return out

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
    if (
        low.startswith("move to vault")
        or low.startswith("move_to_vault")
        or low.startswith("mtv")
        or low.startswith("move new to vault")
    ):
        # parse:
        # - >>move to vault [all]
        # - >>move new to vault
        # - >>move_to_vault new
        # - >>mtv new
        arg = parts[-1].lower() if parts else ""
        newest_only = low.startswith("move new to vault") or arg == "new"
        if arg == "all":
            src = set(k for k in KB_PATHS.keys() if k and k != VAULT_KB_NAME)
        else:
            src = set(k for k in state.attached_kbs if k and k != VAULT_KB_NAME)

        if not src:
            return "[router] No source KBs to promote. Attach KB(s) then: >>move to vault"

        res = move_summ_to_vault(src, newest_only=newest_only)
        mode = "newest-only " if newest_only else ""
        msg = f"[router] {mode}move-to-vault complete: files={res.get('files', 0)} chunks={res.get('chunks', 0)}"
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
        # User-requested behavior: flush also resets profile/session-style identity.
        _reset_profile_runtime_state(state)
        state.rag_last_query = ""
        state.rag_last_hits = 0
        state.vault_last_query = ""
        state.vault_last_hits = 0
        state.fun_sticky = False
        state.fun_rewrite_sticky = False
        state.raw_sticky = False
        purged = 0
        if purge_session_memory_jsonl:
            try:
                base = ""
                if state.vodka is not None and hasattr(state.vodka, "valves"):
                    base = str(getattr(state.vodka.valves, "storage_dir", "") or "").strip()
                purged = int(purge_session_memory_jsonl(base))
            except Exception:
                purged = 0
        if not flush_ctc_cache or not state.vodka:
            return (
                "[flush] profile/session identity reset; CTC cache unavailable "
                f"(Vodka not initialized). session-memory deleted={purged}."
            )
        ctc_msg = flush_ctc_cache(state.vodka)
        return (
            f"{ctc_msg}\n"
            f"[flush] profile/session identity reset.\n"
            f"[flush] session-memory deleted={purged}."
        )

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

