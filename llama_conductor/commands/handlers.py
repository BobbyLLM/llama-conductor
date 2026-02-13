# commands/handlers.py
"""Main command handling logic."""

import os
from typing import List, Optional

from ..config import (
    KB_PATHS,
    VAULT_KB_NAME,
    SUMM_PROMPT_PATH,
    CHEAT_SHEET_PATH,
    FS_TOP_K,
    FS_MAX_CHARS,
)
from ..session_state import SessionState
from ..helpers import is_command, strip_cmd_prefix, parse_args
from ..vault_ops import summ_new_in_kb, move_summ_to_vault

# Optional imports (feature-detected)
try:
    from ..fs_rag import build_fs_facts_block  # type: ignore
except Exception:
    build_fs_facts_block = None  # type: ignore

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


def load_help_text() -> str:
    """Load command cheat sheet."""
    try:
        with open(CHEAT_SHEET_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[help missing: {e}]"


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

    # status/help
    if low == "status":
        return (
            "[status]\n"
            f"session_id={session_id}\n"
            f"attached_kbs={sorted(state.attached_kbs)}\n"
            f"fun_sticky={state.fun_sticky}\n"
            f"fun_rewrite_sticky={state.fun_rewrite_sticky}\n"
            f"last_query={state.rag_last_query!r}\n"
            f"last_hits={state.rag_last_hits}\n"
            f"vault_last_query={state.vault_last_query!r}\n"
            f"vault_last_hits={state.vault_last_hits}\n"
        )

    if low == "help":
        return load_help_text()

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
            # attach all known non-empty kb_paths
            for k in sorted(KB_PATHS.keys()):
                if k:
                    state.attached_kbs.add(k)
            state.attached_kbs.add("scratchpad")
            return f"[router] attached ALL: {sorted(state.attached_kbs)}"

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
            if "scratchpad" in state.attached_kbs and list_scratchpad_records and clear_scratchpad:
                recs = list_scratchpad_records(session_id, limit=1000000)
                scratchpad_count = len(recs)
                scratchpad_deleted = bool(clear_scratchpad(session_id))
            state.attached_kbs.clear()
            if scratchpad_deleted:
                return f"[router] detached ALL and deleted scratchpad data ({scratchpad_count} records dumped)"
            return "[router] detached ALL"
        if target in state.attached_kbs:
            if target == "scratchpad" and list_scratchpad_records and clear_scratchpad:
                recs = list_scratchpad_records(session_id, limit=1000000)
                n = len(recs)
                deleted = bool(clear_scratchpad(session_id))
                state.attached_kbs.remove(target)
                if not deleted:
                    return "[router] detached 'scratchpad' (warning: scratchpad delete failed)"
                return f"[router] detached 'scratchpad' and deleted scratchpad data ({n} records dumped)"
            state.attached_kbs.remove(target)
            return f"[router] detached '{target}'"
        return f"[router] kb not attached: '{target}'"

    if low in ("list_kb", "list", "kbs"):
        known = sorted(set(KB_PATHS.keys()) | {"scratchpad"})
        return "[router] known KBs: " + ", ".join(known)

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
        if not flush_ctc_cache:
            return "[router] flush not available"
        if not state.vodka:
            try:
                from ..vodka_filter import Filter as VodkaFilter  # type: ignore
                state.vodka = VodkaFilter()
            except Exception:
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
