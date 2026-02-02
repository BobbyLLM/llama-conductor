# vodka_filter.py
# version 1.0.4
"""
Vodka v1.0.4 (no code changes - version bump for alignment)
- Debug logging support (debug_log method, v1.0.3)
- Graceful %Z timestamp parsing with fallback 
- Deterministic memory store to JSON (Total Recall)
- Manual save: "!! ... !!" OR message starts/ends with "!!"
- Commands:
    - "!! nuke" / "nuke !!"          -> delete all ctx notes, return early assistant reply
    - "!! forget <query>"            -> delete matching ctx notes, return early assistant reply
    - "?? <query>"                   -> rewrite last user msg into a memory-backed question block
- Breadcrumb expansion:
    - Any "[ctx:...]" markers are expanded on outlet()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field

import os
import re
import time
import datetime as dt

FILTER_VERSION = "1.0.3"
SUMMARY_PREFIX = "[CHAT_SUMMARY] "

# -------------------------------
# JSON backend (minimal)
# -------------------------------

try:
    import orjson as _json

    def _dumps(obj: Any) -> str:
        return _json.dumps(obj).decode("utf-8")

    def _loads(s: str) -> Any:
        return _json.loads(s)

except Exception:
    import json as _json

    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, ensure_ascii=False)

    def _loads(s: str) -> Any:
        return _json.loads(s)


# -------------------------------
# Helpers
# -------------------------------

def _ts() -> str:
    return dt.datetime.now().astimezone().strftime("%B %d, %Y at %H:%M:%S %Z")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _sha256_crc_id(text: str) -> str:
    """Deterministic ID from content: sha256 prefix + crc16-ish."""
    import hashlib
    import zlib

    b = text.encode("utf-8", errors="ignore")
    h = hashlib.sha256(b).hexdigest()[:8]
    crc = zlib.crc32(b) & 0xFFFF
    return f"ctx_{h}_{crc:04x}"


def _is_all_caps_chunk(s: str, min_len: int = 8) -> bool:
    letters = re.sub(r"[^A-Za-z]+", "", s)
    if len(letters) < min_len:
        return False
    return letters.upper() == letters


def _extract_manual_highlights(content: str) -> List[str]:
    """
    Rules:
    - If message startswith '!!' OR endswith '!!' -> store whole message (without !!)
    - If message is ALLCAPS-ish (min_len=8) -> store whole message
    """
    text = (content or "").strip()
    if not text:
        return []

    if text.startswith("!!") or text.endswith("!!"):
        cleaned = text.strip().strip("!").strip()
        return [cleaned] if cleaned else []

    if _is_all_caps_chunk(text):
        return [text]

    return []


def _expand_breadcrumbs_in_text(text: str, fr: "FastRecall") -> str:
    pattern = re.compile(r"\[ctx:([a-z0-9_]+)\]")

    def _repl(match: re.Match) -> str:
        ctx_id = match.group(1)
        rec = fr.retrieve_group_with_meta(ctx_id)
        if rec is None:
            return "(expired memory)"
        val, ttl_days, touch_count = rec
        return f"{val} [TTL = {ttl_days}, Touch count = {touch_count}]"

    return pattern.sub(_repl, text)

# -------------------------------
# Storage backend
# -------------------------------

class Storage:
    def __init__(self, base_dir: str, debug: bool = False):
        self.base_root = base_dir
        self.base = _ensure_dir(os.path.join(base_dir, "total_recall"))
        self.facts_file = os.path.join(self.base, "facts.json")
        self.log_file = os.path.join(self.base, "activity.log")
        self.debug_file = os.path.join(self.base, "vodka_debug.log") if debug else None
        self.debug = debug

        if not os.path.exists(self.facts_file):
            self._atomic_write(self.facts_file, {})

    def _atomic_write(self, path: str, obj: Any):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_dumps(obj))
        os.replace(tmp, path)

    def log(self, msg: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{_ts()} â€” {msg}\n")
        except Exception:
            pass

    def debug_log(self, msg: str):
        """Write to debug log if debug mode is enabled."""
        if not self.debug or not self.debug_file:
            return
        try:
            with open(self.debug_file, "a", encoding="utf-8") as f:
                f.write(f"{_ts()} [DEBUG] {msg}\n")
        except Exception:
            pass

    def load_facts(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.facts_file, "r", encoding="utf-8") as f:
                return _loads(f.read())
        except Exception:
            self._atomic_write(self.facts_file, {})
            return {}

    def save_facts(self, data: Dict[str, Dict[str, Any]]):
        self._atomic_write(self.facts_file, data)


# -------------------------------
# FastRecall
# -------------------------------

class FastRecall:
    """
    Stores ctx entries in facts.json:
      {
        "ctx_xxx": {
           "value": "...",
           "created_at": "...",
           "expires_at": "...",
           "touch_count": 0,
           "type": "vodka_ctx"
        }
      }
    """

    def __init__(
        self,
        storage: Storage,
        base_ttl_days: int = 3,
        touch_extension_days: int = 5,
        max_touches: int = 2,
        max_items: int = 3000,
    ):
        self.S = storage
        self.base_ttl_days = max(1, min(int(base_ttl_days), 30))
        self.touch_extension_days = max(1, min(int(touch_extension_days), 30))
        self.max_touches = max(0, min(int(max_touches), 3))
        self.max_items = max(500, min(int(max_items), 5000))
        self._last_janitor_run = 0.0

    def _now(self) -> dt.datetime:
        return dt.datetime.now().astimezone()

    def _fmt_ts(self, dt_obj: dt.datetime) -> str:
        return dt_obj.strftime("%B %d, %Y at %H:%M:%S %Z")

    def _parse_ts(self, s: str) -> Optional[dt.datetime]:
        try:
            return dt.datetime.strptime(s, "%B %d, %Y at %H:%M:%S %Z").replace(
                tzinfo=dt.datetime.now().astimezone().tzinfo
            )
        except Exception:
            return None

    def store_overflow(self, content: str) -> str:
        content = (content or "").strip()
        if not content:
            return ""

        data = self.S.load_facts()
        ctx_id = _sha256_crc_id(content)
        now = self._now()

        if ctx_id in data:
            rec = data[ctx_id]
            rec.setdefault("type", "vodka_ctx")
            rec.setdefault("touch_count", 0)
            exp = now + dt.timedelta(days=self.base_ttl_days)
            rec["expires_at"] = self._fmt_ts(exp)
            self.S.log(f"VODKA_REFRESH_CTX â€” {ctx_id}")
            self.S.debug_log(f"REFRESH: {ctx_id}")
        else:
            exp = now + dt.timedelta(days=self.base_ttl_days)
            data[ctx_id] = {
                "value": content,
                "created_at": self._fmt_ts(now),
                "expires_at": self._fmt_ts(exp),
                "touch_count": 0,
                "type": "vodka_ctx",
            }
            self.S.log(f"VODKA_ADD_CTX â€” {ctx_id} â€” {content[:80].replace(os.linesep, ' ')}")
            self.S.debug_log(f"ADD: {ctx_id} value_len={len(content)}")

        self.S.save_facts(data)
        self.S.debug_log(f"SAVE_FACTS: {len(data)} entries")
        return ctx_id

    def retrieve_group(self, ctx_id: str) -> Optional[str]:
        data = self.S.load_facts()
        rec = data.get(ctx_id)
        if not rec or rec.get("type") != "vodka_ctx":
            return None

        exp_s = rec.get("expires_at")
        if exp_s:
            exp_dt = self._parse_ts(exp_s)
            if exp_dt and exp_dt < self._now():
                return None

        self._touch(ctx_id, rec, data)
        self.S.save_facts(data)
        self.S.debug_log(f"SAVE_FACTS: {len(data)} entries")
        return str(rec.get("value", "") or "")

    def retrieve_group_with_meta(self, ctx_id: str):
        """
        Retrieve a ctx note and apply touch side-effects.
        Returns (value, ttl_days_remaining, touch_count) or None if missing/expired.
        """
        data = self.S.load_facts()
        rec = data.get(ctx_id)
        if not rec or rec.get("type") != "vodka_ctx":
            return None

        exp_s = rec.get("expires_at")
        if exp_s:
            exp_dt = self._parse_ts(exp_s)
            if exp_dt and exp_dt < self._now():
                return None

        # Apply touch side-effects
        self._touch(ctx_id, rec, data)
        self.S.save_facts(data)
        self.S.debug_log(f"SAVE_FACTS: {len(data)} entries")

        # Compute TTL days remaining from expires_at
        ttl_days = 0
        exp_s2 = rec.get("expires_at")
        if exp_s2:
            exp_dt2 = self._parse_ts(exp_s2)
            if exp_dt2:
                delta = (exp_dt2 - self._now()).total_seconds()
                ttl_days = max(0, int(delta // 86400))
        touch_count = int(rec.get("touch_count", 0))
        return (str(rec.get("value", "") or ""), ttl_days, touch_count)

    def _touch(self, ctx_id: str, rec: Dict[str, Any], data: Dict[str, Any]):
        tc = int(rec.get("touch_count", 0))
        if tc >= self.max_touches:
            return
        tc += 1
        rec["touch_count"] = tc
        now = self._now()
        exp = now + dt.timedelta(days=self.touch_extension_days)
        rec["expires_at"] = self._fmt_ts(exp)
        self.S.log(f"VODKA_TOUCH_CTX â€” {ctx_id} â€” touch_count={tc}")
        self.S.debug_log(f"TOUCH: {ctx_id} count={tc}")

    def janitor_if_due(self, interval_seconds: int = 3600):
        now = time.time()
        if now - self._last_janitor_run < interval_seconds:
            return
        self._last_janitor_run = now
        self._janitor()

    def _janitor(self):
        data = self.S.load_facts()
        now = self._now()

        # expire
        to_delete: List[str] = []
        for k, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue
            exp_s = rec.get("expires_at")
            if not exp_s:
                continue
            exp_dt = self._parse_ts(exp_s)
            if exp_dt and exp_dt < now:
                to_delete.append(k)
        for k in to_delete:
            v_preview = str(data.get(k, {}).get("value", ""))[:80].replace(os.linesep, " ")
            self.S.log(f"VODKA_JANITOR_EXPIRE â€” {k} â€” {v_preview}")
            self.S.debug_log(f"JANITOR: {k}")
            data.pop(k, None)

        # cap
        ctx_keys = [k for k, rec in data.items() if rec.get("type") == "vodka_ctx"]
        if len(ctx_keys) > self.max_items:
            def _key_created(kid: str) -> str:
                return str(data.get(kid, {}).get("created_at", ""))

            ctx_keys_sorted = sorted(ctx_keys, key=_key_created)
            overflow = ctx_keys_sorted[: len(ctx_keys_sorted) - self.max_items]
            for k in overflow:
                v_preview = str(data.get(k, {}).get("value", ""))[:80].replace(os.linesep, " ")
                self.S.log(f"VODKA_JANITOR_CAP â€” {k} â€” {v_preview}")
                self.S.debug_log(f"JANITOR: {k}")
                data.pop(k, None)

        self.S.save_facts(data)
        self.S.debug_log(f"SAVE_FACTS: {len(data)} entries")

    def nuke(self):
        data = self.S.load_facts()
        keys = [k for k, rec in data.items() if rec.get("type") == "vodka_ctx"]
        for k in keys:
            data.pop(k, None)
        self.S.save_facts(data)
        self.S.debug_log(f"SAVE_FACTS: {len(data)} entries")
        self.S.log("VODKA_NUKE_ALL")


# -------------------------------
# Vodka Filter
# -------------------------------

class Filter:
    class Valves(BaseModel):
        # CTC clipping
        n_last_messages: int = Field(default=2, description="Last user/assistant PAIRS to keep.")
        keep_first: bool = Field(default=True, description="Keep the first user+assistant pair.")
        max_chars: int = Field(default=1500, description="Soft cap for non-system message chars (0 disables).")

        # Rolling summary (kept simple; if you donâ€™t want it, disable)
        enable_summary: bool = Field(default=True, description="Enable rolling summary system message.")
        summary_every_n_user_msgs: int = Field(default=4, description="Update summary once >= N user msgs.")
        summary_max_words: int = Field(default=160, description="Max words in summary.")

        # Debug
        debug: bool = Field(default=False, description="Enable debug prints.")
        debug_dir: str = Field(default="", description="Optional debug dump dir.")

        # Storage / TTL
        storage_dir: str = Field(default="", description="Base dir for total_recall storage.")
        base_ttl_days: int = Field(default=3, description="Base TTL for new ctx notes.")
        touch_extension_days: int = Field(default=5, description="TTL extension per touch.")
        max_touches: int = Field(default=2, description="Max touch extensions.")
        max_ctx_items: int = Field(default=3000, description="Max ctx entries.")

    def __init__(self):
        self.valves = self.Valves()
        print(f"[Vodka v{FILTER_VERSION}] __init__ called, valves initialized")
        self._storage: Optional[Storage] = None
        self._fr: Optional[FastRecall] = None
        # Eager-load facts.json on init to eliminate warm-up problem
        self._get_storage_and_fr()  # Force load storage and FastRecall immediately

    def _get_storage_and_fr(self) -> FastRecall:
        base = self.valves.storage_dir or os.getenv("DATA_DIR") or os.getcwd()
        base = os.path.abspath(base)
        _ensure_dir(base)

        if self._storage is None:
            self._storage = Storage(base, debug=bool(self.valves.debug))

        base_ttl = max(1, min(int(self.valves.base_ttl_days or 3), 30))
        ext_ttl = max(1, min(int(self.valves.touch_extension_days or 5), 30))
        max_touches = max(0, min(int(self.valves.max_touches or 2), 3))
        max_items = max(500, min(int(self.valves.max_ctx_items or 3000), 5000))

        if self._fr is None:
            self._fr = FastRecall(
                storage=self._storage,
                base_ttl_days=base_ttl,
                touch_extension_days=ext_ttl,
                max_touches=max_touches,
                max_items=max_items,
            )
        else:
            self._fr.base_ttl_days = base_ttl
            self._fr.touch_extension_days = ext_ttl
            self._fr.max_touches = max_touches
            self._fr.max_items = max_items

        return self._fr

    # -------------------------------
    # Memory search / delete
    # -------------------------------

    def _tokenize_for_search(self, text: str) -> set[str]:
        stopwords = {
            "the","a","an","and","or","but","if","then","else","what","when","where","which",
            "who","whom","how","did","do","does","is","are","was","were","be","been",
            "to","of","in","on","for","with","about","this","that","these","those","it","its","as","at","by","from",
        }
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        return {t for t in tokens if len(t) > 2 and t not in stopwords}

    def _search_vodka_memories(self, fr: FastRecall, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        data = fr.S.load_facts()
        now = fr._now()
        q_tokens = self._tokenize_for_search(query)
        if not q_tokens:
            return []

        scored: List[Tuple[float, str, str]] = []
        for ctx_id, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue

            exp_s = rec.get("expires_at")
            if exp_s:
                exp_dt = fr._parse_ts(exp_s)
                if exp_dt and exp_dt < now:
                    continue

            value = str(rec.get("value", "")).strip()
            if not value:
                continue

            mem_tokens = self._tokenize_for_search(value)
            overlap = q_tokens & mem_tokens
            if not overlap:
                continue

            base = len(overlap) / len(q_tokens)
            bonus = 0.5 if (query or "").lower() in value.lower() else 0.0
            score = base + bonus
            scored.append((score, ctx_id, value))

        if not scored:
            return []
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:max_results]
        return [{"id": cid, "value": val, "score": sc} for sc, cid, val in top]

    def _delete_vodka_memories(self, fr: FastRecall, query: str, max_results: int = 50) -> int:
        q_raw = (query or "").strip()
        q = q_raw.lower()
        if not q:
            return 0

        data = fr.S.load_facts()
        to_delete: List[str] = []
        short_query = len(q) <= 2
        pattern = None
        if short_query:
            pattern = re.compile(r"\b" + re.escape(q) + r"\b", re.IGNORECASE)

        for ctx_id, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue
            value = str(rec.get("value", ""))
            if short_query:
                if pattern and pattern.search(value):
                    to_delete.append(ctx_id)
            else:
                if q in value.lower():
                    to_delete.append(ctx_id)
            if len(to_delete) >= max_results:
                break

        deleted = 0
        for ctx_id in to_delete:
            if ctx_id in data:
                del data[ctx_id]
                deleted += 1

        if deleted:
            fr.S.save_facts(data)
            fr.S.log(f"VODKA_FORGET_QUERY â€” '{query}' â€” deleted={deleted}")
        return deleted

    def _list_all_memories(self, fr: FastRecall, debug_enabled: bool) -> str:
        """
        List ALL stored Vodka memories with metadata (TTL, touch count).
        Used by: ?? list
        Returns: Formatted pretty-print string suitable for model context.
        """
        data = fr.S.load_facts()
        now = fr._now()
        
        # Collect all non-expired memories with metadata
        entries: List[Tuple[str, str, str, int, int]] = []  # (ctx_id, text, created_at, ttl_days, touch_count)
        
        for ctx_id, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue
            
            # Skip expired memories
            exp_s = rec.get("expires_at")
            if exp_s:
                exp_dt = self._parse_ts_safe(exp_s)
                if exp_dt and exp_dt < now:
                    continue
            
            # Calculate TTL days remaining
            ttl_days = 0
            if exp_s:
                exp_dt = self._parse_ts_safe(exp_s)
                if exp_dt:
                    delta = (exp_dt - now).total_seconds()
                    ttl_days = max(0, int(delta // 86400))
            
            text = str(rec.get("value", "")).strip()
            created_at = str(rec.get("created_at", "")).strip()
            touch_count = int(rec.get("touch_count", 0))
            
            entries.append((ctx_id, text, created_at, ttl_days, touch_count))
        
        # Sort by creation date (newest first)
        entries.sort(key=lambda e: e[2], reverse=True)
        
        if not entries:
            # No memories stored
            output = (
                "You have no stored memories. "
                "Create memories by prefixing your text with '!!' (e.g., '!! my server is at 192.168.1.1').\n"
            )
        else:
            # Format all memories with metadata
            lines = ["[Vodka Memory Store]", ""]
            for i, (ctx_id, text, created_at, ttl_days, touch_count) in enumerate(entries, 1):
                # Truncate long text for preview
                text_preview = text if len(text) <= 80 else text[:77] + "..."
                lines.append(
                    f"{i}. [{ctx_id}]"
                )
                lines.append(
                    f"   Text: {text_preview}"
                )
                lines.append(
                    f"   Created: {created_at}"
                )
                lines.append(
                    f"   TTL: {ttl_days} days remaining | Touches: {touch_count}"
                )
                lines.append("")
            
            output = "\n".join(lines).strip()
        
        if debug_enabled:
            print(f"[Vodka] listed {len(entries)} memories")
        
        return output

    def _parse_ts_safe(self, s: str) -> Optional[dt.datetime]:
        """
        Parse timestamp safely, handling both old and new formats.
        Tries multiple formats to handle timezone variations.
        """
        if not s:
            return None
        
        # Try original format first (with timezone code)
        try:
            return dt.datetime.strptime(s, "%B %d, %Y at %H:%M:%S %Z").replace(
                tzinfo=dt.datetime.now().astimezone().tzinfo
            )
        except Exception:
            pass
        
        # Fallback: parse without timezone (just date and time)
        # This handles "January 27, 2026 at 02:34:51 W. Australia Standard Time"
        try:
            # Strip the timezone suffix and parse just the datetime part
            # Format: "Month DD, YYYY at HH:MM:SS TZ_NAME"
            parts = s.rsplit(" at ", 1)  # Split on the last " at "
            if len(parts) == 2:
                date_part = parts[0]  # "January 27, 2026"
                time_part = parts[1].split(" ", 1)[0]  # "02:34:51"
                
                dt_str = f"{date_part} at {time_part}"
                parsed = dt.datetime.strptime(dt_str, "%B %d, %Y at %H:%M:%S")
                # Add current timezone
                return parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
        except Exception:
            pass
        
        # If all else fails, return None
        return None


    # -------------------------------
    # Control commands + highlights
    # -------------------------------

    def _find_last_user_index(self, messages: List[Dict]) -> Optional[int]:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return None

    def _handle_control_commands(
        self,
        messages: List[Dict],
        body: dict,
        fr: FastRecall,
        debug_enabled: bool,
    ) -> Tuple[List[Dict], Optional[dict]]:
        """
        Returns (messages, early_body_or_none)
        """
        idx = self._find_last_user_index(messages)
        if idx is None:
            return messages, None

        content = messages[idx].get("content", "")
        if not isinstance(content, str):
            return messages, None

        # DEFENSIVE: Normalize broken UTF-8 sequences (Â» → ??, Â» → >>)
        # This handles cases where clients send malformed Unicode
        content = content.replace("Â»", ">>").replace("Â»", ">>")  # broken guillemet variants
        content = content.replace("Â¿", "??").replace("Â¿", "??")  # other broken variants
        
        stripped = content.strip()
        norm = stripped.lower()

        # !! nuke / nuke !!
        if norm in ("!! nuke", "nuke !!"):
            fr.nuke()
            fr.S.log("VODKA_NUKE_CMD")
            # remove the user command message
            del messages[idx]
            # add assistant confirmation so router/UI gets a response without LLM
            messages.append({"role": "assistant", "content": "[vodka] nuked all stored notes"})
            body["messages"] = messages
            if debug_enabled:
                print("[Vodka] nuke executed")
            return messages, body

        # !! forget <query>
        if norm.startswith("!! forget"):
            query = stripped[len("!! forget"):].strip()
            deleted = self._delete_vodka_memories(fr, query, max_results=50)
            # remove the user command message
            del messages[idx]
            messages.append({"role": "assistant", "content": f"[vodka] forget='{query}' deleted={deleted}"})
            body["messages"] = messages
            if debug_enabled:
                print(f"[Vodka] forget executed: query='{query}' deleted={deleted}")
            return messages, body

        # ?? <query>  -> rewrite last user message into memory-backed block (no early return)
        if stripped.startswith("??"):
            query = stripped[2:].strip()
            
            # SPECIAL CASE: ?? list -> list all memories with metadata (HARD COMMAND, no LLM)
            if query.lower() in ("list", ""):
                list_output = self._list_all_memories(fr, debug_enabled)
                # Remove the user command message
                del messages[idx]
                # Add assistant reply with the list directly (deterministic, no LLM processing)
                messages.append({"role": "assistant", "content": list_output})
                body["messages"] = messages
                if debug_enabled:
                    print(f"[Vodka] ?? list executed (hard command, deterministic output)")
                return messages, body
            
            matches = self._search_vodka_memories(fr, query, max_results=10)

            if matches:
                # Touch each memory exactly once and append TTL/touch metadata
                seen = set()
                mem_lines = []
                for m in matches:
                    cid = m.get("id")
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    rec = fr.retrieve_group_with_meta(cid)
                    if rec is None:
                        continue
                    val, ttl_days, touch_count = rec
                    mem_lines.append(f"{val} [TTL = {ttl_days}, Touch count = {touch_count}]")
                mem_block = "\n".join(mem_lines) if mem_lines else ""
                if mem_block:
                    new_content = (
                        "The user is asking a question that should be answered using the stored notes below.\n\n"
                        f"User's question: {query or '(no explicit question)'}\n\n"
                        "Stored notes (verbatim):\n"
                        f"{mem_block}\n\n"
                        "Instructions:\n"
                        "- Use ONLY the information in the stored notes above when referring to past facts.\n"
                        "- If the stored notes do not contain enough information to answer, explicitly say the notes do not say.\n"
                        "- Do NOT invent or alter any of the stored text.\n"
                    )
                else:
                    new_content = (
                        "The user attempted a memory lookup, but the matching notes were expired.\n\n"
                        f"User's question: {query or '(no explicit question)'}\n\n"
                        "Instructions:\n"
                        "- There are no valid stored notes for this query.\n"
                        "- Answer only from the current conversation, or say you do not know.\n"
                        "- Do NOT claim you remember notes that are not shown.\n"
                    )
            else:
                new_content = (
                    "The user attempted a memory lookup, but no stored notes matched their query.\n\n"
                    f"User's question: {query or '(no explicit question)'}\n\n"
                    "Instructions:\n"
                    "- There are no relevant stored notes for this query.\n"
                    "- Answer only from the current conversation, or say you do not know.\n"
                    "- Do NOT claim you remember notes that are not shown.\n"
                )

            messages[idx]["content"] = new_content
            if debug_enabled:
                print(f"[Vodka] ?? rewritten for query='{query}' matches={len(matches)}")

        return messages, None

    def _store_manual_highlights(self, messages: List[Dict], fr: FastRecall, debug_enabled: bool):
        idx = self._find_last_user_index(messages)
        if idx is None:
            return

        content = messages[idx].get("content", "")
        if not isinstance(content, str):
            return

        # Don't store highlights for explicit control commands (nuke/forget/??)
        s = content.strip()
        low = s.lower()
        if low in ("!! nuke", "nuke !!") or low.startswith("!! forget") or s.startswith("??"):
            return

        highlights = _extract_manual_highlights(content)
        if not highlights:
            return

        # Store AND strip !! markers from the message so the model doesn't see them
        # (still leaves the core text intact)
        if s.startswith("!!") or s.endswith("!!"):
            cleaned = s.strip().strip("!").strip()
            messages[idx]["content"] = cleaned

        for h in highlights:
            ctx_id = fr.store_overflow(h)
            if debug_enabled:
                print(f"[Vodka] stored highlight ctx={ctx_id}")

    # -------------------------------
    # Summary + clipping
    # -------------------------------

    def _count_user_messages(self, messages: List[Dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")

    def _build_summary_text(self, messages: List[Dict], max_words: int) -> str:
        """
        Deterministic cheap summary: just the last few user+assistant lines truncated.
        (You can replace with LLM summary later; this keeps it safe/offline.)
        """
        ua = [m for m in messages if m.get("role") in ("user", "assistant")]
        tail = ua[-6:]  # last 3 turns-ish
        lines: List[str] = []
        for m in tail:
            role = m.get("role")
            c = m.get("content", "")
            if not isinstance(c, str):
                c = str(c)
            c = c.strip().replace("\n", " ")
            if len(c) > 240:
                c = c[:240] + "..."
            lines.append(f"{role}: {c}")
        text = " | ".join(lines).strip()

        # word cap
        if max_words and max_words > 0:
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]) + " ..."
        return text

    def _upsert_summary_message(self, messages: List[Dict], max_words: int, debug_enabled: bool) -> List[Dict]:
        # Remove existing summary messages
        cleaned = []
        for m in messages:
            c = m.get("content", "")
            if m.get("role") in ("system", "assistant") and isinstance(c, str) and c.startswith(SUMMARY_PREFIX):
                continue
            cleaned.append(m)

        summary = self._build_summary_text(cleaned, max_words=max_words)
        if not summary:
            return cleaned

        summary_msg = {"role": "system", "content": SUMMARY_PREFIX + summary}
        if debug_enabled:
            print("[Vodka] summary updated")
        return [summary_msg] + cleaned

    def _clip_and_trim_messages(self, messages: List[Dict], n_last_pairs: int, keep_first: bool, max_chars: int) -> List[Dict]:
        system = [m for m in messages if m.get("role") == "system"]
        ua = [m for m in messages if m.get("role") in ("user", "assistant")]

        first_pair: List[Dict] = []
        if keep_first:
            first_user = next((m for m in ua if m.get("role") == "user"), None)
            first_asst = next((m for m in ua if m.get("role") == "assistant"), None)
            if first_user:
                first_pair.append(first_user)
            if first_asst and first_asst is not first_user:
                first_pair.append(first_asst)

        if n_last_pairs > 0:
            keep_n = 2 * n_last_pairs
            recent = ua[-keep_n:]
        else:
            recent = ua[:]

        # remove duplicates of first pair from recent
        if keep_first and first_pair:
            recent = [m for m in recent if m not in first_pair]

        out = system + first_pair + recent

        if max_chars and max_chars > 0:
            trimmed: List[Dict] = []
            used = 0
            last_user = None
            for m in reversed(out):
                if m.get("role") == "user":
                    last_user = m
                    break

            for m in out:
                if m.get("role") == "system":
                    trimmed.append(m)
                    continue
                c = m.get("content", "")
                if not isinstance(c, str):
                    c = str(c)
                if used + len(c) <= max_chars:
                    trimmed.append(m)
                    used += len(c)

            # Guarantee latest user survives
            if last_user and last_user not in trimmed:
                trimmed.append(last_user)

            out = trimmed

        return out

    # -------------------------------
    # Public API: inlet/outlet
    # -------------------------------

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages: List[Dict] = body.get("messages", []) or []
        debug_enabled = bool(self.valves.debug)

        fr = self._get_storage_and_fr()
        fr.janitor_if_due(interval_seconds=3600)

        if not messages:
            return body

        # 0) control commands (may early return)
        messages, early_body = self._handle_control_commands(
            messages=messages,
            body=body,
            fr=fr,
            debug_enabled=debug_enabled,
        )
        if early_body is not None:
            return early_body

        # 1) manual highlights (store !! / ALLCAPS)
        self._store_manual_highlights(messages, fr, debug_enabled)

        # 2) summary (optional)
        if bool(self.valves.enable_summary):
            n_user = self._count_user_messages(messages)
            every_n = int(self.valves.summary_every_n_user_msgs or 0)
            if every_n > 0 and n_user >= every_n:
                messages = self._upsert_summary_message(
                    messages=messages,
                    max_words=int(self.valves.summary_max_words or 160),
                    debug_enabled=debug_enabled,
                )

        # 3) clip
        messages = self._clip_and_trim_messages(
            messages=messages,
            n_last_pairs=max(0, int(self.valves.n_last_messages or 0)),
            keep_first=bool(self.valves.keep_first),
            max_chars=max(0, int(self.valves.max_chars or 0)),
        )

        body["messages"] = messages
        return body

    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Expand [ctx:ID] markers so the model sees verbatim stored text.
        """
        fr = self._get_storage_and_fr()
        messages: List[Dict] = body.get("messages", []) or []
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and "[ctx:" in c:
                m["content"] = _expand_breadcrumbs_in_text(c, fr)
        body["messages"] = messages
        return body


# Public alias for router code
VodkaFilter = Filter
