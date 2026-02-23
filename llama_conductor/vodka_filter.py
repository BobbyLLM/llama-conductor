"""Deterministic short-term memory filter (Vodka).

Provides:
- memory save/recall/delete controls (`!!`, `??`, forget/nuke)
- compact context trimming for prompt hygiene
- breadcrumb expansion during output processing
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

import os
import re
import time
import datetime as dt
from .privacy_utils import contains_likely_pii, redact_pii, safe_preview, short_hash

FILTER_VERSION = "1.0.3"
SUMMARY_PREFIX = "[CHAT_SUMMARY] "
MEMORY_PREFIX = "[SESSION_MEMORY] "

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
            self.S.log(f"VODKA_ADD_CTX â€” {ctx_id} â€” len={len(content)}")
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
            self.S.log(f"VODKA_JANITOR_EXPIRE â€” {k}")
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
                self.S.log(f"VODKA_JANITOR_CAP â€” {k}")
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
        summary_inject_max_units: int = Field(default=3, description="Max session-memory units to inject per turn.")
        summary_inject_max_chars: int = Field(default=500, description="Max chars injected by session memory.")
        summary_memory_max_units: int = Field(default=96, description="Max retained session-memory units.")
        summary_segments_per_message: int = Field(default=3, description="Max memory segments extracted per message.")
        summary_require_session_id: bool = Field(default=True, description="Only use session memory when a session_id is provided by client.")
        summary_include_assistant: bool = Field(default=False, description="Include assistant messages in session-memory units.")
        summary_store_pii: bool = Field(default=False, description="Allow storing likely PII in session-memory units.")
        summary_pii_redaction_token: str = Field(default="[REDACTED]", description="Replacement token when PII is detected in memory segment.")

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
        self._mu_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._mu_index: Dict[str, Dict[str, set[str]]] = {}
        self._mu_last_update_turn: Dict[str, int] = {}
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
            fr.S.log(
                f"VODKA_FORGET_QUERY â€” qhash={short_hash(query)} qlen={len(query)} deleted={deleted}"
            )
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
            # Keep the same deterministic header so router hard-return logic is consistent.
            output = (
                "[Vodka Memory Store]\n\n"
                "No stored memories.\n"
                "Create memories by prefixing your text with '!!' "
                "(e.g., '!! my server is at 192.168.1.1').\n"
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
        norm_no_ws = re.sub(r"\s+", "", norm)

        # !! nuke / nuke !!
        if norm_no_ws in ("!!nuke", "nuke!!"):
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
        forget_match = re.match(r"^!!\s*forget(?:\s+(.*))?$", stripped, flags=re.IGNORECASE)
        if forget_match:
            query = (forget_match.group(1) or "").strip()
            deleted = self._delete_vodka_memories(fr, query, max_results=50)
            # remove the user command message
            del messages[idx]
            messages.append({"role": "assistant", "content": f"[vodka] forget='{query}' deleted={deleted}"})
            body["messages"] = messages
            if debug_enabled:
                print(
                    f"[Vodka] forget executed: qhash={short_hash(query)} qlen={len(query)} deleted={deleted}"
                )
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
                print(
                    f"[Vodka] ?? rewritten for qhash={short_hash(query)} qlen={len(query)} matches={len(matches)}"
                )

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
        low_no_ws = re.sub(r"\s+", "", low)
        is_forget_cmd = re.match(r"^!!\s*forget(?:\s+.*)?$", s, flags=re.IGNORECASE) is not None
        if low_no_ws in ("!!nuke", "nuke!!") or is_forget_cmd or s.startswith("??"):
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
            if self._contains_likely_pii(h):
                if debug_enabled:
                    print(f"[Vodka] skipped highlight (possible PII) hash={short_hash(h)}")
                continue
            ctx_id = fr.store_overflow(h)
            if debug_enabled:
                print(f"[Vodka] stored highlight ctx={ctx_id}")

    # -------------------------------
    # Summary + clipping
    # -------------------------------

    def _count_user_messages(self, messages: List[Dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")

    def _normalize_session_id(self, session_id: str) -> str:
        raw = str(session_id or "").strip().lower()
        if not raw:
            return "global"
        return re.sub(r"[^a-z0-9._-]+", "_", raw)[:120] or "global"

    def _session_memory_file(self, session_id: str) -> str:
        fr = self._get_storage_and_fr()
        sid = self._normalize_session_id(session_id)
        folder = _ensure_dir(os.path.join(fr.S.base, "session_memory"))
        return os.path.join(folder, f"{sid}.jsonl")

    def _compact_memory_text(self, value: Any, max_len: int = 220) -> str:
        txt = str(value or "").strip().replace("\n", " ")
        if len(txt) > max_len:
            if max_len <= 40:
                return txt[: max(0, max_len - 3)].rstrip() + "..."
            sep = " ... "
            head = max(20, int((max_len - len(sep)) * 0.6))
            tail = max(12, max_len - len(sep) - head)
            return txt[:head].rstrip() + sep + txt[-tail:].lstrip()
        return txt

    def _split_memory_segments(self, value: Any, *, max_segments: int = 3, max_len: int = 220) -> List[str]:
        raw = str(value or "").strip().replace("\n", " ")
        if not raw:
            return []
        if len(raw) <= max_len:
            return [raw]

        # Prefer sentence boundaries; fallback to clause boundaries.
        segs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        if len(segs) <= 1:
            segs = [s.strip() for s in re.split(r"[;,]\s+", raw) if s.strip()]
        if not segs:
            segs = [raw]

        # If a segment is still too long, split by clause delimiters so we
        # retain mid-sentence entities instead of truncating them away.
        expanded: List[str] = []
        for seg in segs:
            if len(seg) <= max_len:
                expanded.append(seg)
                continue
            parts = [p.strip() for p in re.split(r"[;,]\s+", seg) if p.strip()]
            if len(parts) <= 1:
                expanded.append(seg)
                continue
            chunk = ""
            for part in parts:
                candidate = part if not chunk else (chunk + "; " + part)
                if len(candidate) <= max_len:
                    chunk = candidate
                else:
                    if chunk:
                        expanded.append(chunk)
                    chunk = part
            if chunk:
                expanded.append(chunk)
        segs = expanded or segs

        if len(segs) <= max_segments:
            return [self._compact_memory_text(s, max_len=max_len) for s in segs]

        # Keep first/last for chronology, then choose salient middle segments.
        def _salience(s: str) -> int:
            low = s.lower()
            score = 0
            if "promis" in low:
                score += 4
            if any(k in low for k in ("asked", "gave", "will", "plan", "need", "prefer", "replace", "mentioned", "promised")):
                score += 2
            if any(
                k in low
                for k in (
                    "software",
                    "apps",
                    "app",
                    "firewall",
                    "pi-hole",
                    "smart-tube",
                    "browser",
                    "kaios",
                    "jellyfin",
                    "timer",
                )
            ):
                score += 5
            if any(k in low for k in ("phone", "device", "technology", "tablet", "controller", "console")):
                score += 3
            # Entity-like tokens (proper nouns/acronyms/numbers) are often high-value recall anchors.
            score += len(re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", s))
            score += len(re.findall(r"\b[A-Z0-9]{2,}\b", s))
            score += len(re.findall(r"\d+", s))
            score += min(8, len(self._tokenize_for_search(s)))
            return score

        picks = {0, len(segs) - 1}
        remaining_slots = max_segments - len(picks)
        if remaining_slots > 0:
            mids = [i for i in range(1, len(segs) - 1)]
            mids.sort(key=lambda i: (-_salience(segs[i]), i))
            for i in mids[:remaining_slots]:
                picks.add(i)
        ordered = [segs[i] for i in sorted(picks)][:max_segments]
        return [self._compact_memory_text(s, max_len=max_len) for s in ordered]

    def _classify_memory_kind(self, role: str, text: str) -> str:
        t = (text or "").lower()
        if "?" in t:
            return "open_question"
        if any(k in t for k in ("must", "should", "never", "always", "rule", "constraint")):
            return "constraint"
        if any(k in t for k in ("i prefer", "i like", "i don't like", "dont like", "call me", "don't call me")):
            return "preference"
        if role == "assistant" and any(k in t for k in ("decision", "choose", "route", "use ", "plan")):
            return "decision"
        return "fact"

    def _extract_memory_tags(self, text: str, limit: int = 12) -> List[str]:
        toks = set(self._tokenize_for_search(text))
        low = (text or "").lower()
        software_terms = {
            "software", "app", "apps", "application", "program", "platform", "service", "browser", "os",
            "firewall", "plugin", "extension", "tool", "tools", "api", "sdk", "gateway",
        }
        device_terms = {
            "phone", "mobile", "tablet", "laptop", "desktop", "console", "controller", "device", "hardware", "technology",
        }
        promise_terms = {"promise", "promised", "promises", "promising"}
        word_set = set(re.findall(r"[a-zA-Z0-9]+", low))
        priority: List[str] = []
        if word_set & software_terms:
            toks.update({"software", "apps"})
            priority.extend(["software", "apps"])
        if word_set & device_terms:
            toks.update({"phone", "device", "technology"})
            priority.extend(["phone", "device", "technology"])
        if word_set & promise_terms or "promis" in low:
            toks.add("promised")
            priority.append("promised")
        # Preserve likely entities/acronyms (e.g., product names) without hard-coded literals.
        title_entities = re.findall(r"\b[A-Z][A-Za-z0-9]+\b", text or "")
        upper_entities = re.findall(r"\b[A-Z0-9]{2,}\b", text or "")
        for tok in title_entities:
            toks.add(tok.lower())
        for tok in upper_entities:
            toks.add(tok.lower())
        if (len(title_entities) + len(upper_entities)) >= 2:
            toks.add("entity")
            priority.append("entity")

        ordered: List[str] = []
        seen: set[str] = set()
        for t in priority:
            if t and t not in seen:
                ordered.append(t)
                seen.add(t)
        # Preserve important literal entities in original order, then fill from remaining tokens.
        for t in re.findall(r"[a-zA-Z0-9]+", low):
            if len(t) <= 2:
                continue
            if t in toks and t not in seen:
                ordered.append(t)
                seen.add(t)
        for t in sorted(toks):
            if t not in seen:
                ordered.append(t)
                seen.add(t)
        return ordered[:limit]

    def _contains_likely_pii(self, text: str) -> bool:
        return contains_likely_pii(text or "")

    def _sanitize_memory_segment(self, text: str) -> str:
        token = str(self.valves.summary_pii_redaction_token or "[REDACTED]").strip() or "[REDACTED]"
        return redact_pii(text or "", token=token)

    def _build_memory_units(self, messages: List[Dict], *, n_user: int, max_segments: int = 3, include_assistant: bool = False) -> List[Dict[str, Any]]:
        ua = [m for m in messages if m.get("role") in ("user", "assistant")]
        tail = ua[-24:]  # wider capture to reduce near-term forgetting on long turns
        users_in_tail = sum(1 for m in tail if m.get("role") == "user")
        user_turn = max(1, n_user - users_in_tail + 1)
        units: List[Dict[str, Any]] = []
        for m in tail:
            role = str(m.get("role") or "user")
            if role == "assistant" and not include_assistant:
                continue
            if role == "user":
                turn_marker = user_turn
                user_turn += 1
            else:
                turn_marker = max(1, user_turn - 1)

            segments = self._split_memory_segments(
                m.get("content", ""),
                max_segments=max(1, int(max_segments)),
                max_len=220,
            )
            if not segments:
                continue
            for seg in segments:
                if not bool(self.valves.summary_store_pii):
                    if self._contains_likely_pii(seg):
                        seg = self._sanitize_memory_segment(seg)
                kind = self._classify_memory_kind(role, seg)
                tags = self._extract_memory_tags(seg, limit=12)
                unit_id = _sha256_crc_id(f"{role}|{seg.lower().strip()}")
                units.append(
                    {
                        "id": unit_id,
                        "kind": kind,
                        "text": seg,
                        "tags": tags,
                        "turn_range": [turn_marker, turn_marker],
                        "confidence": 1.0,
                        "created_at": _ts(),
                    }
                )
        return units

    def _merge_memory_units(self, existing: List[Dict[str, Any]], fresh: List[Dict[str, Any]], *, max_units: int) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for rec in existing or []:
            uid = str(rec.get("id") or "")
            if uid:
                merged[uid] = rec
        for rec in fresh or []:
            uid = str(rec.get("id") or "")
            if uid:
                merged[uid] = rec  # fresh overwrites stale duplicate ids

        def _turn_end(r: Dict[str, Any]) -> int:
            tr = r.get("turn_range") or [0, 0]
            try:
                return int(tr[-1])
            except Exception:
                return 0

        ordered = sorted(merged.values(), key=lambda r: (-_turn_end(r), str(r.get("id") or "")))
        keep = max(8, int(max_units or 96))
        return ordered[:keep]

    def _write_memory_units_jsonl(self, path: str, units: List[Dict[str, Any]]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for u in units:
                f.write(_dumps(u) + "\n")
        os.replace(tmp, path)

    def _load_memory_units_jsonl(self, path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict) and rec.get("id") and rec.get("text"):
                        out.append(rec)
        except Exception:
            return []
        return out

    def _refresh_memory_index(self, session_id: str, units: List[Dict[str, Any]]) -> None:
        sid = self._normalize_session_id(session_id)
        by_id: Dict[str, Dict[str, Any]] = {}
        idx: Dict[str, set[str]] = {}
        for u in units:
            uid = str(u.get("id") or "")
            if not uid:
                continue
            by_id[uid] = u
            for tag in (u.get("tags") or []):
                t = str(tag).strip().lower()
                if not t:
                    continue
                idx.setdefault(t, set()).add(uid)
        self._mu_cache[sid] = by_id
        self._mu_index[sid] = idx

    def _maybe_update_session_memory(self, messages: List[Dict], *, session_id: str, every_n: int, debug_enabled: bool) -> None:
        if every_n <= 0:
            return
        n_user = self._count_user_messages(messages)
        if n_user <= 0 or (n_user % every_n) != 0:
            return
        sid = self._normalize_session_id(session_id)
        if self._mu_last_update_turn.get(sid) == n_user:
            return
        units = self._build_memory_units(
            messages,
            n_user=n_user,
            max_segments=int(self.valves.summary_segments_per_message or 3),
            include_assistant=bool(self.valves.summary_include_assistant),
        )
        path = self._session_memory_file(sid)
        existing = self._load_memory_units_jsonl(path)
        merged = self._merge_memory_units(
            existing,
            units,
            max_units=int(self.valves.summary_memory_max_units or 96),
        )
        self._write_memory_units_jsonl(path, merged)
        self._refresh_memory_index(sid, merged)
        self._mu_last_update_turn[sid] = n_user
        if debug_enabled:
            print(f"[Vodka] session memory updated sid={sid} units={len(merged)}")

    def _score_memory_unit(self, query_tokens: set[str], unit: Dict[str, Any]) -> float:
        tags = {str(t).lower() for t in (unit.get("tags") or [])}
        overlap = query_tokens & tags
        q_low = {str(t).lower() for t in query_tokens}
        software_query = bool(q_low & {"software", "apps", "app", "application", "program", "tool"})
        device_query = bool(q_low & {"device", "phone", "tablet", "console", "technology", "hardware"})
        promise_query = bool(q_low & {"promise", "promised", "promises"})
        concept_bonus = 0.0
        if software_query and bool(tags & {"software", "apps"}):
            concept_bonus += 0.75
        if device_query and bool(tags & {"device", "phone", "technology"}):
            concept_bonus += 0.55
        if (software_query or device_query) and "entity" in tags:
            concept_bonus += 0.35
        if promise_query and "promised" in tags:
            concept_bonus += 0.45
        if not overlap and concept_bonus <= 0:
            return 0.0
        text = str(unit.get("text") or "").lower()
        phrase_bonus = 0.25 if " ".join(sorted(query_tokens))[:24] and any(t in text for t in (overlap or q_low)) else 0.0
        return float(len(overlap)) + phrase_bonus + concept_bonus

    def _is_recall_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        if re.search(r"\b(did i|have i|what did i)\b.{0,48}\b(mention|say|said)\b", q):
            return True
        hints = (
            "did i say",
            "have i said",
            "what did i say",
            "what did i mention",
            "what have i",
            "remind me",
            "anything about",
            "what software",
            "what technologies",
            "what did i promise",
            "what have i promised",
        )
        return any(h in q for h in hints)

    def _extract_mention_targets(self, query: str) -> List[str]:
        q = (query or "").lower().strip()
        if not q:
            return []
        if not re.search(r"\b(did i|have i|what did i)\b", q):
            return []
        if "mention" not in q and "say" not in q:
            return []
        scope = q.split("about", 1)[1] if "about" in q else q
        ignore = {
            "did",
            "have",
            "what",
            "mention",
            "mentioned",
            "say",
            "said",
            "anything",
            "point",
            "also",
            "again",
            "there",
            "your",
            "any",
            "etc",
        }
        toks = [t for t in sorted(self._tokenize_for_search(scope)) if t not in ignore]
        return toks[:8]

    def _deterministic_recall_hint(self, query: str, units: List[Dict[str, Any]]) -> str:
        targets = self._extract_mention_targets(query)
        if not targets:
            return ""
        hits: Dict[str, List[str]] = {}
        for t in targets:
            t_low = t.lower()
            for u in units:
                txt = str(u.get("text") or "").strip()
                if txt and t_low in txt.lower():
                    hits.setdefault(t, []).append(txt)

        matched = sorted(hits.keys())
        unmatched = [t for t in targets if t not in hits]
        verdict = "YES" if matched else "NO"

        lines: List[str] = [f"Deterministic recall verdict: {verdict}."]
        lines.append("Targets checked: " + ", ".join(targets))
        if matched:
            lines.append("Matched targets: " + ", ".join(matched))
        if unmatched:
            lines.append("Unmatched targets: " + ", ".join(unmatched))

        if matched:
            seen: set[str] = set()
            ev: List[str] = []
            for t in matched:
                for raw in hits.get(t, []):
                    key = raw.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    ev.append(self._compact_memory_text(raw, max_len=180))
                    if len(ev) >= 3:
                        break
                if len(ev) >= 3:
                    break
            if ev:
                lines.append("Evidence bullets:")
                for e in ev:
                    lines.append(f"- {e}")
        return "\n".join(lines)

    def _deterministic_recall_answer(self, query: str, units: List[Dict[str, Any]]) -> str:
        targets = self._extract_mention_targets(query)
        if not targets:
            return ""
        q_norm = " ".join((query or "").lower().split())
        software_aliases = {"software", "app", "apps", "application", "program", "tool", "tools"}
        hits: Dict[str, List[str]] = {}
        for t in targets:
            t_low = t.lower()
            for u in units:
                txt = str(u.get("text") or "").strip()
                if not txt:
                    continue
                txt_low = txt.lower()
                if " ".join(txt_low.split()) == q_norm:
                    continue
                tags = {str(x).lower() for x in (u.get("tags") or [])}
                software_match = t_low in software_aliases and (
                    {"software", "apps"} & tags
                    or any(k in txt_low for k in ("app", "kaios", "jellyfin", "smart-tube", "firewall", "pi-hole"))
                )
                if t_low in txt_low or software_match:
                    hits.setdefault(t, []).append(txt)

        matched = sorted(hits.keys())
        unmatched = [t for t in targets if t not in hits]
        verdict = "Yes." if matched else "No."

        lines: List[str] = [verdict]
        if matched:
            lines.append("Matched terms: " + ", ".join(matched))
            seen: set[str] = set()
            evidence: List[str] = []
            for t in matched:
                for raw in hits.get(t, []):
                    key = raw.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.append(self._compact_memory_text(raw, max_len=180))
                    if len(evidence) >= 3:
                        break
                if len(evidence) >= 3:
                    break
            if evidence:
                lines.append("Evidence:")
                for e in evidence:
                    lines.append(f"- {e}")
        if unmatched:
            lines.append("Not found in recall evidence: " + ", ".join(unmatched))
        lines.append("Confidence: high | Source: Contextual")
        return "\n".join(lines)

    def _deterministic_recall_summary(self, query: str, units: List[Dict[str, Any]]) -> str:
        q_tokens = self._tokenize_for_search(query)
        want_promise = bool(q_tokens & {"promise", "promised", "promises"})
        want_software = bool(q_tokens & {"software", "apps", "app", "application", "program", "tool"})

        picked: List[str] = []
        for u in units:
            tags = {str(t).lower() for t in (u.get("tags") or [])}
            txt = str(u.get("text") or "").strip()
            if not txt:
                continue
            if want_promise and not ("promised" in tags or "promis" in txt.lower()):
                continue
            if want_software and not ({"software", "apps"} & tags or any(k in txt.lower() for k in ("app", "kaios", "jellyfin", "smart-tube", "firewall"))):
                continue
            picked.append(self._compact_memory_text(txt, max_len=180))

        if not picked:
            for u in units[:4]:
                txt = str(u.get("text") or "").strip()
                if txt:
                    picked.append(self._compact_memory_text(txt, max_len=180))

        if not picked:
            return "No recall evidence found for this query.\n\nConfidence: medium | Source: Contextual"

        lines: List[str] = ["From session recall evidence:"]
        for p in picked[:4]:
            lines.append(f"- {p}")
        lines.append("Confidence: high | Source: Contextual")
        return "\n".join(lines)

    def _score_memory_unit_loose(self, query_tokens: set[str], query: str, unit: Dict[str, Any]) -> float:
        text = str(unit.get("text") or "").lower()
        tags = {str(t).lower() for t in (unit.get("tags") or [])}
        q_low = {str(t).lower() for t in query_tokens}
        overlap = q_low & tags
        # Loose fallback for recall queries: allow text contains on tokens even if not indexed in tags.
        contains = {t for t in q_low if len(t) >= 4 and t in text}
        if not overlap and not contains:
            return 0.0
        kind = str(unit.get("kind") or "").lower()
        kind_bonus = 0.2 if kind in {"fact", "preference", "constraint"} else 0.0
        exact_phrase_bonus = 0.3 if query and query.lower() in text else 0.0
        return float(len(overlap) + len(contains)) + kind_bonus + exact_phrase_bonus

    def _render_memory_injection(self, units: List[Dict[str, Any]], max_chars: int, *, recall_mode: bool = False) -> str:
        lines: List[str] = []
        used = 0
        for u in units:
            line = f"- ({u.get('kind','fact')}) {u.get('text','')}"
            if max_chars > 0 and used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return ""
        if recall_mode:
            return (
                "Recall evidence from this session:\n"
                "Use only the bullets below for recall claims; do not invent unseen items.\n"
                + "\n".join(lines)
            )
        return "Relevant session memory:\n" + "\n".join(lines)

    def _upsert_memory_message(self, messages: List[Dict], text: str, debug_enabled: bool) -> List[Dict]:
        # Remove existing memory/legacy summary messages first
        cleaned = []
        for m in messages:
            c = m.get("content", "")
            if m.get("role") in ("system", "assistant") and isinstance(c, str) and (
                c.startswith(SUMMARY_PREFIX) or c.startswith(MEMORY_PREFIX)
            ):
                continue
            cleaned.append(m)
        if not text:
            return cleaned
        summary_msg = {"role": "system", "content": MEMORY_PREFIX + text}
        if debug_enabled:
            print("[Vodka] session memory injected")
        return [summary_msg] + cleaned

    def _maybe_inject_session_memory(self, messages: List[Dict], *, session_id: str, max_units: int, max_chars: int, debug_enabled: bool) -> List[Dict]:
        idx = self._find_last_user_index(messages)
        if idx is None:
            return messages
        query = messages[idx].get("content", "")
        if not isinstance(query, str):
            query = str(query or "")
        q = query.strip()
        if not q or q.startswith((">>", "##", "!!", "??")):
            return self._upsert_memory_message(messages, "", debug_enabled=False)
        q_tokens = self._tokenize_for_search(q)
        if not q_tokens:
            return self._upsert_memory_message(messages, "", debug_enabled=False)
        recall_mode = self._is_recall_query(q)

        sid = self._normalize_session_id(session_id)
        by_id = self._mu_cache.get(sid)
        idx_map = self._mu_index.get(sid)
        if not by_id or not idx_map:
            path = self._session_memory_file(sid)
            units = self._load_memory_units_jsonl(path)
            self._refresh_memory_index(sid, units)
            by_id = self._mu_cache.get(sid, {})
            idx_map = self._mu_index.get(sid, {})

        candidate_ids: set[str] = set()
        for t in q_tokens:
            candidate_ids.update(idx_map.get(t, set()))
        if not candidate_ids and not recall_mode:
            return self._upsert_memory_message(messages, "", debug_enabled=False)

        # Recall queries score against the full retained memory set to avoid omission.
        if recall_mode and by_id:
            candidate_ids = set(by_id.keys())

        scored: List[Tuple[float, str]] = []
        for uid in candidate_ids:
            rec = by_id.get(uid)
            if not rec:
                continue
            if recall_mode:
                s = self._score_memory_unit_loose(q_tokens, q, rec)
            else:
                s = self._score_memory_unit(q_tokens, rec)
            if s > 0:
                scored.append((s, uid))
        if not scored:
            return self._upsert_memory_message(messages, "", debug_enabled=False)
        scored.sort(key=lambda x: (-x[0], x[1]))
        effective_max_units = max(1, int(max_units or 3))
        if recall_mode:
            effective_max_units = max(effective_max_units, 6)
        picked = [by_id[uid] for _, uid in scored[:effective_max_units] if uid in by_id]
        txt = self._render_memory_injection(
            picked,
            max_chars=max(0, int(max_chars or 0)),
            recall_mode=recall_mode,
        )
        if recall_mode and txt:
            cleaned = self._upsert_memory_message(messages, "", debug_enabled=False)
            # Deterministic recall fast-path for recall-mode queries.
            deterministic_answer = self._deterministic_recall_answer(q, picked)
            if not deterministic_answer:
                deterministic_answer = self._deterministic_recall_summary(q, picked)
            if deterministic_answer:
                user_idx = self._find_last_user_index(cleaned)
                if user_idx is not None:
                    del cleaned[user_idx]
                cleaned.append({"role": "assistant", "content": "[recall_det]\n" + deterministic_answer})
                return cleaned

            # Otherwise keep recall-mode rewrite so the model must use evidence.
            user_idx = self._find_last_user_index(cleaned)
            if user_idx is not None:
                deterministic_hint = self._deterministic_recall_hint(q, picked)
                cleaned[user_idx]["content"] = (
                    "Answer the user's recall question using ONLY the evidence below.\n\n"
                    f"User question: {q}\n\n"
                    + (deterministic_hint + "\n\n" if deterministic_hint else "")
                    +
                    f"{txt}\n\n"
                    "Instructions:\n"
                    "- Use only the evidence above for recall claims.\n"
                    "- If a deterministic verdict is provided above, do not contradict it.\n"
                    "- If asked whether something was mentioned, answer YES or NO directly first.\n"
                    "- If YES, cite the matching evidence bullet(s) briefly.\n"
                    "- If NO, say it was not found in recall evidence.\n"
                    "- Do not invent items not present in evidence.\n"
                )
            return cleaned
        return self._upsert_memory_message(messages, txt, debug_enabled=debug_enabled)

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
            system_msgs = [m for m in out if m.get("role") == "system"]
            ua_msgs = [m for m in out if m.get("role") in ("user", "assistant")]

            selected_rev: List[Dict] = []
            used = 0
            latest_user = next((m for m in reversed(ua_msgs) if m.get("role") == "user"), None)
            latest_assistant = next((m for m in reversed(ua_msgs) if m.get("role") == "assistant"), None)

            # Keep newest turns first under the char budget.
            for m in reversed(ua_msgs):
                c = m.get("content", "")
                if not isinstance(c, str):
                    c = str(c)
                # Always keep the newest message, even if large.
                if not selected_rev:
                    selected_rev.append(m)
                    used += len(c)
                    continue
                if used + len(c) <= max_chars:
                    selected_rev.append(m)
                    used += len(c)

            selected = list(reversed(selected_rev))

            # Guarantee latest user/assistant survive (important for command/recall turns).
            if latest_user and latest_user not in selected:
                selected.append(latest_user)
            if latest_assistant and latest_assistant not in selected:
                selected.append(latest_assistant)

            out = system_msgs + selected

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

        # 2) session memory update + relevance-gated injection (optional)
        if bool(self.valves.enable_summary):
            sid_raw = str(body.get("session_id") or "").strip()
            if bool(self.valves.summary_require_session_id) and not sid_raw:
                sid_raw = ""
            sid = sid_raw or "global"
            if bool(self.valves.summary_require_session_id) and not sid_raw:
                messages = self._upsert_memory_message(messages, "", debug_enabled=False)
            else:
                every_n = int(self.valves.summary_every_n_user_msgs or 0)
                self._maybe_update_session_memory(
                    messages=messages,
                    session_id=sid,
                    every_n=every_n,
                    debug_enabled=debug_enabled,
                )
                messages = self._maybe_inject_session_memory(
                    messages=messages,
                    session_id=sid,
                    max_units=int(self.valves.summary_inject_max_units or 3),
                    max_chars=int(self.valves.summary_inject_max_chars or 500),
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

