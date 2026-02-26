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
from difflib import SequenceMatcher
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


def purge_session_memory_jsonl(storage_dir: str = "") -> int:
    """
    Delete all session-memory JSONL files.
    Intended for router startup so each reboot starts clean.
    Returns number of deleted files.
    """
    base = storage_dir or os.getenv("DATA_DIR") or os.getcwd()
    base = os.path.abspath(base)
    folder = os.path.join(base, "total_recall", "session_memory")
    if not os.path.isdir(folder):
        return 0
    deleted = 0
    for name in os.listdir(folder):
        if not name.lower().endswith(".jsonl"):
            continue
        p = os.path.join(folder, name)
        try:
            os.remove(p)
            deleted += 1
        except Exception:
            continue
    return deleted

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
        self._mu_last_inject_turn: Dict[str, int] = {}
        self._mu_last_inject_count: Dict[str, int] = {}
        self._mu_last_query: Dict[str, str] = {}
        self._mu_last_candidate_count: Dict[str, int] = {}
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

    def _make_matrix_unit(
        self,
        *,
        role: str,
        turn_marker: int,
        matrix_type: str,
        text: str,
        tags: List[str],
        subject: str = "",
        relation: str = "",
        obj: str = "",
    ) -> Dict[str, Any]:
        packed = f"{role}|{matrix_type}|{subject}|{relation}|{obj}|{text}".lower().strip()
        unit_id = _sha256_crc_id(packed)
        return {
            "id": unit_id,
            "role": role,
            "kind": self._classify_memory_kind(role, text),
            "matrix_type": matrix_type,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "text": text,
            "tags": tags,
            "turn_range": [turn_marker, turn_marker],
            "confidence": 1.0,
            "created_at": _ts(),
            "schema_version": 2,
        }

    def _build_matrix_units_for_segment(self, *, role: str, seg: str, turn_marker: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        tags = self._extract_memory_tags(seg, limit=16)
        s = " ".join((seg or "").split()).strip()
        low = s.lower()

        # substitution facts: "<A> over <B>"
        for lhs, rhs in self._extract_substitution_pairs(s):
            txt = f"{lhs} over {rhs}"
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="substitution",
                    text=txt,
                    tags=tags + ["substitution", "technology"],
                    subject=lhs,
                    relation="over",
                    obj=rhs,
                )
            )

        # promise facts: "<X> ... promised"
        if "promis" in low:
            subject = "user"
            promised_obj = ""
            m = re.search(r"(.{3,140}?)\s+is what i(?:'|’)?ve promised", s, flags=re.IGNORECASE)
            if m:
                promised_obj = self._compact_memory_text(m.group(1), max_len=120)
            else:
                m2 = re.search(
                    r"\bi(?:'|’)?\s*(?:have\s+)?promised\s+to\s+(?:get|buy|give|provide)?\s*([^.!?]{3,140})",
                    s,
                    flags=re.IGNORECASE,
                )
                if m2:
                    promised_obj = self._compact_memory_text(m2.group(1), max_len=120)
                else:
                    # fallback: take a compact slice around "promis"
                    i = low.find("promis")
                    start = max(0, i - 80)
                    promised_obj = self._compact_memory_text(s[start : start + 140], max_len=120)
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="promise",
                    text=s,
                    tags=tags + ["promised"],
                    subject=subject,
                    relation="promised",
                    obj=promised_obj,
                )
            )

        # software/tool mentions
        software_terms = self._extract_soft_tech_items(s)
        for term in software_terms:
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="software",
                    text=s,
                    tags=tags + ["software", "apps", term.lower()],
                    subject="user",
                    relation="mentioned",
                    obj=term,
                )
            )

        # generic preference signals
        pref_rx = re.compile(r"\b(i\s+(?:prefer|like|love|don't like|dont like)\s+[^.?!]{2,120})", re.IGNORECASE)
        for m in pref_rx.finditer(s):
            ptxt = self._compact_memory_text(m.group(1), max_len=120)
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="preference",
                    text=s,
                    tags=tags + ["preference"],
                    subject="user",
                    relation="prefers",
                    obj=ptxt,
                )
            )

        # generic constraint signals
        if any(k in low for k in ("must", "should", "never", "always", "cannot", "can't")):
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="constraint",
                    text=s,
                    tags=tags + ["constraint"],
                    subject="user",
                    relation="constraint",
                    obj=self._compact_memory_text(s, max_len=120),
                )
            )

        # generic plan/intention signals
        if any(k in low for k in ("i will", "i'll", "i am going to", "i'm going to", "plan to")):
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="plan",
                    text=s,
                    tags=tags + ["plan"],
                    subject="user",
                    relation="plans",
                    obj=self._compact_memory_text(s, max_len=120),
                )
            )

        # explicit question capture
        if "?" in s:
            out.append(
                self._make_matrix_unit(
                    role=role,
                    turn_marker=turn_marker,
                    matrix_type="question",
                    text=s,
                    tags=tags + ["question"],
                    subject="user",
                    relation="asked",
                    obj=self._compact_memory_text(s, max_len=120),
                )
            )

        # factual fallback unit (always keep one canonical text unit)
        out.append(
            self._make_matrix_unit(
                role=role,
                turn_marker=turn_marker,
                matrix_type="fact",
                text=s,
                tags=tags,
                subject="user" if role == "user" else role,
                relation="said",
                obj="",
            )
        )
        return out

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
                units.extend(self._build_matrix_units_for_segment(role=role, seg=seg, turn_marker=turn_marker))
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
                    role = str(rec.get("role") or "").strip().lower()
                    if (
                        isinstance(rec, dict)
                        and rec.get("id")
                        and rec.get("text")
                        and role == "user"
                        and not self._is_noise_memory_text(str(rec.get("text") or ""))
                    ):
                        out.append(rec)
        except Exception:
            return []
        return out

    def _max_turn_in_units(self, units: List[Dict[str, Any]]) -> int:
        mx = 0
        for rec in units or []:
            tr = rec.get("turn_range") or [0, 0]
            try:
                mx = max(mx, int(tr[-1]))
            except Exception:
                continue
        return mx

    def _reset_session_memory_store(self, sid: str, path: str) -> None:
        try:
            os.remove(path)
        except Exception:
            pass
        self._mu_cache.pop(sid, None)
        self._mu_index.pop(sid, None)
        self._mu_last_update_turn.pop(sid, None)
        self._mu_last_inject_turn.pop(sid, None)
        self._mu_last_inject_count.pop(sid, None)
        self._mu_last_query.pop(sid, None)
        self._mu_last_candidate_count.pop(sid, None)

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
            for extra in (
                str(u.get("matrix_type") or "").strip().lower(),
                str(u.get("subject") or "").strip().lower(),
                str(u.get("relation") or "").strip().lower(),
                str(u.get("object") or "").strip().lower(),
            ):
                if extra:
                    idx.setdefault(extra, set()).add(uid)
            for tok in self._tokenize_for_search(str(u.get("object") or "")):
                idx.setdefault(tok, set()).add(uid)
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
        # Session IDs can be reused by client/IP fallback. If a "fresh" convo
        # starts on a reused id, stale units pollute recall quality.
        if existing and n_user <= 2:
            max_turn_seen = self._max_turn_in_units(existing)
            if max_turn_seen > (n_user + 1):
                existing = []
                self._reset_session_memory_store(sid, path)
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
        # Guardrail: do not hijack long, multi-instruction prompts into recall-mode.
        # Typical recall turns are short and explicit ("did I mention X?", "what did I say about Y?").
        if len(q) > 280:
            return False
        # Guardrail: explicit analysis/editing asks should remain normal model tasks.
        non_recall_intents = (
            "sentiment analysis",
            "factually incorrect",
            "correct any typos",
            "correct typos",
            "typo",
            "proofread",
            "grammar",
        )
        if any(h in q for h in non_recall_intents):
            return False
        if re.search(r"\b(did i|have i|what did i)\b.{0,48}\b(mention|say|said)\b", q):
            return True
        hints = (
            "did i say",
            "have i said",
            "what did i say",
            "what did i mention",
            "what have i",
            "nothing else",
            "anything else",
            "what else",
            "remind me",
            "anything about",
            "what software",
            "what technologies",
            "what did i promise",
            "what have i promised",
        )
        return any(h in q for h in hints)

    def _is_meta_recall_line(self, text: str) -> bool:
        t = " ".join((text or "").strip().lower().split())
        if not t:
            return True
        # User follow-up prompts are not evidence lines.
        if "?" in t:
            return True
        meta_prefixes = (
            "remind me",
            "nothing else",
            "i meant,",
            "what technologies have i",
            "what software did i",
            "did i ",
            "have i ",
            "what did i ",
            "and what have i ",
            "ok but did i ",
        )
        return any(t.startswith(p) for p in meta_prefixes)

    def _is_noise_memory_text(self, text: str) -> bool:
        t = " ".join((text or "").strip().lower().split())
        if not t:
            return True
        if t in {"hi", "hello", "hi there", "hey"}:
            return True
        if "[status]" in t or t.startswith("status"):
            return True
        if "confidence:" in t or "source:" in t or "profile:" in t:
            return True
        # Generic assistant meta/evaluation lines are low-value/noisy for recall.
        noisy_prefixes = (
            "your message is",
            "the technical details",
            "no major factual errors",
            "typos/spelling issues",
            "no typos detected",
        )
        if any(t.startswith(p) for p in noisy_prefixes):
            return True
        return False

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

    def _extract_substitution_pairs(self, text: str) -> List[Tuple[str, str]]:
        s = " ".join((text or "").split())
        if not s:
            return []
        out: List[Tuple[str, str]] = []
        for m in re.finditer(r"([A-Za-z0-9][A-Za-z0-9\-\s]{1,40})\s+over\s+([A-Za-z0-9][A-Za-z0-9\-\s/]{1,40})", s, flags=re.IGNORECASE):
            lhs = " ".join(m.group(1).split()).strip(" ,.;:")
            rhs = " ".join(m.group(2).split()).strip(" ,.;:")
            if not lhs or not rhs:
                continue
            out.append((lhs, rhs))
        return out[:8]

    def _extract_soft_tech_items(self, text: str) -> List[str]:
        """Best-effort extraction of software/tool names from evidence text."""
        s = " ".join((text or "").split())
        if not s:
            return []
        out: List[str] = []
        seen: set[str] = set()

        # Prefer hyphenated tool names first (e.g., smart-tube, pi-hole).
        deny_tech_hyphen = {"ds-lite", "ps-5", "x-box"}
        for m in re.finditer(r"\b([A-Za-z0-9]{2,}(?:-[A-Za-z0-9]{2,})+)\b", s):
            term = m.group(1).strip(" ,.;:()[]{}\"'")
            k = term.lower()
            if k in deny_tech_hyphen:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(term)

        # Then canonical single/multi-token product-ish names.
        token_pat = re.compile(r"\b([A-Za-z][A-Za-z0-9]{2,})\b")
        stop = {
            "confidence", "source", "profile", "neutral", "direct", "medium", "low", "high",
            "only", "kids", "content", "channels", "comment", "section", "blocked", "timers",
            "curated", "technical", "details", "correctly", "connected", "goals", "apps", "app",
            "nokia", "barbie", "smart", "wii", "xbox", "switch", "switches", "olpc", "chromebook",
            "ps5", "flipfone", "flipphone", "phone", "phones",
            "fair", "social", "wiimote", "enough", "some",
        }
        hyphen_parts = set()
        for x in out:
            if "-" in x:
                for p in x.lower().split("-"):
                    hyphen_parts.add(p)
        for m in token_pat.finditer(s):
            term = m.group(1)
            k = term.lower()
            if k in stop or len(k) < 4:
                continue
            if k in hyphen_parts:
                continue
            if k in seen:
                continue
            if k in {"kaios", "jellyfin", "smarttube", "firefox"}:
                seen.add(k)
                out.append(term)
                continue
            # Phrase-style tool naming: "<Token> kids" etc.
            if k == "kids":
                continue

        # Context-aware software capture:
        # - "<Name> app/service/platform/tool/browser/os"
        # - "using/on/with <Name> for ..."
        kw_pat = re.compile(
            r"\b([A-Za-z][A-Za-z0-9\-\+]{2,})\s+(?:app|apps|service|platform|tool|browser|os)\b",
            flags=re.IGNORECASE,
        )
        for m in kw_pat.finditer(s):
            term = m.group(1).strip(" ,.;:()[]{}\"'")
            k = term.lower()
            if k in stop or k in seen or len(k) < 3:
                continue
            if k in {"fair", "social", "wiimote", "only", "enough", "some"}:
                continue
            seen.add(k)
            out.append(term)

        using_pat = re.compile(
            r"\b(?:using|use|used|on|with)\s+([A-Za-z][A-Za-z0-9\-\+]{2,})\b",
            flags=re.IGNORECASE,
        )
        for m in using_pat.finditer(s):
            term = m.group(1).strip(" ,.;:()[]{}\"'")
            k = term.lower()
            if k in stop or k in seen or len(k) < 4:
                continue
            if k in {"that", "this", "only", "kids", "channels", "enough", "some"}:
                continue
            # Keep conservative: only capture when nearby software-ish hints exist.
            span_start, span_end = m.span()
            window = s[max(0, span_start - 36): min(len(s), span_end + 36)].lower()
            if not any(h in window for h in ("app", "software", "service", "platform", "browser", "os", "tool")):
                continue
            seen.add(k)
            out.append(term)

        phrase_m = re.search(r"\b([A-Z][A-Za-z0-9]{1,20}\s+kids)\b", s)
        if phrase_m:
            p = phrase_m.group(1).strip()
            pk = p.lower()
            first = pk.split(" ", 1)[0]
            if first not in {"only", "all", "just"} and pk not in seen:
                seen.add(pk)
                out.append(p)
        return out[:12]

    def _normalize_term(self, term: str) -> str:
        t = (term or "").strip().strip(" ,.;:()[]{}\"'")
        low = t.lower()
        if low in {"pones", "phone", "phones"}:
            return "phones"
        return t

    def _render_bullets(self, header: str, items: List[str]) -> str:
        if not items:
            return ""
        lines = [header]
        for it in items:
            lines.append(f"- {it}")
        return "\n".join(lines)

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for it in items:
            k = " ".join((it or "").lower().split())
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    def _normalize_recall_item(self, text: str) -> str:
        s = " ".join((text or "").split()).strip(" ,.;:")
        if not s:
            return ""
        s = re.sub(
            r"^i(?:'|’)?\s*(?:have\s+)?promised\s+to\s+(?:get|buy|give|provide)\s+",
            "",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"^(?:you|i)\s+(?:also\s+)?mentioned\s+", "", s, flags=re.IGNORECASE)
        return s.strip(" ,.;:")

    def _looks_like_prior_session_reference(self, query: str) -> bool:
        q = " ".join((query or "").lower().split())
        if not q:
            return False
        hints = (
            "again",
            "as before",
            "remind me",
            "did i",
            "have i",
            "what did i",
            "what have i",
            "nothing else",
            "anything else",
        )
        return any(h in q for h in hints)

    def _recall_fail_loud_response(self, query: str, *, reason: str = "insufficient") -> str:
        if reason == "partial_token":
            line = (
                "Sorry, I can't verify that reliably from this session. "
                "I found partial token overlap only, which can be a token trap, so I won't guess."
            )
        elif reason == "possible_flush":
            line = "Cannot verify, as prior session appears to have been flushed? Sorry!"
        else:
            line = "Sorry, I can't verify that from this session, so I won't guess!"
        return f"{line}\n\nConfidence: medium | Source: Contextual"

    def _has_partial_token_trap(self, mention_targets: List[str], evidence_units: List[Dict[str, Any]]) -> bool:
        targets = [
            " ".join((t or "").lower().split()).strip()
            for t in (mention_targets or [])
            if t and len(str(t).strip()) >= 4
        ]
        if not targets:
            return False
        words: set[str] = set()
        for u in (evidence_units or [])[:12]:
            txt = str(u.get("text") or "").lower()
            words.update(re.findall(r"[a-z0-9][a-z0-9\-']+", txt))
        if not words:
            return False
        for target in targets:
            if target in words:
                continue
            for w in words:
                if len(w) < 4:
                    continue
                if target in w or w in target:
                    if w != target:
                        return True
                ratio = SequenceMatcher(None, target, w).ratio()
                if 0.72 <= ratio < 0.99:
                    return True
        return False

    def _infer_recall_intents(self, query: str, mention_targets: List[str]) -> set[str]:
        q = " ".join((query or "").lower().split())
        toks = self._tokenize_for_search(q)
        intents: set[str] = set()
        if toks & {
            "technology",
            "technologies",
            "retro",
            "replaced",
            "substituted",
            "substitute",
            "substitution",
            "substitutions",
            "swap",
            "swapped",
            "replaced",
            "replace",
            "simpler",
        } or " over " in q:
            intents.add("list_substitutions")
        if toks & {"promise", "promised", "promises"}:
            intents.add("list_promises")
        if toks & {"software", "apps", "app", "application", "program", "tool", "tools"}:
            intents.add("list_tools")
        if mention_targets:
            intents.add("did_mention")
        if not intents and (toks & {"remember", "recall", "remind", "again", "what"}):
            intents.add("general_recall")
        return intents

    def _build_human_recall_response(self, query: str, units: List[Dict[str, Any]]) -> str:
        """Global intent-aware rendering for recall answers."""
        q = " ".join((query or "").lower().split())
        q_tokens = self._tokenize_for_search(q)
        mention_targets = [self._normalize_term(t) for t in self._extract_mention_targets(q)]
        intents = self._infer_recall_intents(q, mention_targets)
        evidence_units = [
            u for u in (units or [])
            if not self._is_meta_recall_line(str(u.get("text") or ""))
        ]
        if not evidence_units:
            reason = "possible_flush" if self._looks_like_prior_session_reference(q) else "insufficient"
            if intents:
                return self._recall_fail_loud_response(q, reason=reason)
            return ""
        sub_units = [u for u in evidence_units if str(u.get("matrix_type") or "") == "substitution"]
        promise_units = [u for u in evidence_units if str(u.get("matrix_type") or "") == "promise"]
        software_units = [u for u in evidence_units if str(u.get("matrix_type") or "") == "software"]
        fact_units = [u for u in evidence_units if str(u.get("matrix_type") or "") == "fact"]

        # Follow-up shorthand after a prior recall answer.
        if re.search(r"\b(nothing else|anything else)\b", q):
            extra_items: List[str] = []
            for u in software_units:
                obj = self._normalize_recall_item(str(u.get("object") or ""))
                if obj:
                    extra_items.append(obj)
            for u in evidence_units:
                txt = str(u.get("text") or "")
                extra_items.extend(self._normalize_recall_item(x) for x in self._extract_soft_tech_items(txt))
            extra_items = self._dedupe_keep_order([x for x in extra_items if x])[:5]
            if sub_units:
                return "No, nothing else.\n\nConfidence: high | Source: Contextual"

        # 1) substitution style queries
        if "list_substitutions" in intents:
            pairs: List[str] = []
            standalone: List[str] = []
            for u in sub_units:
                lhs = str(u.get("subject") or "").strip()
                rhs = str(u.get("object") or "").strip()
                if lhs and rhs:
                    pairs.append(f"{lhs} (over {rhs})")
            for u in evidence_units:
                txt = str(u.get("text") or "")
                for lhs, rhs in self._extract_substitution_pairs(txt):
                    pairs.append(f"{lhs} (over {rhs})")
                if re.search(r"\bold\s+flip\w*\b", txt, flags=re.IGNORECASE):
                    m = re.search(r"\bold\s+([A-Za-z0-9\-]+)\b", txt, flags=re.IGNORECASE)
                    if m:
                        standalone.append(f"Old {m.group(1)}")
            items = self._dedupe_keep_order(pairs + standalone)[:8]
            if items:
                header = "As before, you substituted the following:" if "i meant" in q else "You substituted the following:"
                body = self._render_bullets(header, items)
                return f"{body}\n\nConfidence: high | Source: Contextual"

        # 2) promised + software summary (higher priority than generic mention)
        if ("list_promises" in intents) or ("list_tools" in intents):
            promised_items: List[str] = []
            software_items: List[str] = []
            for u in promise_units:
                obj = str(u.get("object") or "").strip()
                if obj:
                    promised_items.append(self._normalize_recall_item(obj))
                else:
                    promised_items.append(
                        self._normalize_recall_item(
                            self._compact_memory_text(str(u.get("text") or ""), max_len=110)
                        )
                    )
            for u in software_units:
                obj = str(u.get("object") or "").strip()
                if obj:
                    software_items.append(self._normalize_recall_item(obj))
            for u in evidence_units:
                txt = str(u.get("text") or "").strip()
                if not txt:
                    continue
                low = txt.lower()
                if "promis" in low:
                    m = re.search(r"(.{0,140}?)\s+is what i(?:'|’)?ve promised", txt, flags=re.IGNORECASE)
                    if m:
                        promised_items.append(
                            self._normalize_recall_item(self._compact_memory_text(m.group(1), max_len=110))
                        )
                    else:
                        promised_items.append(
                            self._normalize_recall_item(self._compact_memory_text(txt, max_len=110))
                        )
                tags = {str(t).lower() for t in (u.get("tags") or [])}
                if {"software", "apps"} & tags or any(k in low for k in ("app", "kaios", "jellyfin", "smart-tube", "pi-hole", "firewall")):
                    software_items.extend(self._normalize_recall_item(x) for x in self._extract_soft_tech_items(txt))

            promised_items = self._dedupe_keep_order(promised_items)[:3]
            software_items = self._dedupe_keep_order(software_items)[:6]
            chunks: List[str] = []
            if promised_items:
                chunks.append(self._render_bullets("You promised the following:", promised_items))
            if software_items:
                chunks.append(self._render_bullets("You also mentioned:", software_items))
            if chunks:
                return "\n\n".join(chunks) + "\n\nConfidence: high | Source: Contextual"

        # 3) mention-target queries like "did I say anything about X, Y"
        if "did_mention" in intents:
            matched_terms: List[str] = []
            per_target_bullet: Dict[str, str] = {}
            kinship_terms = {"daughter", "son", "wife", "husband", "kids", "child", "children"}
            for t in mention_targets:
                t_low = t.lower()
                if t_low in kinship_terms:
                    continue
                # Prefer typed matrix objects first.
                for u in (software_units + promise_units + sub_units):
                    obj = str(u.get("object") or "").strip()
                    if not obj:
                        continue
                    obj_low = obj.lower()
                    if t_low in obj_low or (t_low == "phones" and ("pones" in obj_low or "phone" in obj_low)):
                        matched_terms.append(t)
                        per_target_bullet[t_low] = self._compact_memory_text(obj, max_len=100)
                        break
                else:
                    # Fallback to matching line snippets.
                    for u in evidence_units:
                        txt = str(u.get("text") or "").strip()
                        if not txt:
                            continue
                        low = txt.lower()
                        if t_low in low or (t_low == "phones" and ("pones" in low or "phone" in low)):
                            matched_terms.append(t)
                            per_target_bullet[t_low] = self._compact_memory_text(txt, max_len=140)
                            break
            matched_terms = self._dedupe_keep_order(matched_terms)
            if matched_terms:
                lines = ["Yes, you mentioned:"]
                bullets: List[str] = []
                for mt in matched_terms:
                    b = per_target_bullet.get(mt.lower(), "")
                    if b:
                        bullets.append(b)
                bullets = self._dedupe_keep_order(bullets)[:4]
                for b in bullets:
                    lines.append(f"- {b}")
                lines.append("Confidence: high | Source: Contextual")
                return "\n".join(lines)
            if mention_targets:
                if self._has_partial_token_trap(mention_targets, evidence_units):
                    return self._recall_fail_loud_response(q, reason="partial_token")
                joined = ", ".join(mention_targets[:4])
                return f"No, you did not mention: {joined}.\nConfidence: high | Source: Contextual"
            return "No, you did not mention that.\nConfidence: high | Source: Contextual"

        # 4) broad fallback for "what do you remember/recall"
        if "general_recall" in intents and fact_units:
            bullets: List[str] = []
            for u in fact_units[:4]:
                txt = self._compact_memory_text(str(u.get("text") or ""), max_len=140)
                if txt:
                    bullets.append(txt)
            bullets = self._dedupe_keep_order(bullets)
            if bullets:
                body = self._render_bullets("From session recall evidence:", bullets)
                return f"{body}\n\nConfidence: high | Source: Contextual"

        return ""

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
        human = self._build_human_recall_response(query, units)
        if human:
            return human
        targets = self._extract_mention_targets(query)
        if not targets:
            return ""
        q_norm = " ".join((query or "").lower().split())
        software_aliases = {"software", "app", "apps", "application", "program", "tool", "tools"}
        evidence_units = [
            u for u in (units or [])
            if not self._is_meta_recall_line(str(u.get("text") or ""))
        ]
        if not evidence_units:
            evidence_units = list(units or [])
        hits: Dict[str, List[str]] = {}
        for t in targets:
            t_low = t.lower()
            for u in evidence_units:
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
                typo_phone_match = (t_low == "phones") and ("pones" in txt_low or "phone" in txt_low)
                if t_low in txt_low or software_match or typo_phone_match:
                    hits.setdefault(t, []).append(txt)

        matched = sorted(hits.keys())
        kinship_terms = {"daughter", "son", "wife", "husband", "kids", "child", "children"}
        unmatched = [t for t in targets if t not in hits and t.lower() not in kinship_terms]
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
        human = self._build_human_recall_response(query, units)
        if human:
            return human
        q_tokens = self._tokenize_for_search(query)
        want_promise = bool(q_tokens & {"promise", "promised", "promises"})
        want_software = bool(q_tokens & {"software", "apps", "app", "application", "program", "tool"})
        want_tech = bool(
            q_tokens & {
                "technology",
                "technologies",
                "retro",
                "replaced",
                "substituted",
                "simpler",
                "device",
                "devices",
                "wii",
                "xbox",
                "ps5",
                "switch",
                "switches",
                "olpc",
            }
        )
        evidence_units = [
            u for u in (units or [])
            if not self._is_meta_recall_line(str(u.get("text") or ""))
        ]
        if not evidence_units:
            evidence_units = list(units or [])

        picked: List[str] = []
        for u in evidence_units:
            tags = {str(t).lower() for t in (u.get("tags") or [])}
            txt = str(u.get("text") or "").strip()
            if not txt:
                continue
            if want_promise and not ("promised" in tags or "promis" in txt.lower()):
                continue
            if want_software and not ({"software", "apps"} & tags or any(k in txt.lower() for k in ("app", "kaios", "jellyfin", "smart-tube", "firewall"))):
                continue
            if want_tech and not (
                {"technology", "device", "phone"} & tags
                or any(
                    k in txt.lower()
                    for k in ("wii", "xbox", "ps5", "switch", "olpc", "retro")
                )
            ):
                continue
            picked.append(self._compact_memory_text(txt, max_len=180))

        if not picked:
            for u in evidence_units[:4]:
                txt = str(u.get("text") or "").strip()
                if txt:
                    picked.append(self._compact_memory_text(txt, max_len=180))

        if not picked:
            reason = "possible_flush" if self._looks_like_prior_session_reference(query) else "insufficient"
            return self._recall_fail_loud_response(query, reason=reason)

        if want_tech:
            pairs: List[str] = []
            seen_pairs: set[str] = set()
            for u in evidence_units:
                txt = str(u.get("text") or "")
                for lhs, rhs in self._extract_substitution_pairs(txt):
                    key = f"{lhs.lower()}::{rhs.lower()}"
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    pairs.append(f"- {lhs} over {rhs}")
                    if len(pairs) >= 6:
                        break
                if len(pairs) >= 6:
                    break
            if pairs:
                lines = ["From session recall evidence (substitutions):"] + pairs
                lines.append("Confidence: high | Source: Contextual")
                return "\n".join(lines)

        lines: List[str] = ["From session recall evidence:"]
        for p in picked[:4]:
            lines.append(f"- {p}")
        lines.append("Confidence: high | Source: Contextual")
        return "\n".join(lines)

    def _pick_recall_units(
        self,
        *,
        query: str,
        by_id: Dict[str, Dict[str, Any]],
        max_units: int,
    ) -> List[Dict[str, Any]]:
        """
        Intent-aware unit selection for recall-mode.
        Prefer semantically relevant matrix types first, then backfill by recency.
        """
        all_units = list((by_id or {}).values())
        if not all_units:
            return []

        mention_targets = [self._normalize_term(t) for t in self._extract_mention_targets(query)]
        intents = self._infer_recall_intents(query, mention_targets)
        q_tokens = self._tokenize_for_search(query)

        def _turn_end(u: Dict[str, Any]) -> int:
            tr = u.get("turn_range") or [0, 0]
            try:
                return int(tr[-1])
            except Exception:
                return 0

        def _u_type(u: Dict[str, Any]) -> str:
            return str(u.get("matrix_type") or "").strip().lower()

        def _u_text(u: Dict[str, Any]) -> str:
            return str(u.get("text") or "").strip()

        def _u_tags(u: Dict[str, Any]) -> set[str]:
            return {str(t).lower() for t in (u.get("tags") or [])}

        candidates = [
            u for u in all_units
            if _u_text(u) and not self._is_meta_recall_line(_u_text(u)) and not self._is_noise_memory_text(_u_text(u))
        ]
        if not candidates:
            candidates = [u for u in all_units if _u_text(u)]

        # newest first baseline
        candidates.sort(key=lambda u: (-_turn_end(u), str(u.get("id") or "")))

        selected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _add(unit: Dict[str, Any]) -> None:
            uid = str(unit.get("id") or "")
            if not uid or uid in seen_ids:
                return
            seen_ids.add(uid)
            selected.append(unit)

        def _filter(pred) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for u in candidates:
                if pred(u):
                    out.append(u)
            return out

        if "list_substitutions" in intents:
            for u in _filter(lambda u: _u_type(u) == "substitution"):
                _add(u)

        if "list_promises" in intents:
            for u in _filter(lambda u: _u_type(u) == "promise" or "promised" in _u_tags(u)):
                _add(u)

        if "list_tools" in intents:
            def _tool_pred(u: Dict[str, Any]) -> bool:
                t = _u_type(u)
                tags = _u_tags(u)
                txt_low = _u_text(u).lower()
                return (
                    t == "software"
                    or bool({"software", "apps"} & tags)
                    or any(k in txt_low for k in ("app", "software", "service", "platform", "kaios", "jellyfin", "smart-tube", "smarttube", "pi-hole", "firefox"))
                )
            for u in _filter(_tool_pred):
                _add(u)

        if "did_mention" in intents and mention_targets:
            tset = {t.lower() for t in mention_targets}
            for u in candidates:
                txt_low = _u_text(u).lower()
                obj_low = str(u.get("object") or "").lower()
                if any(t in txt_low or t in obj_low for t in tset):
                    _add(u)

        # Backfill with high loose-score units to preserve generality.
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for u in candidates:
            s = self._score_memory_unit_loose(q_tokens, query, u)
            if s > 0:
                scored.append((s, u))
        scored.sort(key=lambda x: (-x[0], -_turn_end(x[1]), str(x[1].get("id") or "")))
        for _, u in scored:
            _add(u)
            if len(selected) >= max(1, int(max_units)):
                break

        if not selected:
            selected = candidates[: max(1, int(max_units))]
        return selected[: max(1, int(max_units))]

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
        question_penalty = 0.8 if kind == "open_question" or self._is_meta_recall_line(text) else 0.0
        return float(len(overlap) + len(contains)) + kind_bonus + exact_phrase_bonus - question_penalty

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
        turn_now = self._count_user_messages(messages)
        self._mu_last_query[sid] = self._compact_memory_text(q, max_len=180)
        self._mu_last_inject_turn[sid] = max(0, int(turn_now))
        self._mu_last_inject_count[sid] = 0
        self._mu_last_candidate_count[sid] = 0
        by_id = self._mu_cache.get(sid)
        idx_map = self._mu_index.get(sid)
        if not by_id or not idx_map:
            path = self._session_memory_file(sid)
            units = self._load_memory_units_jsonl(path)
            # Same stale-session guard as update path, but evaluated on every
            # injection turn so reused session IDs are cleaned immediately.
            n_user = self._count_user_messages(messages)
            if units and n_user <= 2:
                max_turn_seen = self._max_turn_in_units(units)
                if max_turn_seen > (n_user + 1):
                    self._reset_session_memory_store(sid, path)
                    units = []
            self._refresh_memory_index(sid, units)
            by_id = self._mu_cache.get(sid, {})
            idx_map = self._mu_index.get(sid, {})

        candidate_ids: set[str] = set()
        for t in q_tokens:
            candidate_ids.update(idx_map.get(t, set()))
        self._mu_last_candidate_count[sid] = len(candidate_ids)
        if not candidate_ids and not recall_mode:
            return self._upsert_memory_message(messages, "", debug_enabled=False)

        # Recall queries score against the full retained memory set to avoid omission.
        if recall_mode and by_id:
            candidate_ids = set(by_id.keys())
            self._mu_last_candidate_count[sid] = len(candidate_ids)

        effective_max_units = max(1, int(max_units or 3))
        if recall_mode:
            effective_max_units = max(effective_max_units, 12)
            picked = self._pick_recall_units(
                query=q,
                by_id=by_id,
                max_units=effective_max_units,
            )
        else:
            scored: List[Tuple[float, str]] = []
            for uid in candidate_ids:
                rec = by_id.get(uid)
                if not rec:
                    continue
                s = self._score_memory_unit(q_tokens, rec)
                if s > 0:
                    scored.append((s, uid))
            if not scored:
                return self._upsert_memory_message(messages, "", debug_enabled=False)
            scored.sort(key=lambda x: (-x[0], x[1]))
            picked = [by_id[uid] for _, uid in scored[:effective_max_units] if uid in by_id]
        if not picked:
            return self._upsert_memory_message(messages, "", debug_enabled=False)
        self._mu_last_inject_count[sid] = len(picked)
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
                evidence = []
                for u in picked[:8]:
                    evidence.append(
                        {
                            "id": str(u.get("id") or ""),
                            "matrix_type": str(u.get("matrix_type") or ""),
                            "object": str(u.get("object") or ""),
                            "text": self._compact_memory_text(str(u.get("text") or ""), max_len=220),
                        }
                    )
                payload = _dumps(
                    {
                        "query": q,
                        "draft": deterministic_answer,
                        "evidence": evidence,
                    }
                )
                user_idx = self._find_last_user_index(cleaned)
                if user_idx is not None:
                    del cleaned[user_idx]
                cleaned.append({"role": "assistant", "content": "[recall_det]\n" + payload})
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

    def get_session_memory_status(self, session_id: str, *, user_turn_hint: int = 0) -> Dict[str, Any]:
        """Return lightweight observability for session-memory behavior."""
        sid = self._normalize_session_id(session_id)
        path = self._session_memory_file(sid)
        by_id = self._mu_cache.get(sid)
        if by_id is None:
            units = self._load_memory_units_jsonl(path)
            self._refresh_memory_index(sid, units)
            by_id = self._mu_cache.get(sid, {})

        unit_count = int(len(by_id or {}))
        out = {
            "session_id": sid,
            "enabled_summary": bool(self.valves.enable_summary),
            "require_session_id": bool(self.valves.summary_require_session_id),
            "summary_every_n_user_msgs": int(self.valves.summary_every_n_user_msgs or 0),
            "summary_inject_max_units": int(self.valves.summary_inject_max_units or 0),
            "summary_inject_max_chars": int(self.valves.summary_inject_max_chars or 0),
            "memory_file": path,
            "memory_file_exists": bool(os.path.exists(path)),
            "unit_count": unit_count,
            "last_update_turn": int(self._mu_last_update_turn.get(sid, 0) or 0),
            "last_inject_turn": int(self._mu_last_inject_turn.get(sid, 0) or 0),
            "last_inject_units": int(self._mu_last_inject_count.get(sid, 0) or 0),
            "last_candidate_count": int(self._mu_last_candidate_count.get(sid, 0) or 0),
            "last_query": str(self._mu_last_query.get(sid, "") or ""),
            "user_turn_hint": int(max(0, int(user_turn_hint or 0))),
        }
        return out

    def _run_session_memory_pipeline(self, messages: List[Dict], *, body: Dict[str, Any], debug_enabled: bool) -> List[Dict]:
        """
        Canonical memory flow:
          capture -> units -> retrieve -> render/inject

        Compatibility note:
        - existing helper methods (`_maybe_update_session_memory`, `_maybe_inject_session_memory`)
          remain the implementation primitives for now.
        - this wrapper is the single entry point used by `inlet()`.
        """
        sid_raw = str(body.get("session_id") or "").strip()
        if bool(self.valves.summary_require_session_id) and not sid_raw:
            # Explicitly disable memory injection when caller doesn't provide a session id.
            return self._upsert_memory_message(messages, "", debug_enabled=False)

        sid = sid_raw or "global"
        every_n = int(self.valves.summary_every_n_user_msgs or 0)
        self._maybe_update_session_memory(
            messages=messages,
            session_id=sid,
            every_n=every_n,
            debug_enabled=debug_enabled,
        )
        return self._maybe_inject_session_memory(
            messages=messages,
            session_id=sid,
            max_units=int(self.valves.summary_inject_max_units or 3),
            max_chars=int(self.valves.summary_inject_max_chars or 500),
            debug_enabled=debug_enabled,
        )

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

        # 2) canonical memory pipeline (capture -> retrieve -> render/inject)
        if bool(self.valves.enable_summary):
            messages = self._run_session_memory_pipeline(
                messages=messages,
                body=body,
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

