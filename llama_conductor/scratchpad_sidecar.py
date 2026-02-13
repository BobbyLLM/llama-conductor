"""Session scratchpad sidecar (ephemeral grounded memory).

Purpose:
- Capture selected command outputs (help/status/peek/etc.) as session-local records.
- Provide deterministic retrieval snippets for subsequent reasoning turns.
- Keep data local, inspectable, and bounded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import calendar
from typing import Any, Dict, List, Optional, Sequence, Set

from .config import cfg_get
from .helpers import is_command, strip_cmd_prefix


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_EXHAUSTIVE_QUERY_RE = re.compile(
    r"\b("
    r"all\s+facts|"
    r"list\s+all|"
    r"show\s+all|"
    r"display\s+all|"
    r"everything\s+stored|"
    r"all\s+information|"
    r"full\s+context|"
    r"entire\s+context"
    r")\b",
    flags=re.I,
)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_session_id(session_id: str) -> str:
    s = (session_id or "").strip() or "session"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120] if len(s) > 120 else s


def _storage_root() -> str:
    # Default to vodka storage root if configured; otherwise current working dir.
    base = str(cfg_get("scratchpad.storage_dir", "") or "").strip()
    if not base:
        base = str(cfg_get("vodka.storage_dir", "") or "").strip()
    if not base:
        base = "."
    return os.path.abspath(base)


def get_session_scratchpad_path(session_id: str) -> str:
    root = _storage_root()
    sid = _safe_session_id(session_id)
    folder = os.path.join(root, "total_recall", "session_kb")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{sid}.jsonl")


def _tokenize(text: str) -> Set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


_GENERIC_EXHAUSTIVE_TOKENS = {
    "all", "facts", "fact", "list", "show", "display", "everything", "stored",
    "information", "context", "from", "about", "pertaining", "scratchpad",
}
_SEMANTIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
}


def _semantic_query_tokens(query: str) -> Set[str]:
    toks = _tokenize(query or "")
    return {
        t for t in toks
        if len(t) > 2 and t not in _GENERIC_EXHAUSTIVE_TOKENS and t not in _SEMANTIC_STOPWORDS
    }


def _wants_exhaustive_query(query: str) -> bool:
    return bool(_EXHAUSTIVE_QUERY_RE.search((query or "").strip()))


def wants_exhaustive_query(query: str) -> bool:
    """Public wrapper used by router/handlers."""
    return _wants_exhaustive_query(query)


def _load_records(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _parse_utc_ts(s: str) -> Optional[float]:
    t = (s or "").strip()
    if not t:
        return None
    try:
        # Stored format: 2026-02-11T12:34:56Z
        return float(calendar.timegm(time.strptime(t, "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return None


def _prune_expired_records(
    records: Sequence[Dict[str, Any]],
    *,
    max_age_minutes: int,
) -> List[Dict[str, Any]]:
    if max_age_minutes <= 0:
        return list(records)

    now = time.time()
    max_age_s = float(max_age_minutes) * 60.0
    out: List[Dict[str, Any]] = []
    for rec in records:
        ts = _parse_utc_ts(str(rec.get("ts_utc", "")))
        if ts is None:
            # Drop malformed timestamps (fail-closed for ephemerality).
            continue
        age_s = now - ts
        if age_s <= max_age_s:
            out.append(rec)
    return out


def _save_records(path: str, records: Sequence[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _load_pruned_records(path: str) -> List[Dict[str, Any]]:
    """Load records, apply hard age expiry, and persist cleanup."""
    records = _load_records(path)
    max_age = int(cfg_get("scratchpad.max_age_minutes", 60))
    pruned = _prune_expired_records(records, max_age_minutes=max_age)

    # Persist pruning result.
    if not pruned:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass
        return []

    if len(pruned) != len(records):
        try:
            _save_records(path, pruned)
        except Exception:
            pass
    return pruned


def _extract_query_snippet(text: str, q_tokens: Set[str], max_len: int = 420) -> str:
    txt = (text or "").strip()
    if not txt:
        return ""
    if not q_tokens:
        return txt[:max_len].strip()

    low = txt.lower()
    best_pos = -1
    for tok in q_tokens:
        p = low.find(tok)
        if p >= 0 and (best_pos == -1 or p < best_pos):
            best_pos = p
    if best_pos < 0:
        return txt[:max_len].strip()

    # Prefer sentence-aligned snippet around first hit.
    sentence_start = txt.rfind(".", 0, best_pos)
    sentence_start_q = txt.rfind("?", 0, best_pos)
    sentence_start_b = txt.rfind("!", 0, best_pos)
    sentence_start = max(sentence_start, sentence_start_q, sentence_start_b)
    if sentence_start < 0:
        sentence_start = 0
    else:
        sentence_start += 1

    sentence_end = len(txt)
    for ch in (".", "?", "!"):
        p = txt.find(ch, best_pos)
        if p >= 0:
            sentence_end = min(sentence_end, p + 1)

    snippet = txt[sentence_start:sentence_end].strip()
    if not snippet:
        start = max(0, best_pos - (max_len // 3))
        end = min(len(txt), start + max_len)
        snippet = txt[start:end].strip()
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 1].rstrip() + "..."
    return snippet


def _build_preview(text: str, max_len: int = 220) -> str:
    """Build a compact 1-2 sentence preview for list output."""
    t = " ".join((text or "").split()).strip()
    if not t:
        return ""

    # Prefer first 1-2 sentences.
    parts = re.split(r"(?<=[.!?])\s+", t)
    if len(parts) >= 2:
        preview = f"{parts[0]} {parts[1]}".strip()
    else:
        preview = parts[0].strip()

    if len(preview) > max_len:
        preview = preview[: max_len - 1].rstrip() + "..."
    return preview


def _is_low_signal_record(source_command: str, text: str) -> bool:
    """Ignore session/meta command captures when building retrieval context."""
    src = (source_command or "").strip().lower()
    if src in {
        ">>status",
        ">>list",
        ">>kbs",
        ">>list_kb",
        ">>scratchpad status",
        ">>scratchpad list",
    }:
        return True

    t = (text or "").strip().lower()
    if t.startswith("[status]") and "session_id=" in t:
        return True
    if t.startswith("[scratchpad] last ") and "capture(s):" in t:
        return True
    return False


def capture_scratchpad_output(
    *,
    session_id: str,
    source_command: str,
    text: str,
    max_entries: int = 12,
    max_capture_chars: int = 12000,
) -> Optional[Dict[str, Any]]:
    txt = (text or "").strip()
    if not txt:
        return None

    path = get_session_scratchpad_path(session_id)
    clipped = txt[: max(1, int(max_capture_chars))]
    sha = hashlib.sha256(clipped.encode("utf-8", errors="ignore")).hexdigest()
    rec: Dict[str, Any] = {
        "ts_utc": _now_utc(),
        "source_command": (source_command or "").strip() or "unknown",
        "sha256": sha,
        "text": clipped,
        "preview": _build_preview(clipped),
    }

    records = _load_pruned_records(path)
    records.append(rec)
    records = records[-max(1, int(max_entries)) :]
    _save_records(path, records)
    return rec


def clear_scratchpad(session_id: str) -> bool:
    path = get_session_scratchpad_path(session_id)
    if os.path.isfile(path):
        try:
            os.remove(path)
            return True
        except Exception:
            return False
    return True


def list_scratchpad_records(session_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    path = get_session_scratchpad_path(session_id)
    records = _load_pruned_records(path)
    if limit > 0:
        return records[-limit:]
    return records


def delete_scratchpad_by_index(session_id: str, index_1based: int) -> bool:
    """Delete one record by 1-based index in current list ordering."""
    path = get_session_scratchpad_path(session_id)
    records = _load_pruned_records(path)
    if index_1based < 1 or index_1based > len(records):
        return False
    del records[index_1based - 1]
    if not records:
        try:
            if os.path.isfile(path):
                os.remove(path)
            return True
        except Exception:
            return False
    try:
        _save_records(path, records)
        return True
    except Exception:
        return False


def delete_scratchpad_by_query(session_id: str, query: str) -> int:
    """Delete records by token-aware query match (prevents substring over-deletes)."""
    q = (query or "").strip()
    q_tokens = _tokenize(q)
    if not q_tokens:
        return 0
    path = get_session_scratchpad_path(session_id)
    records = _load_pruned_records(path)
    keep: List[Dict[str, Any]] = []
    removed = 0
    q_token_set = set(q_tokens)
    for rec in records:
        t = str(rec.get("text", "") or "").lower()
        p = str(rec.get("preview", "") or "").lower()
        rec_tokens = _tokenize(f"{t} {p}")
        if q_token_set.issubset(rec_tokens):
            removed += 1
            continue
        keep.append(rec)

    if removed <= 0:
        return 0
    if not keep:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass
        return removed
    try:
        _save_records(path, keep)
    except Exception:
        return 0
    return removed


def build_scratchpad_dump_text(*, session_id: str, query: str = "") -> str:
    """Deterministic full-text dump for exhaustive intents / explicit show command."""
    records = list_scratchpad_records(session_id, limit=0)
    if not records:
        return "[scratchpad] empty"
    q = (query or "").strip()
    q_tokens = _tokenize(q)
    semantic_q_tokens = _semantic_query_tokens(q)
    exhaustive = _wants_exhaustive_query(q)
    rows: List[tuple[int, Dict[str, Any]]] = []
    for i, rec in enumerate(records, 1):
        txt = str(rec.get("text", "") or "")
        src = str(rec.get("source_command", "") or "")
        if not txt:
            continue
        if _is_low_signal_record(src, txt):
            continue
        # Exhaustive means "no truncation", not "ignore target topic".
        # If the query includes meaningful tokens (e.g. "jesus", "master"),
        # still require overlap so unrelated captures are excluded.
        if semantic_q_tokens:
            if not (_tokenize(txt) & semantic_q_tokens):
                continue
        elif q_tokens and not exhaustive:
            if not (_tokenize(txt) & q_tokens):
                continue
        rows.append((i, rec))
    if not rows:
        return "[scratchpad] no matching records"
    lines: List[str] = [f"[scratchpad dump] matches={len(rows)}"]
    for idx, rec in rows:
        src = str(rec.get("source_command", "unknown"))
        txt = str(rec.get("text", "") or "").strip()
        lines.append(f"{idx}. {src} chars={len(txt)}")
        lines.append(txt)
        lines.append("")
    return "\n".join(lines).rstrip()


def build_scratchpad_facts_block(
    *,
    session_id: str,
    query: str,
    top_k: int = 3,
    max_chars: int = 1200,
) -> str:
    path = get_session_scratchpad_path(session_id)
    records = _load_pruned_records(path)
    if not records:
        return ""

    q = (query or "").strip()
    q_tokens = _tokenize(q)
    semantic_q_tokens = _semantic_query_tokens(q)
    exhaustive = _wants_exhaustive_query(q)
    scored: List[tuple[float, bool, Dict[str, Any]]] = []

    for idx, rec in enumerate(records):
        txt = str(rec.get("text", "") or "")
        src = str(rec.get("source_command", "") or "")
        if not txt:
            continue
        if _is_low_signal_record(src, txt):
            continue

        # Base recency score (newer records stronger)
        recency = float(idx + 1) / float(max(1, len(records)))

        if q_tokens:
            t_tokens = _tokenize(txt)
            match_tokens = semantic_q_tokens if semantic_q_tokens else q_tokens
            overlap = len(match_tokens & t_tokens)
            is_match = overlap > 0
            if overlap <= 0:
                # Keep weak recency-only fallback but at low weight
                score = 0.05 + (0.20 * recency)
            else:
                score = (overlap / float(max(1, len(match_tokens)))) + (0.25 * recency)
        else:
            score = 0.20 + (0.30 * recency)
            is_match = False

        scored.append((score, is_match, rec))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    if exhaustive:
        # Explicit "all facts/everything" intent: bypass top_k/max_chars clipping.
        # Keep deterministic order by score (desc), then natural record order.
        chosen = [rec for _, _, rec in scored]
    else:
        chosen = [rec for _, _, rec in scored[: max(1, int(top_k))]]

    # Single-record safety path:
    # If there is only one usable scratchpad record, reason over its full content
    # instead of a short snippet so follow-up synthesis is not starved.
    usable_records = [
        rec for rec in records
        if str(rec.get("text", "") or "").strip()
        and not _is_low_signal_record(
            str(rec.get("source_command", "") or ""),
            str(rec.get("text", "") or ""),
        )
    ]
    matched_records = [rec for _, is_match, rec in scored if is_match]
    # Primary rule: if the query narrows to exactly one matching record,
    # reason over that full record even if other unrelated records exist.
    query_single_match_mode = (not exhaustive) and bool(q_tokens) and (len(matched_records) == 1)
    if query_single_match_mode:
        chosen = [matched_records[0]]

    single_full_mode = (
        (not exhaustive)
        and (
            query_single_match_mode
            or ((len(usable_records) == 1) and (len(chosen) == 1))
        )
    )
    single_record_cap = max(1, int(cfg_get("scratchpad.max_capture_chars", 12000)))
    effective_max_chars = max_chars
    if single_full_mode:
        effective_max_chars = max(int(max_chars), int(single_record_cap))

    pieces: List[str] = []
    used = 0
    for rec in chosen:
        source = str(rec.get("source_command", "unknown"))
        ts = str(rec.get("ts_utc", ""))
        sha = str(rec.get("sha256", ""))[:12]
        if exhaustive or single_full_mode:
            snippet = str(rec.get("text", "") or "").strip()
        else:
            snippet = _extract_query_snippet(str(rec.get("text", "")), q_tokens, max_len=420)
        if not snippet:
            continue

        line = f"- [scratchpad cmd={source} ts={ts} sha={sha}] {snippet}"
        if (not exhaustive) and effective_max_chars > 0 and used + len(line) + 2 > effective_max_chars:
            remaining = effective_max_chars - used - 2
            if remaining > 12:
                clipped = line[: remaining - 3].rstrip() + "..."
                pieces.append(clipped)
                used += len(clipped) + 2
            break
        pieces.append(line)
        used += len(line) + 2

    return "\n\n".join(pieces) if pieces else ""


def maybe_capture_command_output(
    *,
    session_id: str,
    state: Any,
    cmd_text: str,
    reply_text: str,
) -> None:
    attached = getattr(state, "attached_kbs", set()) or set()
    if "scratchpad" not in attached:
        return

    raw = (cmd_text or "").lstrip()
    if not is_command(raw):
        return

    # Parse top-level command token
    body = strip_cmd_prefix(raw).strip()
    if not body:
        return
    first = body.split()[0].lower()

    # Auto-capture only selected introspection/debug style commands.
    auto_capture = {
        "help",
        "peek",
        "find",
        "wiki",
        "exchange",
        "weather",
    }
    if first not in auto_capture:
        return

    max_entries = int(cfg_get("scratchpad.max_entries", 12))
    max_capture_chars = int(cfg_get("scratchpad.max_capture_chars", 12000))
    capture_scratchpad_output(
        session_id=session_id,
        source_command=f">>{first}",
        text=reply_text or "",
        max_entries=max_entries,
        max_capture_chars=max_capture_chars,
    )
