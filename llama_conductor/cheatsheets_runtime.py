"""Deterministic Cheatsheets + Track B wiki gating.

Track A:
- Match user turn against static JSONL entries (term + tags)
- Inject FACTS block when lane policy allows

Track B:
- Optional wiki sidecar fallback when no static match and definition-miss gate passes
- Disabled by default at caller
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


_QUESTION_FRAMING_RE = re.compile(
    r"\b(what(?:'s|\s+is)|explain|tell me about|how does\b.+\bwork)\b|\?",
    re.IGNORECASE,
)
_DISTRESS_SUBS = {"distress_hint", "vulnerable_under_humour"}
_PERSONAL_OR_DISTRESS_RE = re.compile(
    r"\b(personal|distress)\b",
    re.IGNORECASE,
)
_DISTRESS_TEXT_RE = re.compile(
    r"\b("
    r"fraud|overwhelmed|anxious|panic|stressed|hurt|pain|broken|"
    r"sad|lonely|depressed|not okay|not fine|feels bad|feel bad|too old"
    r")\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTED_TERM_RE = re.compile(r"['\"]([^'\"\n]{2,64})['\"]")
_CAP_TERM_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_/-]{2,}(?:\s+[A-Z][A-Za-z0-9_/-]{2,}){0,3}\b"
)
_ORDINAL_CAP_TERM_RE = re.compile(
    r"\b\d{1,3}(?:st|nd|rd|th)\s+[A-Z][A-Za-z0-9_/-]{2,}(?:\s+[A-Z][A-Za-z0-9_/-]{2,}){0,4}\b"
)
_STRICT_LOOKUP_RE = re.compile(
    r"^\s*(?:who(?:'s|\s+is)|what(?:'s|\s+is)|define|explain|tell me about)\s+(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_OPEN_ENDED_LOOKUP_RE = re.compile(
    r"\b("
    r"list|characters?|examples?|what do you know|who do you know|all of|all the|"
    r"anything else|top \d+|best of|which ones"
    r")\b",
    re.IGNORECASE,
)
_INDEX_QUERY_RE = re.compile(
    r"^\s*(?:what\s+topics\s+do\s+you\s+know(?:\s+about)?|what\s+can\s+you\s+talk\s+about)\s*\??\s*$",
    re.IGNORECASE,
)
_LOCAL_KNOWLEDGE_INTENT_RE = re.compile(
    r"^\s*(?:"
    r"what\s+do\s+you\s+know(?:\s+about)?|"
    r"what\s+local\s+knowledge\s+do\s+you\s+have|"
    r"tell\s+me\s+about\s+your\s+local\s+knowledge|"
    r"what(?:'s|\s+is)\s+in\s+your\s+knowledge\s+base|"
    r"list\s+index"
    r")\s*\??\s*$",
    re.IGNORECASE,
)
_LOCAL_KNOWLEDGE_TOPICS_VARIANT_RE = re.compile(
    r"^\s*what\s+topics\s+can\s+you\s+help\s+with\b.*\b(local|knowledge|index|knowledge\s+base)\b.*\??\s*$",
    re.IGNORECASE,
)
_LOCAL_GEO_ENV_RE = re.compile(
    r"\b(immediate\s+environment|local\s+area|where\s+you\s+operate|your\s+surroundings)\b",
    re.IGNORECASE,
)
_TOPIC_LOOKUP_RE = re.compile(
    r"^\s*(?:what\s+do\s+you\s+know\s+about|tell\s+me\s+about|explain)\s+(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_LIST_CMD_RE = re.compile(r"^\s*list\s+(.+?)\s*[?.!]*\s*$", re.IGNORECASE)
_SUMMARY_CMD_RE = re.compile(r"^\s*summary\s+(.+?)\s*[?.!]*\s*$", re.IGNORECASE)
_LIST_INDEX_EXACT_RE = re.compile(r"^\s*list\s+index\s*\??\s*$", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^\s*(?:the|a|an)\s+")
_UMBRELLA_TAGS = {"pop culture", "pop_culture", "gen x", "gen_x"}


_COMMON_WORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "what",
    "were",
    "where",
    "when",
    "why",
    "who",
    "how",
    "about",
    "with",
    "from",
    "into",
    "your",
    "yours",
    "mine",
    "ours",
    "their",
    "hello",
    "hey",
    "thanks",
    "please",
    "today",
    "tomorrow",
    "yesterday",
    "question",
    "answer",
    "problem",
    "issue",
    "coding",
    "python",
}


@dataclass(frozen=True)
class CheatsheetEntry:
    term: str
    category: str
    definition: str
    confidence: str
    tags: Tuple[str, ...]
    norm_term: str
    norm_category: str
    norm_tags: Tuple[str, ...]


@dataclass(frozen=True)
class CheatsheetsTurnResult:
    facts_block: str
    constraints_block: str
    deterministic_answer: str
    local_knowledge_line: str
    track: str  # "A" | "B" | ""
    footer_source: str  # "Cheatsheets" | "Wiki" | "Mixed" | ""
    footer_confidence: str  # "low|medium|high|top|unverified|"
    kb_lookup_candidate: bool
    matched_terms: Tuple[str, ...]
    wiki_term: str


_FILE_META: Dict[Path, Tuple[int, str]] = {}
_PARSED_BY_SHA: Dict[str, Tuple[CheatsheetEntry, ...]] = {}
_WARNINGS_BY_SHA: Dict[str, Tuple[str, ...]] = {}
_LAST_PARSE_WARNINGS: Tuple[str, ...] = ()
_INDEX_LAST_KEY = ""


def _norm_text(s: str) -> str:
    t = str(s or "").strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall(_norm_text(s))


def _warn(msg: str) -> None:
    print(f"[cheatsheets] warning: {msg}", flush=True)


def get_cheatsheets_parse_warnings() -> Tuple[str, ...]:
    """Return parse/runtime warnings from the latest cheatsheets load.

    Non-empty values indicate malformed JSONL rows/files were skipped.
    """
    return tuple(_LAST_PARSE_WARNINGS)


def _build_index_payload(entries: Sequence[CheatsheetEntry], files: Sequence[Path]) -> Dict[str, Any]:
    by_file: Dict[str, Dict[str, Any]] = {}
    for p in files:
        stem = str(p.stem or "").strip().lower()
        if not stem:
            continue
        by_file.setdefault(stem, {"terms": set()})

    by_topic: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        entry_tags = {str(t or "").strip().lower() for t in e.tags if str(t or "").strip()}
        for stem in list(by_file.keys()):
            if stem == "pop_culture":
                # Most character facts live here via domain tags.
                if "pop_culture" in entry_tags:
                    by_file[stem]["terms"].add(str(e.term or "").strip())
            elif e.norm_category == _norm_text(stem):
                by_file[stem]["terms"].add(str(e.term or "").strip())
        topic = str(e.category or "general").strip().lower() or "general"
        row = by_topic.setdefault(topic, {"terms": set(), "tags": set()})
        row["terms"].add(str(e.term or "").strip())
        for t in e.tags:
            ts = str(t or "").strip()
            if ts:
                row["tags"].add(ts)
    topics: List[Dict[str, Any]] = []
    for topic in sorted(by_file.keys()):
        terms = sorted(str(x) for x in by_file[topic]["terms"] if str(x).strip())
        topics.append(
            {
                "topic": topic,
                "term_count": len(terms),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": int(len(entries)),
        "topic_count": int(len(topics)),
        "topics": topics,
    }


def _write_index_json(
    cheatsheets_dir: Path,
    entries: Sequence[CheatsheetEntry],
    files: Sequence[Path],
    key: str,
) -> None:
    global _INDEX_LAST_KEY
    if not str(key or "").strip():
        return
    if key == _INDEX_LAST_KEY:
        return
    payload = _build_index_payload(entries, files)
    out_path = cheatsheets_dir / "index.json"
    try:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _INDEX_LAST_KEY = key
    except Exception:
        _warn(f"failed to write {out_path.name}; continuing without index refresh")


def _load_index_topics(cheatsheets_dir: Path, entries: Sequence[CheatsheetEntry]) -> List[str]:
    idx = cheatsheets_dir / "index.json"
    topics: List[str] = []
    try:
        if idx.exists():
            raw = json.loads(idx.read_text(encoding="utf-8"))
            rows = list(raw.get("topics", [])) if isinstance(raw, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                topic = str(row.get("topic", "") or "").strip().lower()
                if topic:
                    topics.append(topic)
    except Exception:
        topics = []
    if topics:
        return sorted(set(topics))
    return sorted(
        {
            str(e.category or "").strip().lower()
            for e in entries
            if str(e.category or "").strip()
        }
    )


def _build_index_facts(topics: Sequence[str]) -> str:
    return "\n".join(
        [
            "[CHEATSHEETS INDEX]",
            f"Topic count: {len([t for t in topics if str(t).strip()])}",
            "Source: Cheatsheets",
            "---",
        ]
    ).strip()


def _build_index_constraints() -> str:
    return "\n".join(
        [
            "Cheatsheets index policy:",
            "- Keep answer concise and natural.",
            "- Provide general capabilities briefly in one sentence.",
            "- Do not enumerate specific domain/topic names in the general-capabilities sentence.",
            "- Do not include numeric claims about topic coverage.",
            "- Do not claim local topics that are not in CHEATSHEETS INDEX.",
            "- Do not output a dedicated local-topics line; router appends it deterministically.",
        ]
    )


def _format_topic_label(topic: str) -> str:
    t = str(topic or "").strip().replace("_", " ")
    s = re.sub(r"\s+", " ", t).strip().title()
    upper_map = {
        "Mcu": "MCU",
    }
    return upper_map.get(s, s)


def _build_local_knowledge_line(topics: Sequence[str]) -> str:
    friendly = [_format_topic_label(t) for t in topics if str(t or "").strip()]
    if not friendly:
        return ""
    return "Local knowledge includes: " + ", ".join(friendly)


def _build_local_knowledge_geo_guard_line(topics: Sequence[str]) -> str:
    line = _build_local_knowledge_line(topics)
    suffix = line.replace("Local knowledge includes:", "Local knowledge refers to:").strip()
    if not suffix:
        return "I don't have geographic or environmental knowledge."
    return f"I don't have geographic or environmental knowledge. {suffix}"


def _escape_markdown_leading_quotes(text: str) -> str:
    """Escape leading markdown quote markers so literal '>>' doesn't render as blockquote."""
    out_lines: List[str] = []
    for line in str(text or "").splitlines():
        m = re.match(r"^(\s*)(>+)(.*)$", line)
        if not m:
            out_lines.append(line)
            continue
        indent, marks, rest = m.groups()
        out_lines.append(f"{indent}{'\\>' * len(marks)}{rest}")
    return "\n".join(out_lines)


def _extract_requested_topic(user_text: str) -> str:
    m = _TOPIC_LOOKUP_RE.match(str(user_text or "").strip())
    if not m:
        return ""
    out = _norm_text(m.group(1))
    out = _LEADING_ARTICLE_RE.sub("", out).strip()
    return out


def _topic_entries(entries: Sequence[CheatsheetEntry], requested_topic_norm: str) -> List[CheatsheetEntry]:
    if not requested_topic_norm:
        return []
    category_hits: List[CheatsheetEntry] = []
    tag_hits: List[CheatsheetEntry] = []
    for e in entries:
        cat = _norm_text(e.category)
        tags = {_norm_text(t) for t in e.tags}
        if requested_topic_norm == cat:
            category_hits.append(e)
            continue
        # Query-target tags should be explicit domain tags, not umbrella metadata.
        if requested_topic_norm in tags and requested_topic_norm not in _UMBRELLA_TAGS:
            tag_hits.append(e)
    if category_hits:
        return sorted(category_hits, key=lambda e: str(e.term or "").lower())
    return sorted(tag_hits, key=lambda e: str(e.term or "").lower())


def _term_exact_entries(entries: Sequence[CheatsheetEntry], term_norm: str) -> List[CheatsheetEntry]:
    if not term_norm:
        return []
    out = [e for e in entries if _norm_text(e.term) == term_norm]
    return sorted(out, key=lambda e: str(e.term or "").lower())


def _category_entries(entries: Sequence[CheatsheetEntry], category_norm: str) -> List[CheatsheetEntry]:
    if not category_norm:
        return []
    cat_hits = [e for e in entries if _norm_text(e.category) == category_norm]
    if cat_hits:
        return sorted(cat_hits, key=lambda e: str(e.term or "").lower())
    # For command-path category listing, allow exact tag category (including pop_culture).
    tag_hits: List[CheatsheetEntry] = []
    for e in entries:
        tags = {_norm_text(t) for t in e.tags}
        if category_norm in tags:
            tag_hits.append(e)
    return sorted(tag_hits, key=lambda e: str(e.term or "").lower())


def _build_list_category_answer(category_norm: str, rows: Sequence[CheatsheetEntry]) -> str:
    label = _format_topic_label(category_norm.replace(" ", "_"))
    terms = [str(e.term or "").strip() for e in rows if str(e.term or "").strip()]
    if not terms:
        return f'No local cheatsheets category match for "{label}".'
    by_subcat: Dict[str, List[str]] = {}
    for e in rows:
        sub = _norm_text(e.category) or "general"
        by_subcat.setdefault(sub, []).append(str(e.term or "").strip())
    # If everything is the same subcategory, keep single-line output.
    if len(by_subcat) <= 1:
        return f"**[{label}]**\n\n{', '.join(sorted(set(terms)))}"
    lines: List[str] = [f"**[{label}]**"]
    for sub in sorted(by_subcat.keys()):
        sub_label = _format_topic_label(sub.replace(" ", "_"))
        sub_terms = sorted({t for t in by_subcat[sub] if t})
        lines.append(f"**[{sub_label}]**\n\n" + ", ".join(sub_terms))
    return "\n\n".join(lines).strip()


def _build_summary_answer(category_or_term_norm: str, rows: Sequence[CheatsheetEntry]) -> str:
    label = _format_topic_label(category_or_term_norm.replace(" ", "_"))
    if not rows:
        return f'No local cheatsheets summary available for "{label}".'
    blocks: List[str] = []
    for e in rows:
        t = str(e.term or "").strip()
        d = str(e.definition or "").strip()
        if t and d:
            blocks.append(f"**{t}**\n{d}")
    return "\n\n".join(blocks).strip()


def _build_topic_deterministic_answer(topic_norm: str, topic_entries: Sequence[CheatsheetEntry]) -> str:
    if not topic_entries:
        return ""
    topic_label = _format_topic_label(topic_norm.replace(" ", "_"))
    terms = [str(e.term or "").strip() for e in topic_entries if str(e.term or "").strip()]
    terms_txt = ", ".join(sorted(set(terms)))
    return f"Local cheatsheets coverage for {topic_label}: {terms_txt}."


def _read_jsonl_entries(path: Path) -> Tuple[Tuple[CheatsheetEntry, ...], Tuple[str, ...]]:
    warns: List[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        msg = f"{path.name}: read failed; skipping file"
        warns.append(msg)
        _warn(msg)
        return (), tuple(warns)
    rows: List[CheatsheetEntry] = []
    for line_no, ln in enumerate(raw_text.splitlines(), start=1):
        s = str(ln or "").strip()
        if not s:
            continue
        try:
            item = json.loads(s)
        except json.JSONDecodeError as e:
            msg = f"{path.name}:{line_no}: json decode error: {e.msg}; skipping line"
            warns.append(msg)
            _warn(msg)
            continue
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "") or "").strip()
        category = str(item.get("category", "") or "").strip().lower()
        definition = str(item.get("definition", "") or "").strip()
        confidence = str(item.get("confidence", "medium") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high", "top", "unverified"}:
            confidence = "medium"
        tags_raw = item.get("tags", [])
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                ts = str(t or "").strip()
                if ts:
                    tags.append(ts)
        if not term or not definition:
            continue
        if not category:
            if tags:
                category = _norm_text(tags[0]).replace(" ", "_")
            else:
                category = "general"
        norm_term = _norm_text(term)
        norm_category = _norm_text(category)
        norm_tags_set = {_norm_text(t) for t in tags if _norm_text(t)}
        if norm_category:
            norm_tags_set.add(norm_category)
        norm_tags = tuple(sorted(norm_tags_set))
        rows.append(
            CheatsheetEntry(
                term=term,
                category=category,
                definition=definition,
                confidence=confidence,
                tags=tuple(tags),
                norm_term=norm_term,
                norm_category=norm_category,
                norm_tags=norm_tags,
            )
        )
    return tuple(rows), tuple(warns)


def load_cheatsheets_entries(cheatsheets_dir: Path) -> Tuple[CheatsheetEntry, ...]:
    global _LAST_PARSE_WARNINGS
    files = sorted([p for p in cheatsheets_dir.glob("*.jsonl")])
    if not files:
        _FILE_META.clear()
        _LAST_PARSE_WARNINGS = ()
        return ()

    out: List[CheatsheetEntry] = []
    load_warns: List[str] = []
    alive = set(files)
    for stale in list(_FILE_META.keys()):
        if stale not in alive:
            _FILE_META.pop(stale, None)
    for p in files:
        try:
            st = p.stat()
            mtime_ns = int(st.st_mtime_ns)
        except Exception:
            continue
        cached = _FILE_META.get(p)
        if cached and int(cached[0]) == mtime_ns and cached[1] in _PARSED_BY_SHA:
            out.extend(_PARSED_BY_SHA[cached[1]])
            load_warns.extend(list(_WARNINGS_BY_SHA.get(cached[1], ())))
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            msg = f"{p.name}: read failed; skipping file"
            load_warns.append(msg)
            _warn(msg)
            continue
        sha = hashlib.sha256(raw).hexdigest()
        _FILE_META[p] = (mtime_ns, sha)
        if sha not in _PARSED_BY_SHA:
            parsed_rows, parsed_warns = _read_jsonl_entries(p)
            _PARSED_BY_SHA[sha] = parsed_rows
            _WARNINGS_BY_SHA[sha] = parsed_warns
        out.extend(_PARSED_BY_SHA[sha])
        load_warns.extend(list(_WARNINGS_BY_SHA.get(sha, ())))
    entries = tuple(out)
    _LAST_PARSE_WARNINGS = tuple(sorted(set(str(w).strip() for w in load_warns if str(w).strip())))
    index_key = "|".join(
        sorted(
            str(_FILE_META.get(p, (0, ""))[1] or "")
            for p in files
            if str(_FILE_META.get(p, (0, ""))[1] or "").strip()
        )
    )
    _write_index_json(cheatsheets_dir, entries, files, index_key)
    return entries


def _question_framing(text: str) -> bool:
    return bool(_QUESTION_FRAMING_RE.search(str(text or "")))


def _entry_match_specificity(entry: CheatsheetEntry, user_norm: str, user_tokens: set[str]) -> Tuple[bool, List[str], int]:
    term_hit = False
    term_tokens = [t for t in _TOKEN_RE.findall(entry.norm_term) if t]
    if term_tokens:
        if len(term_tokens) == 1:
            term_hit = term_tokens[0] in user_tokens
        else:
            term_hit = all(t in user_tokens for t in term_tokens) or (entry.norm_term in user_norm)
    hit_tags: List[str] = []
    for tag in entry.norm_tags:
        tag_tokens = [t for t in _TOKEN_RE.findall(tag) if t]
        if not tag_tokens:
            continue
        if len(tag_tokens) == 1:
            if tag_tokens[0] in user_tokens:
                hit_tags.append(tag)
        elif all(t in user_tokens for t in tag_tokens) or (tag in user_norm):
            hit_tags.append(tag)
    specificity = (2 if term_hit else 0) + len(set(hit_tags))
    return term_hit, hit_tags, specificity


def _select_matches(entries: Sequence[CheatsheetEntry], user_text: str) -> List[Tuple[CheatsheetEntry, int]]:
    user_norm = _norm_text(user_text)
    user_tokens = set(_tokens(user_text))
    hits: List[Tuple[CheatsheetEntry, int]] = []
    for e in entries:
        term_hit, _tag_hits, specificity = _entry_match_specificity(e, user_norm, user_tokens)
        if term_hit or specificity > 0:
            hits.append((e, specificity))
    # Deterministic tie-break:
    # 1) specificity desc, 2) term length desc, 3) term alpha asc
    hits.sort(
        key=lambda row: (
            -int(row[1]),
            -len(str(row[0].norm_term or "")),
            str(row[0].norm_term or ""),
        )
    )
    return hits


def _has_direct_term_hit(*, entry: CheatsheetEntry, user_norm: str, user_tokens: set[str]) -> bool:
    term_tokens = [t for t in _TOKEN_RE.findall(entry.norm_term) if t]
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in user_tokens
    return all(t in user_tokens for t in term_tokens) or (entry.norm_term in user_norm)


def _is_strict_lookup_query(*, user_text: str, matches: Sequence[Tuple[CheatsheetEntry, int]]) -> bool:
    q = str(user_text or "").strip()
    if not _STRICT_LOOKUP_RE.match(q):
        return False
    if _OPEN_ENDED_LOOKUP_RE.search(q):
        return False
    user_norm = _norm_text(q)
    user_tokens = set(_tokens(q))
    return any(_has_direct_term_hit(entry=e, user_norm=user_norm, user_tokens=user_tokens) for e, _ in matches)


def _direct_term_matches(*, user_text: str, matches: Sequence[Tuple[CheatsheetEntry, int]]) -> List[CheatsheetEntry]:
    user_norm = _norm_text(user_text)
    user_tokens = set(_tokens(user_text))
    out: List[CheatsheetEntry] = []
    seen = set()
    for e, _ in matches:
        if e.norm_term in seen:
            continue
        if _has_direct_term_hit(entry=e, user_norm=user_norm, user_tokens=user_tokens):
            out.append(e)
            seen.add(e.norm_term)
    return out


def _build_track_a_constraints(*, strict_lookup: bool) -> str:
    if strict_lookup:
        lines = [
            "Cheatsheets grounding policy:",
            "- Use CHEATSHEETS FACTS as the only factual source for this answer.",
            "- If requested information is not in CHEATSHEETS FACTS, say it is not available and do not guess.",
            "- Do not add extra entities, relationships, or details beyond CHEATSHEETS FACTS.",
        ]
        return "\n".join(lines)
    lines = [
        "Cheatsheets grounding policy:",
        "- Prioritize CHEATSHEETS FACTS for matched terms.",
        "- If the user asks broadly beyond the provided terms, you may synthesize cautiously.",
        "- Mark uncertainty briefly instead of presenting unsupported details as fact.",
    ]
    return "\n".join(lines)


def _build_cheatsheets_facts(matches: Sequence[Tuple[CheatsheetEntry, int]]) -> str:
    blocks: List[str] = []
    for e, _spec in matches:
        blocks.append(
            "\n".join(
                [
                    "[CHEATSHEETS FACTS]",
                    f"Term: {e.term}",
                    f"Definition: {e.definition}",
                    f"Confidence: {e.confidence}",
                    "Source: Cheatsheets",
                    "---",
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def _wiki_candidates(user_text: str, known_norm_terms: set[str]) -> List[str]:
    out: List[str] = []
    raw_user = str(user_text or "").strip()
    if raw_user and ("?" in raw_user or re.match(r"^\s*(?:who|what|when|where|why|how)\b", raw_user, flags=re.IGNORECASE)):
        out.append(raw_user)
    # Preserve numeric/ordinal specificity first (e.g., "97th Academy Awards").
    for c in _ORDINAL_CAP_TERM_RE.findall(str(user_text or "")):
        cand = str(c or "").strip()
        if cand:
            out.append(cand)
    for q in _QUOTED_TERM_RE.findall(str(user_text or "")):
        cand = str(q or "").strip()
        if cand:
            out.append(cand)
    for c in _CAP_TERM_RE.findall(str(user_text or "")):
        cand = str(c or "").strip()
        if cand:
            out.append(cand)
    uniq: List[str] = []
    seen = set()
    for raw in out:
        n = _norm_text(raw)
        if not n or n in seen:
            continue
        seen.add(n)
        if n in known_norm_terms:
            continue
        if n in _COMMON_WORDS:
            continue
        uniq.append(raw)

    def _cand_score(c: str) -> Tuple[int, int, str]:
        n = _norm_text(c)
        toks = _TOKEN_RE.findall(n)
        has_num = 1 if any(re.search(r"\d", t) for t in toks) else 0
        # Prefer specific candidates: numeric tokens, more tokens, longer string.
        return (has_num, len(toks), len(n))

    uniq.sort(key=lambda c: _cand_score(c), reverse=True)
    return uniq


def _parse_wiki_payload(raw: str) -> Tuple[bool, str, str]:
    t = str(raw or "").strip()
    if not t or not t.lower().startswith("[wiki]"):
        return False, "", ""
    low = t.lower()
    if "request timeout" in low or "not found" in low or "http error" in low or low.endswith(":"):
        return False, "", ""
    m = re.match(r"^\[wiki\]\s*([^:]+):\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return False, "", ""
    term = str(m.group(1) or "").strip()
    summary = str(m.group(2) or "").strip()
    if not term or not summary:
        return False, "", ""
    return True, term, summary


def _build_wiki_facts(term: str, summary: str) -> str:
    return "\n".join(
        [
            "[WIKI FACTS]",
            f"Term: {term}",
            f"Summary: {summary}",
            "Source: Wiki",
            "---",
        ]
    ).strip()


def _track_b_query_tokens(text: str) -> List[str]:
    toks = [t for t in _TOKEN_RE.findall(_norm_text(text)) if t]
    out: List[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in _COMMON_WORDS:
            continue
        out.append(t)
    return out


def _build_track_b_deterministic_answer(term: str, summary: str, *, user_text: str) -> str:
    t = str(term or "").strip()
    s = str(summary or "").strip()
    if not t or not s:
        return ""
    q_tokens = set(_track_b_query_tokens(user_text))
    s_tokens = set(_TOKEN_RE.findall(_norm_text(s)))
    # Fail-loud when retrieved summary does not cover requested query tokens.
    if q_tokens and not q_tokens.intersection(s_tokens):
        return "Not available in retrieved wiki facts."
    # Slot-aware guard: require at least one meaningful pre-entity token
    # (tokens before first numeric anchor like "97th") to appear in summary.
    q_norm_toks = _TOKEN_RE.findall(_norm_text(user_text))
    first_num_idx = -1
    for i, tok in enumerate(q_norm_toks):
        if re.search(r"\d", tok):
            first_num_idx = i
            break
    if first_num_idx > 0:
        pre_entity = [
            tok for tok in q_norm_toks[:first_num_idx]
            if len(tok) >= 3 and tok not in _COMMON_WORDS
        ]
        s_norm = _norm_text(s)
        if pre_entity and not any(tok in s_tokens for tok in pre_entity):
            return "Not available in retrieved wiki facts."
        # Phrase-level slot guard for multi-token asks (e.g., "best picture").
        if len(pre_entity) >= 2:
            pre_phrases = [
                f"{pre_entity[i]} {pre_entity[i+1]}"
                for i in range(len(pre_entity) - 1)
            ]
            if pre_phrases and not any(p in s_norm for p in pre_phrases):
                return "Not available in retrieved wiki facts."
    return f"According to Wikipedia summary for {t}: {s}"


def _build_track_b_constraints() -> str:
    return "\n".join(
        [
            "Wiki grounding policy:",
            "- Use WIKI FACTS in this turn as the factual source.",
            "- If the requested detail is not present in WIKI FACTS, say it is not available.",
            "- Do not guess or add unsupported details.",
        ]
    )


def resolve_cheatsheets_turn(
    *,
    user_text: str,
    macro: str,
    subsignals: Iterable[str],
    cheatsheets_dir: Path,
    has_existing_facts: bool,
    prior_distress_carry: bool,
    prior_user_text: str,
    track_b_enabled: bool,
    wiki_lookup_fn: Optional[Callable[[str], str]],
) -> CheatsheetsTurnResult:
    entries = load_cheatsheets_entries(cheatsheets_dir)
    subs = {str(s).strip().lower() for s in (subsignals or []) if str(s or "").strip()}
    macro_l = str(macro or "").strip().lower()
    q_framing = _question_framing(user_text)
    personal_or_distress = bool(
        macro_l == "personal"
        or (_DISTRESS_SUBS & subs)
        or _PERSONAL_OR_DISTRESS_RE.search(" ".join(sorted(subs)))
        or bool(prior_distress_carry)
        or _DISTRESS_TEXT_RE.search(str(user_text or ""))
        or _DISTRESS_TEXT_RE.search(str(prior_user_text or ""))
    )
    user_clean = str(user_text or "").strip()

    # Deterministic command lane.
    if not personal_or_distress:
        if _LIST_INDEX_EXACT_RE.match(user_clean):
            topics = _load_index_topics(cheatsheets_dir, entries)
            line = _build_local_knowledge_line(topics)
            return CheatsheetsTurnResult(
                facts_block="",
                constraints_block="",
                deterministic_answer=_escape_markdown_leading_quotes(line),
                local_knowledge_line="",
                track="A",
                footer_source="Cheatsheets",
                footer_confidence="unverified",
                kb_lookup_candidate=True,
                matched_terms=tuple(),
                wiki_term="",
            )

        list_m = _LIST_CMD_RE.match(user_clean)
        if list_m:
            arg = _norm_text(_LEADING_ARTICLE_RE.sub("", str(list_m.group(1) or "").strip()))
            by_term = _term_exact_entries(entries, arg)
            if by_term:
                e = by_term[0]
                return CheatsheetsTurnResult(
                    facts_block="",
                    constraints_block="",
                    deterministic_answer=_escape_markdown_leading_quotes(f"{e.term}: {e.definition}"),
                    local_knowledge_line="",
                    track="A",
                    footer_source="Cheatsheets",
                    footer_confidence=str(e.confidence or "high").strip().lower() or "high",
                    kb_lookup_candidate=True,
                    matched_terms=tuple(x.term for x in by_term),
                    wiki_term="",
                )
            by_cat = _category_entries(entries, arg)
            return CheatsheetsTurnResult(
                facts_block="",
                constraints_block="",
                deterministic_answer=_escape_markdown_leading_quotes(_build_list_category_answer(arg, by_cat)),
                local_knowledge_line="",
                track="A",
                footer_source="Cheatsheets",
                footer_confidence="high" if by_cat else "unverified",
                kb_lookup_candidate=True,
                matched_terms=tuple(x.term for x in by_cat),
                wiki_term="",
            )

        summary_m = _SUMMARY_CMD_RE.match(user_clean)
        if summary_m:
            arg = _norm_text(_LEADING_ARTICLE_RE.sub("", str(summary_m.group(1) or "").strip()))
            by_term = _term_exact_entries(entries, arg)
            if len(by_term) == 1:
                e = by_term[0]
                return CheatsheetsTurnResult(
                    facts_block="",
                    constraints_block="",
                    deterministic_answer=_escape_markdown_leading_quotes(_build_summary_answer(arg, [e])),
                    local_knowledge_line="",
                    track="A",
                    footer_source="Cheatsheets",
                    footer_confidence=str(e.confidence or "high").strip().lower() or "high",
                    kb_lookup_candidate=True,
                    matched_terms=(e.term,),
                    wiki_term="",
                )
            rows = by_term if by_term else _category_entries(entries, arg)
            return CheatsheetsTurnResult(
                facts_block="",
                constraints_block="",
                deterministic_answer=_escape_markdown_leading_quotes(_build_summary_answer(arg, rows)),
                local_knowledge_line="",
                track="A",
                footer_source="Cheatsheets",
                footer_confidence="high",
                kb_lookup_candidate=True,
                matched_terms=tuple(x.term for x in rows),
                wiki_term="",
            )

    index_query = bool(_INDEX_QUERY_RE.match(str(user_text or "").strip()))
    local_geo_env = bool(_LOCAL_GEO_ENV_RE.search(str(user_text or "")))
    local_intent = bool(
        _LOCAL_KNOWLEDGE_INTENT_RE.match(str(user_text or "").strip())
        or _LOCAL_KNOWLEDGE_TOPICS_VARIANT_RE.match(str(user_text or "").strip())
        or index_query
        or local_geo_env
    )
    if local_intent and (not personal_or_distress):
        topics = _load_index_topics(cheatsheets_dir, entries)
        if topics:
            if local_geo_env:
                return CheatsheetsTurnResult(
                    facts_block="",
                    constraints_block="",
                    deterministic_answer=_escape_markdown_leading_quotes(_build_local_knowledge_geo_guard_line(topics)),
                    local_knowledge_line="",
                    track="A",
                    footer_source="Cheatsheets",
                    footer_confidence="high",
                    kb_lookup_candidate=True,
                    matched_terms=tuple(),
                    wiki_term="",
                )
            return CheatsheetsTurnResult(
                facts_block="",
                constraints_block="",
                deterministic_answer="",
                local_knowledge_line=_build_local_knowledge_line(topics),
                track="A",
                footer_source="Mixed",
                footer_confidence="unverified",
                kb_lookup_candidate=True,
                matched_terms=tuple(),
                wiki_term="",
            )

    requested_topic_norm = _extract_requested_topic(user_text)
    if requested_topic_norm and (not personal_or_distress):
        scoped_entries = _topic_entries(entries, requested_topic_norm)
        if scoped_entries:
            facts = _build_cheatsheets_facts([(e, 2) for e in scoped_entries])
            deterministic_answer = _build_topic_deterministic_answer(requested_topic_norm, scoped_entries)
            return CheatsheetsTurnResult(
                facts_block=facts,
                constraints_block=_build_track_a_constraints(strict_lookup=True),
                deterministic_answer=_escape_markdown_leading_quotes(deterministic_answer),
                local_knowledge_line="",
                track="A",
                footer_source="Cheatsheets",
                footer_confidence="high",
                kb_lookup_candidate=True,
                matched_terms=tuple(e.term for e in scoped_entries),
                wiki_term="",
            )

    matches = _select_matches(entries, user_text)
    kb_lookup_candidate = bool(matches)
    if matches:
        # Track A lane restriction: do not inject on personal/distress unless explicit question framing.
        if personal_or_distress and (not q_framing):
            return CheatsheetsTurnResult(
                facts_block="",
                constraints_block="",
                deterministic_answer="",
                local_knowledge_line="",
                track="",
                footer_source="",
                footer_confidence="",
                kb_lookup_candidate=kb_lookup_candidate,
                matched_terms=tuple(e.term for e, _ in matches),
                wiki_term="",
            )
        strict_lookup = _is_strict_lookup_query(user_text=user_text, matches=matches)
        direct_hits = _direct_term_matches(user_text=user_text, matches=matches)
        facts = _build_cheatsheets_facts(matches)
        constraints = _build_track_a_constraints(strict_lookup=strict_lookup)
        top_conf = str(matches[0][0].confidence or "medium").strip().lower()
        if top_conf not in {"low", "medium", "high", "top", "unverified"}:
            top_conf = "medium"
        fully_grounded = bool(strict_lookup and (not has_existing_facts))
        source = "Cheatsheets" if fully_grounded else "Mixed"
        conf = top_conf if fully_grounded else "medium"
        deterministic_answer = ""
        if fully_grounded and len(direct_hits) == 1:
            t = str(direct_hits[0].term or "").strip()
            d = str(direct_hits[0].definition or "").strip()
            if t and d:
                deterministic_answer = f"{t}: {d}"
            else:
                deterministic_answer = d or t
        return CheatsheetsTurnResult(
            facts_block=facts,
            constraints_block=constraints,
            deterministic_answer=_escape_markdown_leading_quotes(deterministic_answer),
            local_knowledge_line="",
            track="A",
            footer_source=source,
            footer_confidence=conf,
            kb_lookup_candidate=kb_lookup_candidate,
            matched_terms=tuple(e.term for e, _ in matches),
            wiki_term="",
        )

    # Track B gate:
    # Track B is intentionally working-only by design; do not harmonize silently.
    if not track_b_enabled:
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")
    if macro_l != "working":
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")
    if not q_framing:
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")
    if wiki_lookup_fn is None:
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")

    known_norm_terms = {e.norm_term for e in entries if e.norm_term}
    candidates = _wiki_candidates(user_text, known_norm_terms)
    if not candidates:
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")
    ok = False
    title = ""
    summary = ""
    for term in candidates:
        ok, title, summary = _parse_wiki_payload(str(wiki_lookup_fn(term) or ""))
        if ok:
            break
    if not ok:
        return CheatsheetsTurnResult("", "", "", "", "", "", "", False, tuple(), "")
    facts = _build_wiki_facts(title, summary)
    source = "Mixed" if has_existing_facts else "Wiki"
    conf = "medium"
    return CheatsheetsTurnResult(
        facts_block=facts,
        constraints_block=_build_track_b_constraints(),
        deterministic_answer=_build_track_b_deterministic_answer(title, summary, user_text=user_text),
        local_knowledge_line="",
        track="B",
        footer_source=source,
        footer_confidence=conf,
        kb_lookup_candidate=False,
        matched_terms=tuple(),
        wiki_term=title,
    )
