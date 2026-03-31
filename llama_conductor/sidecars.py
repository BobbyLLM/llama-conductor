"""Deterministic utility sidecars (non-LLM retrieval/calculation).

Includes helpers for:
- calculator and memory utilities
- quote/KB search helpers
- wiki, define, exchange, and weather lookups
"""

from __future__ import annotations
import re
import html as ihtml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import requests
from urllib.parse import quote, urlparse, parse_qs, unquote

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CalcResult:
    """Result of >>calc operation."""
    value: Optional[float] = None
    formatted: str = ""
    error: Optional[str] = None


@dataclass
class MemoryEntry:
    """Single Vodka memory entry for >>list."""
    ctx_id: str
    text: str
    ttl_days: int
    touch_count: int
    created_at: str


@dataclass
class QuoteResult:
    """Result of >>find / >>quote operation."""
    file: str
    rel_path: str
    kb: str
    snippet: str
    location: str  # "line X" or similar context


@dataclass
class WikiResult:
    """Result of >>wiki query."""
    title: Optional[str] = None
    summary: str = ""
    error: Optional[str] = None


@dataclass
class DefineResult:
    """Result of >>define query."""
    term: Optional[str] = None
    etymology: str = ""
    error: Optional[str] = None


@dataclass
class ExchangeResult:
    """Result of >>exchange query."""
    amount: Optional[float] = None
    from_ccy: Optional[str] = None
    to_ccy: Optional[str] = None
    converted: Optional[float] = None
    error: Optional[str] = None


@dataclass
class WeatherResult:
    """Result of >>weather query."""
    location: Optional[str] = None
    condition: str = ""
    error: Optional[str] = None


@dataclass
class WebSearchHit:
    """Normalized web search hit.

    Core fields (always populated at instantiation):
        title, url, snippet, timestamp, source, rank, score

    Extended metadata fields (default to sentinel; populated by retrieval pipeline):
        domain        -- eTLD+1 extracted from url (e.g. 'pubmed.ncbi.nlm.nih.gov')
        source_type   -- categorical: 'official' | 'wiki' | 'reference' | 'forum' | 'news' | 'academic' | 'unknown'
        serp_score    -- rank-derived score component (0.0 until scoring wired)
        content_score -- content quality score component (0.0 until scoring wired)
        corroboration_score -- cross-hit agreement score (0.0 until scoring wired)
        fetch_status  -- 'skipped' | 'pending' | 'fetched' | 'failed'
        canonical_url -- resolved URL post-redirect ('' means use url)
    """
    title: str
    url: str
    snippet: str
    timestamp: str
    source: str
    rank: int
    score: float = 0.0
    # --- extended metadata (swarm/observability groundwork) ---
    domain: str = ""
    source_type: str = "unknown"
    serp_score: float = 0.0
    content_score: float = 0.0
    corroboration_score: float = 0.0
    fetch_status: str = "skipped"  # sentinel: fetch not attempted; 'pending' reserved for queued fetch
    canonical_url: str = ""


# ============================================================================
# Calculator (>>calc)
# ============================================================================


def parse_and_eval_calc(expr: str) -> CalcResult:
    """
    Parse and evaluate a mathematical expression safely.
    
    Supports: +, -, *, /, %, **, parentheses, numbers, basic functions
    Examples:
        >>calc 30% of 79.95
        >>calc 14*365
        >>calc (100 + 50) / 2
        >>calc 5**2
    """
    expr = (expr or "").strip()
    if not expr:
        return CalcResult(error="Empty expression")

    try:
        # Safe eval with limited builtins
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        # Handle percentage syntax: "30% of 79.95"
        expr_normalized = _normalize_calc_expression(expr)

        # Evaluate
        result = eval(expr_normalized, {"__builtins__": {}}, allowed_names)

        # Format result
        if isinstance(result, float):
            if result == int(result):
                formatted = f"{int(result)}"
            else:
                formatted = f"{result:.2f}"
        else:
            formatted = str(result)

        return CalcResult(value=float(result), formatted=formatted)

    except ZeroDivisionError:
        return CalcResult(error="Division by zero")
    except ValueError as e:
        return CalcResult(error=f"Invalid value: {e}")
    except SyntaxError:
        return CalcResult(error="Invalid expression syntax")
    except NameError as e:
        return CalcResult(error=f"Unknown function or variable: {e}")
    except Exception as e:
        return CalcResult(error=f"Calculation error: {e}")


def _normalize_calc_expression(expr: str) -> str:
    """Normalize common expression patterns."""
    expr = expr.strip()

    # Handle "X% of Y" → "(X / 100) * Y"
    if "% of " in expr.lower():
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)",
            r"(\1 / 100) * \2",
            expr,
            flags=re.IGNORECASE,
        )

    # Handle "X per Y" → "X / Y"
    if " per " in expr.lower():
        expr = re.sub(
            r"(\d+(?:\.\d+)?)\s+per\s+(\d+(?:\.\d+)?)",
            r"\1 / \2",
            expr,
            flags=re.IGNORECASE,
        )

    return expr


def format_calc_result(result: CalcResult) -> str:
    """Format calculator result for user display."""
    if result.error:
        return f"[calc error] {result.error}"
    return result.formatted


# ============================================================================
# Vodka Memory Listing (>>list)
# ============================================================================


def list_vodka_memories(vodka: Optional[object]) -> List[MemoryEntry]:
    """
    List all stored Vodka memories with metadata.
    
    Returns: List of MemoryEntry objects (empty if vodka is None or has no memories)
    """
    if vodka is None:
        return []

    try:
        # Access Vodka's storage (vodka_filter.py uses _get_storage_and_fr())
        fr = vodka._get_storage_and_fr()
        if fr is None:
            return []

        data = fr.S.load_facts()
        now = fr._now()

        entries: List[MemoryEntry] = []

        for ctx_id, rec in data.items():
            if rec.get("type") != "vodka_ctx":
                continue

            # Skip expired
            exp_s = rec.get("expires_at")
            if exp_s:
                exp_dt = fr._parse_ts(exp_s)
                if exp_dt and exp_dt < now:
                    continue

            # Calculate TTL days remaining
            ttl_days = 0
            if exp_s:
                exp_dt = fr._parse_ts(exp_s)
                if exp_dt:
                    delta = (exp_dt - now).total_seconds()
                    ttl_days = max(0, int(delta // 86400))

            touch_count = int(rec.get("touch_count", 0))
            text = str(rec.get("value", "")).strip()
            created_at = str(rec.get("created_at", "")).strip()

            entries.append(
                MemoryEntry(
                    ctx_id=ctx_id,
                    text=text,
                    ttl_days=ttl_days,
                    touch_count=touch_count,
                    created_at=created_at,
                )
            )

        # Sort by creation date (newest first)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    except Exception as e:
        print(f"[sidecars.list_vodka_memories] Error: {e}")
        return []


def format_memory_list(entries: List[MemoryEntry]) -> str:
    """Format Vodka memory list for user display."""
    if not entries:
        return "[list] No stored memories."

    lines = ["[vodka memories]"]
    for i, entry in enumerate(entries, 1):
        preview = entry.text[:80].replace("\n", " ").strip()
        lines.append(
            f"{i}. [{entry.ctx_id}] {preview}... (TTL={entry.ttl_days}d, touches={entry.touch_count})"
        )

    return "\n".join(lines)


# ============================================================================
# KB Quote Finding (>>find / >>quote)
# ============================================================================


def find_quote_in_kbs(
    query: str,
    attached_kbs: Set[str],
    kb_paths: Dict[str, str],
) -> Optional[QuoteResult]:
    """
    Find a quote/passage in attached KBs.
    
    Searches SUMM_*.md files in attached KB folders.
    Returns first exact match, or closest substring match.
    """
    query = (query or "").strip().lower()
    if not query:
        return None

    # Try exact substring match first
    for kb in sorted(attached_kbs):
        if kb == "vault":
            continue  # Skip vault, it's Qdrant-based
        
        folder = kb_paths.get(kb)
        if not folder:
            continue

        result = _search_kb_folder(query, kb, folder, exact=True)
        if result:
            return result

    # Fall back to word-token match
    for kb in sorted(attached_kbs):
        if kb == "vault":
            continue
        
        folder = kb_paths.get(kb)
        if not folder:
            continue

        result = _search_kb_folder(query, kb, folder, exact=False)
        if result:
            return result

    return None


def _search_kb_folder(query: str, kb: str, folder: str, exact: bool = True) -> Optional[QuoteResult]:
    """Search a single KB folder for a query."""
    import os

    if not os.path.isdir(folder):
        return None

    for root, _, files in os.walk(folder):
        # Skip /original/ subfolder
        if "original" in {p.lower() for p in root.split(os.sep)}:
            continue

        for fn in sorted(files):
            if not fn.startswith("SUMM_") or not fn.lower().endswith(".md"):
                continue

            fpath = os.path.join(root, fn)
            rel_path = os.path.relpath(fpath, folder)

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue

            # Search in text
            result = _find_in_text(query, text, kb, fn, rel_path, exact=exact)
            if result:
                return result

    return None


def _find_in_text(
    query: str, text: str, kb: str, file: str, rel_path: str, exact: bool = True
) -> Optional[QuoteResult]:
    """Search for query within text content."""
    query_lower = query.lower()

    if exact:
        # Exact substring match
        idx = text.lower().find(query_lower)
        if idx >= 0:
            # Extract snippet (context)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 100)
            snippet = text[start:end].strip()

            # Calculate approximate line number
            line_num = text[:idx].count("\n") + 1

            return QuoteResult(
                file=file,
                rel_path=rel_path,
                kb=kb,
                snippet=snippet,
                location=f"line ~{line_num}",
            )
    else:
        # Word token match: all query words must be in text
        query_words = set(w.lower() for w in query.split() if w)
        text_lower = text.lower()

        all_found = all(w in text_lower for w in query_words)
        if all_found:
            # Find first occurrence of any query word
            first_pos = len(text)
            for word in query_words:
                pos = text_lower.find(word)
                if pos >= 0:
                    first_pos = min(first_pos, pos)

            start = max(0, first_pos - 100)
            end = min(len(text), first_pos + 200)
            snippet = text[start:end].strip()
            line_num = text[:first_pos].count("\n") + 1

            return QuoteResult(
                file=file,
                rel_path=rel_path,
                kb=kb,
                snippet=snippet,
                location=f"line ~{line_num}",
            )

    return None


def format_quote_result(result: QuoteResult) -> str:
    """Format quote search result for user display."""
    if not result:
        return "[find] Quote not found."

    return (
        f"[find] Located in: {result.kb}/{result.rel_path}\n"
        f"File: {result.file}\n"
        f"Location: {result.location}\n\n"
        f"Snippet:\n{result.snippet}"
    )


# ============================================================================
# CTC Cache Flushing (>>flush)
# ============================================================================


def flush_ctc_cache(vodka: Optional[object]) -> str:
    """
    Flush the CTC (Cut-The-Crap) rolling cache in Vodka.
    
    This is a message history trimming cache, not the memory store.
    Flushing resets the cache for the next turn.
    """
    if vodka is None:
        return "[flush] Vodka not initialized."

    try:
        # CTC cache is the trimmed message list. We signal a "reset" by clearing
        # any cached state. Since messages are per-request, we just confirm.
        # The actual CTC happens at inlet() time in Vodka.
        
        # If Vodka has a reset method, call it. Otherwise, just confirm.
        if hasattr(vodka, "_last_janitor_run"):
            # This is a signal to vodka to re-run janitor next time
            vodka._last_janitor_run = 0.0

        return "[flush] CTC cache reset for next turn. (Memory store preserved.)"

    except Exception as e:
        return f"[flush] Error: {e}"


# ============================================================================
# Wikipedia (>>wiki)
# ============================================================================


def _normalize_wiki_topic(topic: str) -> str:
    """Normalize topic for Wikipedia URL: 'Boris Becker' → 'Boris_Becker'."""
    t = re.sub(r"\s+", " ", str(topic or "").strip())
    t = re.sub(r"[?!.,;:]+$", "", t).strip()
    t = re.sub(
        r"^(?:who|what|when|where|why|how)\s+(?:is|are|was|were)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"^(?:tell\s+me\s+about|explain)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^the\s+", "", t, flags=re.IGNORECASE)
    return t.replace(" ", "_")


def _wiki_search_best_title(query: str, timeout_s: int = 5) -> str:
    """Resolve a best-effort Wikipedia page title via OpenSearch."""
    q = re.sub(r"\s+", " ", str(query or "").strip())
    if not q:
        return ""
    try:
        headers = {"User-Agent": "llama-conductor/1.0.1"}
        params = {
            "action": "opensearch",
            "search": q,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            headers=headers,
            params=params,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list) and data[1]:
            title = str(data[1][0] or "").strip()
            return title
    except Exception:
        return ""
    return ""


_WIKI_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WIKI_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "did", "do",
}


def _wiki_query_tokens(query: str) -> list[str]:
    toks = [t.lower() for t in _WIKI_TOKEN_RE.findall(str(query or ""))]
    out: list[str] = []
    for t in toks:
        if len(t) < 2 or t in _WIKI_STOPWORDS:
            continue
        out.append(t)
    return out


def _wiki_sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [p.strip() for p in parts if str(p or "").strip()]


def _wiki_best_extract_sentences(*, query: str, extract_text: str, max_sentences: int = 2) -> str:
    """Deterministic extraction:
    score = query-token intersection count + numeric bonus.
    """
    q_tokens = _wiki_query_tokens(query)
    if not q_tokens:
        return ""
    sentences = _wiki_sentence_split(extract_text)
    if not sentences:
        return ""
    scored: list[tuple[int, int]] = []
    q_set = set(q_tokens)
    for i, s in enumerate(sentences):
        s_tokens = [t.lower() for t in _WIKI_TOKEN_RE.findall(s)]
        if not s_tokens:
            continue
        s_set = set(s_tokens)
        inter = len(q_set.intersection(s_set))
        has_numeric = 1 if any(re.search(r"\d", t) for t in s_tokens) else 0
        score = int(inter) + int(has_numeric)
        if score > 0:
            scored.append((score, i))
    if not scored:
        return ""
    # Tie-break: higher score, then original order.
    scored.sort(key=lambda row: (-row[0], row[1]))
    top_i = scored[0][1]
    picks = [top_i]
    if len(picks) < max(1, int(max_sentences)):
        for score, i in scored[1:]:
            if abs(i - top_i) <= 1:
                picks.append(i)
                break
    picks = sorted(set(picks))
    return " ".join(sentences[i].rstrip() for i in picks if 0 <= i < len(sentences)).strip()


def handle_wiki_query(topic: str, max_chars: int = 800) -> str:
    """
    Fetch Wikipedia summary via JSON API.
    
    Example: >>wiki Albert Einstein
    Returns: "[wiki] Albert Einstein: A German-born theoretical physicist..."
    Fetches summary/extractive response (up to max_chars chars).
    """
    topic = (topic or "").strip()
    if not topic:
        return "[wiki] No topic provided"

    headers = {"User-Agent": "llama-conductor/1.0.1"}
    normalized = _normalize_wiki_topic(topic)

    def _fetch_summary(normalized_topic: str) -> tuple[int, dict]:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{normalized_topic}"
        resp = requests.get(url, headers=headers, timeout=5)
        status = int(resp.status_code or 0)
        data = resp.json() if status >= 200 and status < 300 else {}
        return status, data

    def _fetch_extract_by_title(title: str, chars: int = 5000) -> str:
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 0,
                "redirects": 1,
                "titles": str(title or "").strip(),
            }
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                headers=headers,
                params=params,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            pages = (((data or {}).get("query") or {}).get("pages") or {})
            if isinstance(pages, dict):
                for _pid, row in pages.items():
                    if not isinstance(row, dict):
                        continue
                    ex = str(row.get("extract", "") or "").strip()
                    if ex:
                        ex = re.sub(r"\s+", " ", ex).strip()
                        if len(ex) > int(chars):
                            ex = ex[: int(chars)]
                        return ex
        except Exception:
            return ""
        return ""

    try:
        status, data = _fetch_summary(normalized)
        if status == 404:
            best = _wiki_search_best_title(topic)
            if best:
                status, data = _fetch_summary(_normalize_wiki_topic(best))
        if status == 404:
            return f"[wiki] '{topic}' not found"
        if status < 200 or status >= 300:
            return f"[wiki] HTTP error: {status}"

        title = data.get("title") or normalized.replace("_", " ")
        summary = (data.get("extract") or "").strip()
        if not summary:
            return f"[wiki] '{title}' not found"

        # Question-shaped asks: deterministic extractive selection from page text.
        if bool(re.search(r"[?]|^(?:who|what|when|where|why|how)\b", str(topic or "").strip(), flags=re.IGNORECASE)):
            full_extract = _fetch_extract_by_title(title, chars=5000)
            picked = _wiki_best_extract_sentences(
                query=str(topic or ""),
                extract_text=full_extract,
                max_sentences=2,
            )
            if picked:
                summary = picked
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
        return f"[wiki] {title}: {summary}"
    except requests.exceptions.Timeout:
        return "[wiki] Request timeout"
    except Exception as e:
        return f"[wiki] Error: {e}"


# ============================================================================
# Etymology Definition (>>define)
# ============================================================================


_WS_RE = re.compile(r"\s+")


def _normalize_define_term(term: str) -> str:
    t = " ".join((term or "").strip().split())
    # Drop trailing punctuation that often appears in natural prompts.
    t = re.sub(r"[.!?,;:)\]\"'`]+$", "", t)
    t = re.sub(r"^[([\"'`]+", "", t)
    return t.strip()


def handle_define_query(term: str, max_chars: int = 500) -> str:
    """
    Fetch and normalize etymology from Etymonline first entry block.
    """
    q = _normalize_define_term(term)
    if not q:
        return "[define] No term provided"

    try:
        url = f"https://www.etymonline.com/word/{quote(q)}"
        resp = requests.get(url, headers={"User-Agent": "llama-conductor/1.5.2 (+define etymonline)"}, timeout=8)
        resp.raise_for_status()
        html = resp.text

        # First etymology entry block on page.
        m = re.search(
            r"<section class=\"prose-lg[^\"]*\"[^>]*>.*?<h2[^>]*>"
            r"(?:.*?)<span[^>]*>(?P<word>.*?)</span>\s*"
            r"<span[^>]*>(?P<pos>\(.*?\))</span>.*?</h2>"
            r".*?<section[^>]*>\s*<p>(?P<body>.*?)</p>",
            html,
            flags=re.S | re.I,
        )
        if not m:
            return f"[define] '{q}' not found or entry parse failed"

        def _clean_html_text(s: str) -> str:
            t = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.S | re.I)
            t = re.sub(r"<style[^>]*>.*?</style>", " ", t, flags=re.S | re.I)
            t = re.sub(r"</?(i|em|b|strong)[^>]*>", "", t, flags=re.I)
            t = re.sub(r"<a[^>]*>", "", t, flags=re.I)
            t = t.replace("</a>", "")
            t = re.sub(r"<[^>]+>", " ", t)
            t = ihtml.unescape(t)
            t = _WS_RE.sub(" ", t).strip()
            return t

        word = _clean_html_text(m.group("word"))
        pos = _clean_html_text(m.group("pos"))
        body = _clean_html_text(m.group("body"))
        if not body:
            return f"[define] '{q}' found, but etymology body was empty"
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + "..."

        return f"[define] {word}{pos}\n\n{body}"

    except requests.exceptions.Timeout:
        return "[define] Request timeout"
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"[define] '{q}' not found"
        if e.response is not None and e.response.status_code == 403:
            return "[define] Upstream blocked request (403). Retry later."
        return f"[define] HTTP error: {e.response.status_code if e.response is not None else 'unknown'}"
    except Exception as e:
        return f"[define] Error: {e}"


# ============================================================================
# Web Search (>>web)
# ============================================================================

_WEB_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WEB_QUOTED_RE = re.compile(r"['\"]([^'\"\n]{2,160})['\"]")
_WEB_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "of", "on", "or", "quote", "source", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "who", "why", "with",
}
_WEB_TRUST_DOMAINS = {
    "imdb.com",
    "wikipedia.org",
    "wikiquote.org",
    "tvtropes.org",
    "fandom.com",
    "simpsons.fandom.com",
    "springfieldspringfield.co.uk",
    "npr.org",
    "bbc.co.uk",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "abc.net.au",
    "wdsu.com",
    "6abc.com",
}
_WEB_RELEVANCE_THRESHOLD = 3.0
_WEB_USER_TRUST_DOMAINS_MAX = 100


def _web_cfg(path: str, default):
    try:
        from .config import cfg_get  # lazy import to avoid hard startup coupling

        return cfg_get(path, default)
    except Exception:
        return default


def _web_cfg_list(path: str) -> list[str]:
    raw = _web_cfg(path, [])
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    if isinstance(raw, list):
        out: list[str] = []
        for p in raw:
            s = str(p or "").strip()
            if s:
                out.append(s)
        return out
    return []


def _normalize_domain_list(values: list[str], *, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        d = str(v or "").strip().lower()
        if not d:
            continue
        if d.startswith("www."):
            d = d[4:]
        # simple domain safety: labels + dots + hyphen, no protocol/path
        if not re.match(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$", d):
            continue
        if ".." in d:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
        if max_items is not None and len(out) >= int(max_items):
            break
    return out


def _effective_trust_domains() -> list[str]:
    builtins = _normalize_domain_list(list(_WEB_TRUST_DOMAINS))
    user_raw = _web_cfg_list("web_search.user_trust_domains")
    user_norm = _normalize_domain_list(user_raw, max_items=_WEB_USER_TRUST_DOMAINS_MAX)
    return _normalize_domain_list([*builtins, *user_norm])


def _web_domain(url: str) -> str:
    try:
        host = (urlparse(str(url or "")).hostname or "").strip().lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _domain_is_allowed(url: str, include_domains: list[str], exclude_domains: list[str]) -> bool:
    host = _web_domain(url)
    if not host:
        return False
    inc = [d.strip().lower() for d in include_domains if str(d or "").strip()]
    exc = [d.strip().lower() for d in exclude_domains if str(d or "").strip()]
    if inc and not any(host == d or host.endswith("." + d) for d in inc):
        return False
    if exc and any(host == d or host.endswith("." + d) for d in exc):
        return False
    return True


def _normalize_hit(
    *,
    title: str,
    url: str,
    snippet: str,
    timestamp: str,
    source: str,
    rank: int,
    include_domains: list[str],
    exclude_domains: list[str],
) -> Optional[WebSearchHit]:
    u = str(url or "").strip()
    if not u:
        return None
    if not (u.startswith("http://") or u.startswith("https://")):
        return None
    # Reject low-signal dictionary/reference domains from general web grounding.
    try:
        host_norm = _web_domain(u)
        if host_norm:
            blocked = (
                "merriam-webster.com",
                "dictionary.com",
                "vocabulary.com",
                "thesaurus.com",
                "wordreference.com",
            )
            if any(host_norm == d or host_norm.endswith("." + d) for d in blocked):
                return None
    except Exception:
        return None
    # Reject known tracker/ad redirect URLs from search wrappers.
    try:
        pu = urlparse(u)
        host = str(pu.netloc or "").lower()
        path = str(pu.path or "")
        ql = str(pu.query or "").lower()
        if host.endswith("duckduckgo.com") and (path.startswith("/y.js") or path.startswith("/y/")):
            return None
        if "ad_domain=" in ql or "ad_provider=" in ql or "aclick" in ql:
            return None
        if host.endswith("bing.com") and path.startswith("/aclick"):
            return None
    except Exception:
        return None
    if not _domain_is_allowed(u, include_domains, exclude_domains):
        return None
    t = re.sub(r"\s+", " ", str(title or "").strip())
    s = re.sub(r"\s+", " ", str(snippet or "").strip())
    ts = re.sub(r"\s+", " ", str(timestamp or "").strip())
    if not t:
        t = u
    return WebSearchHit(
        title=t[:240],
        url=u,
        snippet=s[:320],
        timestamp=ts[:80],
        source=str(source or "").strip() or "web",
        rank=max(1, int(rank or 1)),
        score=0.0,
    )


def _ddg_extract_url(raw_url: str) -> str:
    u = str(raw_url or "").strip()
    if not u:
        return ""
    if "duckduckgo.com/l/?" in u:
        try:
            parsed = urlparse(u)
            qs = parse_qs(parsed.query)
            uddg = (qs.get("uddg") or [""])[0].strip()
            if uddg:
                return unquote(uddg)
        except Exception:
            return ""
    return u


def _web_norm_text(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _query_nonstop_tokens(query: str) -> list[str]:
    toks = [t.lower() for t in _WEB_TOKEN_RE.findall(str(query or ""))]
    out: list[str] = []
    for t in toks:
        if len(t) < 2:
            continue
        if t in _WEB_STOPWORDS:
            continue
        out.append(t)
    return out


def _quote_phrase_candidates(query: str) -> list[str]:
    out: list[str] = []

    # Quoted input remains highest-confidence phrase evidence.
    for q in _WEB_QUOTED_RE.findall(str(query or "")):
        n = _web_norm_text(q)
        if n:
            out.append(n)
    if out:
        return out

    raw_tokens = [t.lower() for t in _WEB_TOKEN_RE.findall(_web_norm_text(query))]
    if not raw_tokens:
        return out

    # Strip framing words so phrase matching targets content-bearing chunks.
    phrase_stop = set(_WEB_STOPWORDS).union(
        {
            "won",
            "winner",
            "winners",
            "winning",
            "please",
            "can",
            "could",
            "would",
            "you",
        }
    )
    tokens = [t for t in raw_tokens if t not in phrase_stop]
    if len(tokens) < 2:
        return out

    seen: set[str] = set()

    def _push(cand: str) -> None:
        c = _web_norm_text(cand)
        if not c or c in seen:
            return
        seen.add(c)
        out.append(c)

    # Prefer most-specific phrase first, then bounded n-grams (4..2).
    _push(" ".join(tokens))
    max_n = min(4, len(tokens))
    for n in range(max_n, 1, -1):
        for i in range(0, len(tokens) - n + 1):
            _push(" ".join(tokens[i : i + n]))

    return out


def _domain_trust_boost(url: str) -> float:
    host = _web_domain(url)
    if not host:
        return 0.0
    for d in _effective_trust_domains():
        if host == d or host.endswith("." + d):
            return 1.0
    return 0.0


def _score_hit(query: str, hit: WebSearchHit) -> float:
    text = _web_norm_text(f"{hit.title} {hit.snippet}")
    phrase_score = 0.0
    for p in _quote_phrase_candidates(query):
        if p and p in text:
            phrase_score = 5.0
            break
    q_toks = _query_nonstop_tokens(query)
    ratio = 0.0
    if q_toks:
        q_set = set(q_toks)
        s_set = set(_WEB_TOKEN_RE.findall(text))
        overlap = len(q_set.intersection(s_set))
        ratio = float(overlap) / float(max(1, len(q_set)))
    token_score = ratio * 3.0
    trust = _domain_trust_boost(hit.url)
    return round(phrase_score + token_score + trust, 2)


def _relevance_passes(score: float) -> bool:
    return float(score) >= float(_WEB_RELEVANCE_THRESHOLD)


def _web_search_ddg_lite(
    query: str,
    *,
    timeout_sec: float,
    max_results: int,
    include_domains: list[str],
    exclude_domains: list[str],
) -> tuple[list[WebSearchHit], str]:
    headers = {"User-Agent": "llama-conductor/1.8.0 (+web ddg-lite)"}
    url = "https://lite.duckduckgo.com/lite/"
    resp = requests.get(url, params={"q": query}, headers=headers, timeout=timeout_sec)
    # DDG anomaly/challenge can return 202 with HTML that is not search results.
    if int(resp.status_code) != 200:
        return [], f"http status {resp.status_code}"
    html = str(resp.text or "")
    low = html.lower()
    if ("anomaly-modal" in low) or ("bots use duckduckgo too" in low) or ("anomaly.js" in low):
        return [], "ddg challenge page"
    rows = re.findall(
        r"<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*href=|$)",
        html,
        flags=re.I | re.S,
    )
    if not rows:
        return [], "parse-shape mismatch"
    out: list[WebSearchHit] = []
    for i, (href, title_html, tail_html) in enumerate(rows, start=1):
        if len(out) >= max_results:
            break
        title = re.sub(r"<[^>]+>", " ", ihtml.unescape(title_html or ""))
        tail_txt = re.sub(r"<[^>]+>", " ", ihtml.unescape(tail_html or ""))
        snippet = re.sub(r"\s+", " ", tail_txt).strip()
        resolved = _ddg_extract_url(href)
        hit = _normalize_hit(
            title=title,
            url=resolved,
            snippet=snippet,
            timestamp="",
            source="ddg_lite",
            rank=i,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        if hit is not None:
            out.append(hit)
    return out, ""


def _web_search_tavily(
    query: str,
    *,
    timeout_sec: float,
    max_results: int,
    include_domains: list[str],
    exclude_domains: list[str],
    recency_days: int,
) -> tuple[list[WebSearchHit], str]:
    api_key = str(_web_cfg("web_search.tavily.api_key", "") or "").strip()
    if not api_key:
        api_key = str(_web_cfg("web_search.api_key", "") or "").strip()
    if not api_key:
        return [], "missing api key"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    if recency_days > 0:
        payload["days"] = recency_days
    resp = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        headers={"User-Agent": "llama-conductor/1.8.0 (+web tavily)"},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    rows = data.get("results", []) if isinstance(data, dict) else []
    out: list[WebSearchHit] = []
    for i, row in enumerate(rows, start=1):
        if len(out) >= max_results:
            break
        if not isinstance(row, dict):
            continue
        hit = _normalize_hit(
            title=str(row.get("title", "") or ""),
            url=str(row.get("url", "") or ""),
            snippet=str(row.get("content", "") or row.get("snippet", "") or ""),
            timestamp=str(row.get("published_date", "") or row.get("date", "") or ""),
            source="tavily",
            rank=i,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        if hit is not None:
            out.append(hit)
    return out, ""


def _web_search_json_provider(
    *,
    provider: str,
    base_url: str,
    query: str,
    timeout_sec: float,
    max_results: int,
    include_domains: list[str],
    exclude_domains: list[str],
    safe_search: bool,
    recency_days: int,
) -> tuple[list[WebSearchHit], str]:
    if not base_url:
        return [], "missing base_url"
    params = {"q": query, "format": "json"}
    if provider == "searxng":
        params["safesearch"] = 1 if safe_search else 0
        if recency_days > 0:
            params["time_range"] = "day" if recency_days <= 1 else "month"
    headers = {"User-Agent": f"llama-conductor/1.8.0 (+web {provider})"}
    resp = requests.get(base_url, params=params, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    rows = []
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            rows = data.get("results", [])
        elif isinstance(data.get("items"), list):
            rows = data.get("items", [])
    out: list[WebSearchHit] = []
    for i, row in enumerate(rows, start=1):
        if len(out) >= max_results:
            break
        if not isinstance(row, dict):
            continue
        hit = _normalize_hit(
            title=str(row.get("title", "") or ""),
            url=str(row.get("url", "") or row.get("link", "") or ""),
            snippet=str(row.get("content", "") or row.get("snippet", "") or ""),
            timestamp=str(row.get("publishedDate", "") or row.get("published", "") or row.get("date", "") or ""),
            source=provider,
            rank=i,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        if hit is not None:
            out.append(hit)
    return out, ""


def _dedupe_hits(hits: list[WebSearchHit], max_results: int) -> list[WebSearchHit]:
    seen: set[str] = set()
    out: list[WebSearchHit] = []
    for h in hits:
        norm = str(h.url or "").strip().lower().rstrip("/")
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(h)
        if len(out) >= max_results:
            break
    return out


def _sorted_hits(hits: list[WebSearchHit]) -> list[WebSearchHit]:
    # Deterministic ordering: score desc, rank asc, domain asc, url asc.
    return sorted(hits, key=lambda h: (-float(h.score or 0.0), int(h.rank), _web_domain(h.url), h.url))


def _run_web_search(query: str) -> tuple[list[WebSearchHit], str, str]:
    q = re.sub(r"\s+", " ", str(query or "").strip())
    if not q:
        return [], "", "no query provided"
    if not bool(_web_cfg("web_search.enabled", False)):
        return [], "", "disabled in config"

    provider = str(_web_cfg("web_search.provider", "ddg_lite") or "ddg_lite").strip().lower()
    max_results = max(1, int(_web_cfg("web_search.max_results", 5)))
    timeout_sec = float(_web_cfg("web_search.timeout_sec", 8))
    safe_search = bool(_web_cfg("web_search.safe_search", True))
    include_domains = _web_cfg_list("web_search.include_domains")
    exclude_domains = _web_cfg_list("web_search.exclude_domains")
    recency_days = int(_web_cfg("web_search.recency_days", 0))

    if provider == "ddg_lite":
        hits, parse_err = _web_search_ddg_lite(
            q,
            timeout_sec=timeout_sec,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        if parse_err:
            return [], provider, f"parse failed ({provider}): {parse_err}"
    elif provider == "tavily":
        hits, parse_err = _web_search_tavily(
            q,
            timeout_sec=timeout_sec,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            recency_days=recency_days,
        )
        if parse_err:
            return [], provider, f"error ({provider}): {parse_err}"
    elif provider in {"searxng", "custom"}:
        base_url = str(_web_cfg(f"web_search.{provider}.base_url", "") or "").strip()
        if not base_url:
            base_url = str(_web_cfg("web_search.base_url", "") or "").strip()
        hits, parse_err = _web_search_json_provider(
            provider=provider,
            base_url=base_url,
            query=q,
            timeout_sec=timeout_sec,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            safe_search=safe_search,
            recency_days=recency_days,
        )
        if parse_err:
            return [], provider, f"error ({provider}): {parse_err}"
    else:
        return [], provider, f"unsupported provider: {provider}"

    return _sorted_hits(_dedupe_hits(hits, max_results=max_results)), provider, ""


def _relevant_hits(query: str, hits: list[WebSearchHit]) -> list[WebSearchHit]:
    scored: list[WebSearchHit] = []
    for h in hits:
        h.score = _score_hit(query, h)
        if _relevance_passes(h.score):
            scored.append(h)
    return _sorted_hits(scored)


def resolve_web_evidence(query: str) -> Tuple[bool, Dict[str, Any], str]:
    """Return bounded grounded web evidence rows for automatic cascade use."""
    q = re.sub(r"\s+", " ", str(query or "").strip())
    if not q:
        return False, {}, "no query provided"
    try:
        hits, provider, err = _run_web_search(q)
        if err:
            return False, {}, err
        filtered = _relevant_hits(q, hits)
        if not filtered:
            return False, {}, "no relevant results"
        keep_n = int(_web_cfg("web_search.auto_top_n", 3) or 3)
        if keep_n < 1:
            keep_n = 1
        if keep_n > 5:
            keep_n = 5
        kept = filtered[:keep_n]
        rows: list[dict[str, Any]] = []
        for h in kept:
            rows.append(
                {
                    "title": str(h.title or "").strip(),
                    "url": str(h.url or "").strip(),
                    "snippet": str(h.snippet or "").strip(),
                    "score": float(h.score or 0.0),
                }
            )
        top = kept[0]
        evidence = {
            "query": q,
            "provider": provider,
            "result_count": int(len(kept)),
            "result_count_total": int(len(filtered)),
            "rows": rows,
            "title": str(top.title or "").strip(),
            "url": str(top.url or "").strip(),
            "snippet": str(top.snippet or "").strip(),
            "score": float(top.score or 0.0),
            "relevance_gate": "pass",
            "threshold": float(_WEB_RELEVANCE_THRESHOLD),
        }
        return True, evidence, ""
    except requests.exceptions.Timeout:
        return False, {}, "request timeout"
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "unknown"
        return False, {}, f"http error: {code}"
    except Exception as e:
        return False, {}, f"error: {e}"


def handle_web_query(query: str) -> str:
    """
    Provider-agnostic deterministic web lookup.
    """
    q = re.sub(r"\s+", " ", str(query or "").strip())
    if not q:
        return "[web] No query provided"
    try:
        hits, provider, err = _run_web_search(q)
        if err:
            return f"[web] {err}"
        filtered = _relevant_hits(q, hits)
        if not filtered:
            return f"[web] no relevant results for: {q}"
        lines = [f"[web] {len(filtered)} relevant result(s) for: {q}"]
        for i, h in enumerate(filtered, start=1):
            lines.append(f"{i}. {h.title}")
            lines.append(f"   {h.url}")
            if h.snippet:
                lines.append(f"   {h.snippet}")
            lines.append(f"   score={h.score:.2f} provider={provider}")
        return "\n".join(lines)
    except requests.exceptions.Timeout:
        return "[web] Request timeout"
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "unknown"
        return f"[web] HTTP error: {code}"
    except Exception as e:
        return f"[web] Error: {e}"


# ============================================================================
# Currency Exchange (>>exchange)
# ============================================================================


def _normalize_currency_token(token: str) -> str:
    """Map common currency names to ISO 4217 codes."""
    if not token:
        return ""
    t = token.upper()
    aliases = {
        "YEN": "JPY",
        "EURO": "EUR",
        "EUROS": "EUR",
        "POUND": "GBP",
        "POUNDS": "GBP",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
    }
    return aliases.get(t, t)


def _parse_exchange_query(text: str) -> Optional[Tuple[float, str, str]]:
    """
    Parse exchange query: "1 USD to EUR" → (1.0, 'USD', 'EUR')
    Handles: "10 aud to jpy", "usd to eur", "convert aud to jpy"
    """
    raw = text.strip()
    lo = raw.lower()

    # Quick filter
    if not any(k in lo for k in [" to ", " in ", "exchange rate", "convert "]):
        return None

    # Pattern with amount: "10 usd to eur"
    m = re.search(
        r"(?i)\b(\d+(?:\.\d+)?)\s*([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\s+"
        r"(?:to|in)\s+([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\b",
        raw,
    )
    if m:
        amount = float(m.group(1))
        from_ccy = _normalize_currency_token(m.group(2))
        to_ccy = _normalize_currency_token(m.group(3))
        return amount, from_ccy, to_ccy

    # Pattern without amount: "usd to eur"
    m = re.search(
        r"(?i)\b([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\s+"
        r"(?:to|in)\s+([A-Z]{3}|[a-z]{3}|yen|euro|euros|pound|pounds|dollar|dollars)\b",
        raw,
    )
    if m:
        amount = 1.0
        from_ccy = _normalize_currency_token(m.group(1))
        to_ccy = _normalize_currency_token(m.group(2))
        return amount, from_ccy, to_ccy

    return None


def handle_exchange_query(query: str) -> str:
    """
    Fetch currency exchange rate via Frankfurter API.
    
    Example: >>exchange 1 USD to EUR
    Returns: "[exchange] 1.0 USD = 0.92 EUR"
    """
    parsed = _parse_exchange_query(query)
    if not parsed:
        return "[exchange] Not a currency query (e.g. '1 USD to EUR')"
    
    amount, from_ccy, to_ccy = parsed

    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_ccy}&to={to_ccy}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        rate = data.get("rates", {}).get(to_ccy)
        if rate:
            return f"[exchange] {amount} {from_ccy} = {rate:.2f} {to_ccy}"
        return f"[exchange] Currency pair {from_ccy}/{to_ccy} not supported"
    
    except requests.exceptions.Timeout:
        return "[exchange] Request timeout"
    except requests.exceptions.HTTPError:
        return f"[exchange] Cannot convert {from_ccy} to {to_ccy}"
    except Exception as e:
        return f"[exchange] Error: {e}"


# ============================================================================
# Weather (>>weather) – Open-Meteo API
# ============================================================================

def _decode_weather_code(code: int) -> str:
    """Decode WMO weather code to human-readable description."""
    # WMO Weather interpretation codes (simplified)
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with hail",
    }
    return codes.get(code, f"Weather code {code}")


def handle_weather_query(location: str) -> str:
    """
    Fetch current weather via Open-Meteo API with Nominatim geocoding (OSM).
    
    Uses Nominatim (OpenStreetMap) for geocoding (better regional coverage),
    then Open-Meteo for weather forecast.
    Example: >>weather Carnarvon Western Australia
    Returns: "[weather] Carnarvon, Western Australia: 22°C, Partly cloudy, 65% humidity"
    """
    location = (location or "").strip()
    if not location:
        return "[weather] No location provided"

    try:
        # Step 1: Geocode via Nominatim (OpenStreetMap) – better regional coverage
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_params = {
            "q": location,
            "format": "json",
            "limit": 1,
            "language": "en"
        }
        
        # Add User-Agent (Nominatim requires it)
        headers = {"User-Agent": "llama-conductor/1.0.2"}
        geo_resp = requests.get(geo_url, params=geo_params, headers=headers, timeout=5)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if not geo_data:
            return f"[weather] Location '{location}' not found"
        
        place = geo_data[0]
        latitude = float(place.get("lat"))
        longitude = float(place.get("lon"))
        display_name = place.get("display_name", location)
        
        # Step 2: Fetch current weather via Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m",
            "temperature_unit": "celsius"
        }
        
        weather_resp = requests.get(weather_url, params=weather_params, timeout=5)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        humidity = current.get("relative_humidity_2m")
        
        condition = _decode_weather_code(code)
        
        return f"[weather] {display_name}: {temp}°C, {condition}, {humidity}% humidity"
    
    except requests.exceptions.Timeout:
        return "[weather] Request timeout"
    except requests.exceptions.HTTPError:
        return f"[weather] Error fetching weather for '{location}'"
    except Exception as e:
        return f"[weather] Error: {e}"


# ============================================================================
# Testing / Standalone Usage
# ============================================================================

if __name__ == "__main__":
    # Test calc
    print("=== Testing >>calc ===")
    result = parse_and_eval_calc("30% of 79.95")
    print(f"30% of 79.95: {format_calc_result(result)}")

    result = parse_and_eval_calc("14*365")
    print(f"14*365: {format_calc_result(result)}")

    # Test wiki
    print("\n=== Testing >>wiki ===")
    print(handle_wiki_query("Albert Einstein"))

    # Test exchange
    print("\n=== Testing >>exchange ===")
    print(handle_exchange_query("1 USD to EUR"))

    # Test weather
    print("\n=== Testing >>weather ===")
    print(handle_weather_query("Perth"))

