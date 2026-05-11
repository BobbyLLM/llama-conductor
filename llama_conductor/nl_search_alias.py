from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

from .model_calls import call_model_prompt

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None  # type: ignore

__all__ = ["is_bare_nl_search_alias", "resolve_nl_search_query", "summarize_nl_search_alias_response"]

_BARE_NL_SEARCH_ALIAS_PHRASES = {
    "do a search",
    "search for it",
    "search for that",
    "search for this",
    "search for them",
    "search for those",
    "search for him",
    "search for her",
    "search for one",
    "look it up",
    "look that up",
    "look this up",
    "look them up",
    "look those up",
    "go check",
    "go check that",
    "verify that",
}

_SUBSTANTIVE_MIN_WORDS = 5
_LOW_SALIENCE_PHRASES = {
    "wii",
    "wii u",
    "wii-u",
    "pc",
    "ps3",
    "ps4",
    "xbox",
    "xbox one",
    "xbox 360",
    "playstation",
    "playstation 3",
    "playstation 4",
    "version",
    "release",
    "game",
    "games",
    "platform",
    "search",
    "search result",
    "search results",
    "query",
    "answer",
    "topic",
    "question",
}
_GENERIC_TOPIC_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "same",
    "she",
    "so",
    "some",
    "story",
    "that",
    "the",
    "there",
    "then",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "nope",
    "never",
    "got",
    "came",
    "out",
    "deal",
    "either",
    "ever",
    "period",
    "version",
    "release",
    "search",
    "query",
    "answer",
    "wii",
    "u",
    "pc",
    "ps3",
    "ps4",
    "xbox",
    "one",
    "playstation",
    "game",
    "games",
    "platform",
}
_STOP_RUN_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "same",
    "she",
    "so",
    "some",
    "story",
    "that",
    "the",
    "there",
    "then",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "nope",
    "never",
    "got",
    "came",
    "out",
    "deal",
    "either",
    "ever",
    "period",
    "version",
    "release",
    "search",
    "query",
    "answer",
    "game",
    "games",
    "platform",
    "wii",
    "u",
    "pc",
    "ps3",
    "ps4",
    "xbox",
    "one",
    "playstation",
}
_SHORT_TOPIC_TOKENS = {"ex"}
_LEADING_FILLERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "just",
    "really",
    "very",
    "more",
    "most",
    "not",
    "nope",
    "no",
    "never",
    "still",
    "only",
}
_TRAILING_FILLERS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "about",
    "period",
    "either",
    "ever",
}


def _strip_footer_lines(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        if line.lstrip().lower().startswith(("confidence:", "source:", "sources:", "profile:")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _recent_texts(values: Any) -> Iterable[str]:
    items = list(values or [])
    for item in reversed(items):
        text = _strip_footer_lines(str(item or "").strip())
        if text:
            yield text


def is_bare_nl_search_alias(text: str) -> bool:
    tail = str(text or "").strip().lower()
    if not tail:
        return False
    for sep in (".", "!", "?"):
        if sep in tail:
            tail = tail.rsplit(sep, 1)[-1].strip()
    tail = " ".join(tail.split())
    return tail in _BARE_NL_SEARCH_ALIAS_PHRASES


def _tokenize(text: str) -> list[str]:
    cleaned = (
        str(text or "")
        .replace("\u2014", " ")
        .replace("\u2013", " ")
        .replace("\u2012", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("{", " ")
        .replace("}", " ")
        .replace('"', " ")
        .replace("`", " ")
    )
    return [tok for tok in cleaned.split() if tok]


def _substantive_word_count(text: str) -> int:
    return len(_tokenize(text))


@lru_cache(maxsize=3)
def _load_spacy_model(name: str):
    return spacy.load(name, disable=("textcat", "lemmatizer"))  # type: ignore[union-attr]


def _normalize_tokens(tokens: list[str]) -> list[str]:
    while tokens and tokens[0].lower() in _LEADING_FILLERS:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in _TRAILING_FILLERS:
        tokens.pop()
    return tokens


def _phrase_score(tokens: list[str]) -> int:
    if not tokens:
        return -10_000
    lowered = [tok.lower() for tok in tokens]
    phrase = " ".join(lowered)
    if phrase in _LOW_SALIENCE_PHRASES and not _looks_like_named_hint(tokens):
        return -10_000
    generic_count = sum(1 for tok in lowered if tok in _GENERIC_TOPIC_TOKENS)
    if generic_count >= max(2, len(tokens) - 1):
        return -10_000
    if generic_count / max(1, len(tokens)) >= 0.6:
        return -10_000
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return -1_000
    score = len(tokens) * 10
    if any(tok[:1].isupper() for tok in tokens):
        score += 8
    if any(len(tok) >= 5 for tok in tokens):
        score += 5
    if any(tok.isupper() and len(tok) >= 2 for tok in tokens):
        score += 4
    if any(tok.lower() in {"search", "version", "release", "game", "games", "wii", "pc"} for tok in tokens):
        score -= 12
    return score


def _looks_like_named_hint(tokens: list[str]) -> bool:
    if not tokens:
        return False
    lowered = [tok.lower() for tok in tokens]
    if any(tok in _HINT_ALWAYS_TOKENS for tok in lowered):
        return True
    if any(tok[:1].isupper() for tok in tokens):
        return True
    if any(tok.isupper() and len(tok) >= 2 for tok in tokens):
        return True
    if any(any(ch.isdigit() for ch in tok) for tok in tokens):
        return True
    return False


def _is_topic_token(token: str) -> bool:
    low = token.lower()
    if not token:
        return False
    if low in _STOP_RUN_WORDS and low not in _SHORT_TOPIC_TOKENS:
        return False
    if token.isupper() and len(token) >= 2:
        return True
    if token[:1].isupper() and len(token) >= 2:
        return True
    if any(ch.isdigit() for ch in token):
        return True
    return len(token) >= 4 and low not in _GENERIC_TOPIC_TOKENS


def _runs_from_tokens(tokens: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if _is_topic_token(token):
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _extract_topic_with_spacy(text: str) -> Optional[str]:
    if spacy is None:
        return None
    nlp = None
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            nlp = _load_spacy_model(name)
            break
        except Exception:
            continue
    if nlp is None:
        return None
    try:
        doc = nlp(text)
    except Exception:
        return None

    candidates: list[list[str]] = []
    try:
        for ent in getattr(doc, "ents", []) or []:
            tokens = _normalize_tokens(_tokenize(getattr(ent, "text", "")))
            if tokens:
                candidates.append(tokens)
    except Exception:
        pass
    try:
        for chunk in getattr(doc, "noun_chunks", []) or []:
            tokens = _normalize_tokens(_tokenize(getattr(chunk, "text", "")))
            if tokens:
                candidates.append(tokens)
    except Exception:
        pass

    best: Optional[list[str]] = None
    best_score = -10_000
    for cand in candidates:
        score = _phrase_score(cand)
        if score > best_score:
            best = cand
            best_score = score
    if not best or best_score < 15:
        return None
    return " ".join(best)


def _extract_topic_fallback(text: str) -> Optional[str]:
    tokens = _tokenize(text)
    if not tokens:
        return None

    candidates: list[list[str]] = []
    for run in _runs_from_tokens(tokens):
        run = _normalize_tokens(run[:])
        if run:
            candidates.append(run)

    if not candidates:
        content = [tok for tok in tokens if tok.lower() not in _GENERIC_TOPIC_TOKENS]
        content = _normalize_tokens(content[:])
        if content:
            candidates.append(content)

    best: Optional[list[str]] = None
    best_score = -10_000
    for cand in candidates:
        score = _phrase_score(cand)
        if score > best_score:
            best = cand
            best_score = score
    if not best or best_score < 15:
        return None
    return " ".join(best)


def _extract_topic(text: str) -> Optional[str]:
    spacy_topic = _extract_topic_with_spacy(text)
    fallback_topic = _extract_topic_fallback(text)
    candidates = [cand for cand in (spacy_topic, fallback_topic) if cand]
    if not candidates:
        return None
    best = None
    best_score = -10_000
    for cand in candidates:
        score = _phrase_score(_tokenize(cand))
        if score > best_score:
            best = cand
            best_score = score
    return best if best and best_score >= 15 else None


_HINT_ALWAYS_TOKENS = {
    "wii",
    "u",
    "wiiu",
    "xbox",
    "playstation",
    "pc",
    "ps3",
    "ps4",
    "switch",
    "xboxone",
    "xbox360",
}


def _hint_score(tokens: list[str]) -> int:
    if not tokens:
        return -10_000
    lowered = [tok.lower() for tok in tokens]
    if any(tok in {"do", "search", "look", "go", "check", "verify"} for tok in lowered):
        return -10_000
    if any(tok in _GENERIC_TOPIC_TOKENS for tok in lowered) and not any(tok in _HINT_ALWAYS_TOKENS for tok in lowered):
        return -10_000
    if " ".join(lowered) in _LOW_SALIENCE_PHRASES and not _looks_like_named_hint(tokens):
        return -10_000
    score = len(tokens) * 8
    if any(tok in _HINT_ALWAYS_TOKENS for tok in lowered):
        score += 16
    if any(tok[:1].isupper() for tok in tokens):
        score += 4
    return score


def _is_hint_token(token: str) -> bool:
    low = token.lower()
    if not token:
        return False
    if low in {"do", "search", "look", "go", "check", "verify"}:
        return False
    if low in _LEADING_FILLERS or low in _TRAILING_FILLERS:
        return False
    if low in _GENERIC_TOPIC_TOKENS and low not in _HINT_ALWAYS_TOKENS:
        return False
    if low in _HINT_ALWAYS_TOKENS:
        return True
    if token.isupper() and len(token) >= 2:
        return True
    if token[:1].isupper() and len(token) >= 2:
        return True
    if any(ch.isdigit() for ch in token):
        return True
    return False


def _hint_runs_from_tokens(tokens: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if _is_hint_token(token):
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _extract_hint_with_spacy(text: str, primary_query: str = "") -> Optional[str]:
    if spacy is None:
        return None
    nlp = None
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            nlp = _load_spacy_model(name)
            break
        except Exception:
            continue
    if nlp is None:
        return None
    try:
        doc = nlp(text)
    except Exception:
        return None

    candidates: list[list[str]] = []
    try:
        for ent in getattr(doc, "ents", []) or []:
            tokens = _normalize_tokens(_tokenize(getattr(ent, "text", "")))
            if tokens and _looks_like_named_hint(tokens):
                candidates.append(tokens)
    except Exception:
        pass
    try:
        for chunk in getattr(doc, "noun_chunks", []) or []:
            tokens = _normalize_tokens(_tokenize(getattr(chunk, "text", "")))
            if tokens and _looks_like_named_hint(tokens):
                candidates.append(tokens)
    except Exception:
        pass

    best: Optional[list[str]] = None
    best_score = -10_000
    primary_low = str(primary_query or "").lower()
    for cand in candidates:
        phrase = " ".join(cand).lower()
        if primary_low and phrase and phrase in primary_low:
            continue
        score = _hint_score(cand)
        if score > best_score:
            best = cand
            best_score = score
    return " ".join(best) if best and best_score > 0 else None


def _extract_hint_fallback(text: str, primary_query: str = "") -> Optional[str]:
    tokens = _tokenize(text)
    if not tokens:
        return None
    primary_low = str(primary_query or "").lower()
    candidates: list[list[str]] = []
    for run in _hint_runs_from_tokens(tokens):
        run = _normalize_tokens(run[:])
        if 1 <= len(run) <= 4 and _looks_like_named_hint(run):
            candidates.append(run)
    best: Optional[list[str]] = None
    best_score = -10_000
    for cand in candidates:
        phrase = " ".join(cand).lower()
        if primary_low and phrase and phrase in primary_low:
            continue
        score = _hint_score(cand)
        if score > best_score:
            best = cand
            best_score = score
    return " ".join(best) if best and best_score > 0 else None


def _extract_hint(text: str, primary_query: str = "") -> Optional[str]:
    spacy_hint = _extract_hint_with_spacy(text, primary_query=primary_query)
    fallback_hint = _extract_hint_fallback(text, primary_query=primary_query)
    candidates = [cand for cand in (spacy_hint, fallback_hint) if cand]
    if not candidates:
        return None
    best = None
    best_score = -10_000
    for cand in candidates:
        score = _hint_score(_tokenize(cand))
        if score > best_score:
            best = cand
            best_score = score
    return best if best and best_score > 0 else None


def _display_phrase(text: str) -> str:
    parts: list[str] = []
    for tok in _tokenize(text):
        low = tok.lower()
        if low == "wii":
            parts.append("Wii")
        elif low == "u":
            parts.append("U")
        elif low == "pc":
            parts.append("PC")
        elif low == "xbox":
            parts.append("Xbox")
        elif low == "playstation":
            parts.append("PlayStation")
        else:
            parts.append(tok)
    return " ".join(parts).strip()


def _extract_recent_user_chain_query(user_turns: Iterable[str]) -> Optional[str]:
    substantive: list[str] = []
    for text in _recent_texts(user_turns):
        if is_bare_nl_search_alias(text):
            continue
        if _substantive_word_count(text) < _SUBSTANTIVE_MIN_WORDS:
            continue
        substantive.append(text)
        if len(substantive) >= 2:
            break

    if len(substantive) < 2:
        return None

    latest_hint = _extract_hint(substantive[0], primary_query="")
    older_topic = _extract_topic(substantive[1])
    if not latest_hint or not older_topic:
        return None
    if not _looks_like_named_hint(_tokenize(latest_hint)):
        return None
    return f"{older_topic} {_display_phrase(latest_hint)}".strip()


def resolve_nl_search_query(state) -> str | None:
    """
    Given session state, extract the most recent substantive topic
    from conversation history and return it as a web search query.
    Returns None if no topic can be extracted.
    """
    if state is None:
        return None

    assistant_bodies = getattr(state, "kaioken_recent_assistant_bodies", []) or []
    user_turns = getattr(state, "kaioken_recent_user_turns", []) or []
    user_chain_query = _extract_recent_user_chain_query(user_turns)

    primary = None
    for text in _recent_texts(assistant_bodies):
        if _substantive_word_count(text) < _SUBSTANTIVE_MIN_WORDS:
            continue
        primary = _extract_topic(text)
        if primary:
            break

    if user_chain_query:
        if not primary:
            return user_chain_query
        if user_chain_query.lower() not in primary.lower():
            return user_chain_query

    if primary:
        for text in _recent_texts(user_turns):
            if is_bare_nl_search_alias(text):
                continue
            if _substantive_word_count(text) < _SUBSTANTIVE_MIN_WORDS:
                continue
            hint = _extract_hint(text, primary_query=primary)
            if hint and hint.lower() not in primary.lower():
                return f"{primary} {_display_phrase(hint)}".strip()
            break
        return primary

    for text in _recent_texts(user_turns):
        if is_bare_nl_search_alias(text):
            continue
        if _substantive_word_count(text) < _SUBSTANTIVE_MIN_WORDS:
            continue
        topic = _extract_topic(text)
        if topic:
            return topic

    return None


def _web_rows_from_state(state: Any) -> list[dict[str, Any]]:
    rows = getattr(state, "last_web_results", []) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            url = str(row.get("url", "") or "").strip()
            title = str(row.get("title", "") or "").strip()
            snippet = str(row.get("snippet", "") or "").strip()
            if url and (title or snippet):
                out.append({"url": url, "title": title, "snippet": snippet})
    return out


def summarize_nl_search_alias_response(*, state: Any, query: str, raw_reply: str) -> str:
    rows = _web_rows_from_state(state)
    if len(rows) < 2:
        return raw_reply

    facts: list[str] = []
    urls: list[str] = []
    for idx, row in enumerate(rows[:5], start=1):
        title = str(row.get("title", "") or "").strip()
        snippet = str(row.get("snippet", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        if not (title or snippet):
            continue
        facts.append(f"[{idx}] {title}".strip())
        if snippet:
            facts.append(f"    {snippet}")
        if url:
            facts.append(f"    URL: {url}")
            if url.startswith("http://") or url.startswith("https://"):
                urls.append(url)

    if len(facts) < 2:
        return raw_reply

    prompt = (
        "Use only the web facts below. Write 2-4 concise sentences that summarize what the evidence says.\n"
        "Do not invent details. Do not use prior knowledge. If the evidence is thin, keep the answer brief and factual.\n\n"
        f"Question: {query}\n\n"
        f"Web facts:\n{chr(10).join(facts)}\n\n"
        "Respond with prose only."
    )
    raw = str(
        call_model_prompt(
            role="thinker",
            prompt=prompt,
            max_tokens=220,
            temperature=0.2,
            top_p=0.9,
            debug_context="nl_search_alias_summary",
        )
        or ""
    ).strip()
    if not raw:
        return raw_reply
    if raw.lower().startswith("[model '") or raw.lower().startswith("[router error"):
        return raw_reply
    raw = _strip_footer_lines(raw).strip()
    if not raw:
        return raw_reply
    dedup_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        dedup_urls.append(url)
    if dedup_urls:
        raw = f"{raw}\n\n" + "\n".join(f"See: {u}" for u in dedup_urls)
    return f"{raw}\n\nConfidence: medium | Source: Web"




