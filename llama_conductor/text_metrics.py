from __future__ import annotations

import re
from typing import List

WORD_RE = re.compile(r"[A-Za-z0-9_']+")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "to",
    "of",
    "for",
    "with",
    "about",
    "on",
    "in",
    "at",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "you",
    "me",
    "my",
    "your",
    "i",
    "we",
    "they",
    "he",
    "she",
    "them",
    "us",
    "do",
    "did",
    "does",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can",
    "could",
    "should",
    "would",
    "please",
    "hey",
    "heya",
    "sup",
    "yo",
    "hi",
    "hello",
    "lol",
    "ok",
    "okay",
}


def tokens(text: str) -> List[str]:
    return [tok.lower() for tok in WORD_RE.findall(str(text or ""))]


def content_word_count(text: str) -> int:
    return sum(1 for tok in tokens(text) if tok not in STOPWORDS)
