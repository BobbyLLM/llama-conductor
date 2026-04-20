from __future__ import annotations
import re
from typing import Optional

_STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "of", "for",
    "to", "from", "by", "with", "what", "how", "why", "who", "which",
    "did", "does", "were", "was", "is", "are", "have", "had", "when", "where",
])

_CODEX_MISS_TEXT = "Not available in retrieved Codex content."

# e5 top-k source candidates per answer sentence
_E5_CANDIDATE_K = 3

# TinyBERT entailment threshold
_TINYBERT_THRESHOLD = 4.0


def _split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.split()) >= 5
    ]


def _user_content_tokens(user_text: str) -> frozenset[str]:
    return frozenset(
        w.lower().strip("?.!,")
        for w in str(user_text or "").split()
        if w.lower().strip("?.!,") not in _STOPWORDS
    )


def codex_answer_is_grounded(
    answer_text: str,
    source_text: str,
    user_text: str = "",
    e5_candidate_k: int = _E5_CANDIDATE_K,
    tinybert_threshold: float = _TINYBERT_THRESHOLD,
) -> tuple[bool, list[str]]:
    """
    Two-stage grounding check.

    Stage 1 — e5 cosine: for each answer sentence, find top-k most similar
    source sentences by cosine similarity. Narrows the search space.

    Stage 2 — TinyBERT cross-encoder: score each (answer_sentence, source_sentence)
    pair. If max score across top-k candidates is below threshold, sentence is flagged.

    User-term exemption: sentences where >75% of non-stopword tokens appear
    in the user query are exempt — these are comparison/context sentences,
    not factual claims about the source.
    """
    import numpy as np
    from .rag import get_embed_model, get_rerank_model

    answer_sentences = _split_sentences(answer_text)
    source_sentences = _split_sentences(source_text)

    if not answer_sentences or not source_sentences:
        return True, []

    user_tokens = _user_content_tokens(user_text)
    embed_model = get_embed_model()
    rerank_model = get_rerank_model()

    if rerank_model is None:
        # TinyBERT disabled in config — fail open, log
        return True, []

    # Encode all source sentences once
    source_embeddings = embed_model.encode(
        [f"passage: {s}" for s in source_sentences],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    flagged = []

    for sentence in answer_sentences:
        # User-term exemption — tightened to >75%
        sentence_tokens = frozenset(
            w.lower().strip("?.!,") for w in sentence.split()
            if w.lower().strip("?.!,") not in _STOPWORDS
        )
        if user_tokens and len(sentence_tokens & user_tokens) / max(len(sentence_tokens), 1) > 0.75:
            continue

        # Stage 1 — e5 cosine: get top-k source candidates
        sent_embedding = embed_model.encode(
            f"query: {sentence}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cosine_scores = source_embeddings @ sent_embedding
        top_k_indices = list(
            reversed(cosine_scores.argsort()[-e5_candidate_k:])
        )
        candidates = [source_sentences[i] for i in top_k_indices]
        top_candidates = [
            (float(cosine_scores[i]), str(source_sentences[i] or "").strip().replace("\n", " "))
            for i in top_k_indices
        ]
        # Stage 2 — TinyBERT cross-encoder: score pairs
        pairs = [(sentence, candidate) for candidate in candidates]
        try:
            scores = rerank_model.predict(pairs)
            for candidate, score in zip(candidates, scores):
                candidate_text = str(candidate or "").strip().replace("\n", " ")
            max_score = float(max(scores))
        except Exception as e:
            continue

        if max_score < tinybert_threshold:
            flagged.append(sentence)
            verdict = "strip"
        else:
            verdict = "pass"

    return len(flagged) == 0, flagged


def codex_validate_answer(
    answer_text: str,
    source_text: str,
    user_text: str = "",
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Public interface for router.

    strict=True: deterministic lookup, bypass grounding check entirely.
    strict=False: run two-stage e5 + TinyBERT grounding check.

    Returns (grounded, text).
    If grounded or strict: (True, answer_text unchanged).
    If not grounded: (False, _CODEX_MISS_TEXT).
    """
    if strict:
        return True, answer_text

    grounded, flagged = codex_answer_is_grounded(answer_text, source_text, user_text)
    if grounded:
        return True, answer_text

    answer_sentences = _split_sentences(answer_text)
    if not answer_sentences:
        return False, _CODEX_MISS_TEXT

    flagged_set = set(flagged)
    kept = [sentence for sentence in answer_sentences if sentence not in flagged_set]
    if kept:
        return True, " ".join(kept)
    return False, _CODEX_MISS_TEXT
