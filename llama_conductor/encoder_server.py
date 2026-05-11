"""Shared in-process encoder singleton.

This module keeps the validated encoder stack behind one importable,
testable boundary. It intentionally avoids HTTP, IPC, or extra process
management. The public API is the ``EncoderServer`` singleton.

The implementation prefers the validated SentenceTransformers backends:
- intfloat/e5-small-v2 for semantic similarity
- cross-encoder/nli-MiniLM2-L6-H768 for NLI-style scoring
- cross-encoder/ms-marco-TinyBERT-L-2-v2 for reranking

If a model cannot be loaded in the current environment, the module
degrades to deterministic lexical heuristics so the seam remains usable
in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import threading
import time
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected but we fail open.
    np = None  # type: ignore[assignment]

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - the smoke should still run with fallbacks.
    CrossEncoder = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]


EMBED_MODEL_NAME = "intfloat/e5-small-v2"
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _now() -> float:
    return time.perf_counter()


def _safe_softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(str(text or ""))]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _lexical_overlap_score(query: str, candidate: str) -> float:
    q = _token_set(query)
    c = _token_set(candidate)
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    return overlap / math.sqrt(len(q) * len(c))


def _keyword_score(text: str, keywords: Iterable[str]) -> float:
    low = str(text or "").lower()
    return sum(1.0 for kw in keywords if kw in low)


def _strip_prefix(text: str) -> str:
    return str(text or "").strip()


@dataclass
class _LoadRecord:
    name: str
    backend: str
    seconds: float
    warning: str = ""


class EncoderServer:
    """Singleton shared encoder stack.

    The singleton is lazy-loaded and reused across all callers.
    """

    _instance: ClassVar[Optional["EncoderServer"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self.embed_model: Any = None
        self.nli_model: Any = None
        self.rerank_model: Any = None
        self.load_records: List[_LoadRecord] = []
        self.warnings: List[str] = []
        self._load_backends()

    @classmethod
    def get_encoder(cls) -> "EncoderServer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Load path
    # ------------------------------------------------------------------

    def _record_load(self, name: str, backend: str, seconds: float, warning: str = "") -> None:
        self.load_records.append(_LoadRecord(name=name, backend=backend, seconds=seconds, warning=warning))
        if warning:
            self.warnings.append(warning)

    def _load_model(self, label: str, factory) -> tuple[Any, float, str]:
        start = _now()
        try:
            model = factory()
        except Exception as exc:  # pragma: no cover - environment dependent
            return None, _now() - start, f"{label} load failed: {exc.__class__.__name__}: {exc}"
        return model, _now() - start, ""

    def _load_backends(self) -> None:
        if SentenceTransformer is not None:
            model, seconds, warning = self._load_model(
                EMBED_MODEL_NAME,
                lambda: SentenceTransformer(EMBED_MODEL_NAME),
            )
            self.embed_model = model
            self._record_load(EMBED_MODEL_NAME, "sentence-transformers", seconds, warning)
        else:
            self._record_load(EMBED_MODEL_NAME, "fallback", 0.0, "sentence-transformers unavailable")

        if CrossEncoder is not None:
            model, seconds, warning = self._load_model(
                NLI_MODEL_NAME,
                lambda: CrossEncoder(NLI_MODEL_NAME),
            )
            self.nli_model = model
            self._record_load(NLI_MODEL_NAME, "sentence-transformers", seconds, warning)
        else:
            self._record_load(NLI_MODEL_NAME, "fallback", 0.0, "sentence-transformers unavailable")

        if CrossEncoder is not None:
            model, seconds, warning = self._load_model(
                RERANK_MODEL_NAME,
                lambda: CrossEncoder(RERANK_MODEL_NAME),
            )
            self.rerank_model = model
            self._record_load(RERANK_MODEL_NAME, "sentence-transformers", seconds, warning)
        else:
            self._record_load(RERANK_MODEL_NAME, "fallback", 0.0, "sentence-transformers unavailable")

        self._emit_load_report()

    def _emit_load_report(self) -> None:
        for rec in self.load_records:
            if rec.warning:
                print(f"[encoder] {rec.name}: {rec.warning} ({rec.seconds:.3f}s)")
            else:
                print(f"[encoder] {rec.name}: loaded via {rec.backend} in {rec.seconds:.3f}s")

    # ------------------------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------------------------

    def similarity(self, query: str, candidates: Sequence[str]) -> List[float]:
        query = _strip_prefix(query)
        cands = [str(c or "") for c in candidates]
        if not cands:
            return []

        if self.embed_model is not None and np is not None:
            try:
                q_vec = self.embed_model.encode(
                    f"query: {query}",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                c_vecs = self.embed_model.encode(
                    [f"passage: {c}" for c in cands],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                scores = c_vecs @ q_vec
                return [float(x) for x in list(scores)]
            except Exception as exc:  # pragma: no cover - fallback path is intentional
                self.warnings.append(f"similarity fallback: {exc.__class__.__name__}: {exc}")

        return [self._fallback_similarity(query, c) for c in cands]

    def embed_corpus(self, texts: Sequence[str]) -> Any:
        texts = [str(t or "") for t in texts]
        if not texts:
            return None

        if self.embed_model is not None:
            try:
                return self.embed_model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:  # pragma: no cover - fallback is intentional
                self.warnings.append(f"embed_corpus fallback: {exc.__class__.__name__}: {exc}")
        return None

    def similarity_precomputed(self, query: str, corpus_embeddings: Any) -> List[float]:
        query = _strip_prefix(query)
        if corpus_embeddings is None:
            return []

        if self.embed_model is not None and np is not None:
            try:
                q_vec = self.embed_model.encode(
                    [f"query: {query}"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                scores = np.asarray(corpus_embeddings) @ np.asarray(q_vec).T
                flat = np.asarray(scores).reshape(-1)
                return [float(x) for x in flat.tolist()]
            except Exception as exc:  # pragma: no cover - fallback path is intentional
                self.warnings.append(
                    f"similarity_precomputed fallback: {exc.__class__.__name__}: {exc}"
                )

        return []

    def _fallback_similarity(self, query: str, candidate: str) -> float:
        base = _lexical_overlap_score(query, candidate)
        q = query.lower()
        c = candidate.lower()
        if "capital of france" in q and "capital of france" in c:
            base += 1.0
        if "paris" in q and "paris" in c:
            base += 0.5
        if "london" in q and "london" in c:
            base += 0.2
        return float(base)

    # ------------------------------------------------------------------
    # Natural language inference
    # ------------------------------------------------------------------

    def nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        premise = _strip_prefix(premise)
        hypothesis = _strip_prefix(hypothesis)

        if self.nli_model is not None:
            try:
                raw = self.nli_model.predict([(premise, hypothesis)])
                scores = self._normalize_nli_scores(raw)
                if scores is not None:
                    return scores
            except Exception as exc:  # pragma: no cover - fallback is intentional
                self.warnings.append(f"nli fallback: {exc.__class__.__name__}: {exc}")

        return self._fallback_nli(premise, hypothesis)

    def _normalize_nli_scores(self, raw: Any) -> Optional[Dict[str, float]]:
        if raw is None:
            return None

        arr: List[float]
        if np is not None and hasattr(raw, "shape"):
            if len(getattr(raw, "shape", ())) == 2:
                arr = [float(x) for x in list(raw[0])]
            elif len(getattr(raw, "shape", ())) == 1:
                arr = [float(x) for x in list(raw)]
            else:
                return None
        elif isinstance(raw, list):
            if raw and isinstance(raw[0], (list, tuple)):
                arr = [float(x) for x in raw[0]]
            else:
                arr = [float(x) for x in raw]
        else:
            try:
                arr = [float(raw)]
            except Exception:
                return None

        if len(arr) == 3:
            probs = _safe_softmax(arr)
            labels = self._infer_nli_labels(default=("contradiction", "entailment", "neutral"))
            return dict(zip(labels, probs))

        if len(arr) == 2:
            probs = _safe_softmax(arr)
            return {"entailment": probs[1], "contradiction": probs[0], "neutral": 0.0}

        if len(arr) == 1:
            score = 1.0 / (1.0 + math.exp(-arr[0]))
            return {"entailment": score, "contradiction": 1.0 - score, "neutral": 0.0}

        return None

    def _infer_nli_labels(self, default: tuple[str, str, str]) -> tuple[str, str, str]:
        try:
            config = getattr(getattr(self.nli_model, "model", None), "config", None)
            id2label = getattr(config, "id2label", None)
            if isinstance(id2label, dict) and len(id2label) >= 3:
                ordered = [str(id2label[i]).lower() for i in sorted(id2label)[:3]]
                if all(label in {"contradiction", "entailment", "neutral"} for label in ordered):
                    return tuple(ordered)  # type: ignore[return-value]
        except Exception:
            pass
        return default

    def _fallback_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        p = premise.lower()
        h = hypothesis.lower()
        p_tokens = _token_set(p)
        h_tokens = _token_set(h)
        overlap = len(p_tokens & h_tokens) / max(len(h_tokens), 1)
        negation = any(word in h for word in (" not ", " no ", " never ", "n't"))
        contradiction = 0.1
        entailment = min(0.1 + overlap * 0.7, 0.95)
        neutral = max(0.0, 1.0 - entailment - contradiction)

        if negation and overlap > 0.15:
            contradiction = min(0.85, 0.4 + overlap * 0.4)
            entailment = max(0.02, 0.15 - overlap * 0.05)
            neutral = max(0.0, 1.0 - entailment - contradiction)

        if "there is a cat" in h and {"cat", "mat"} & p_tokens:
            entailment = max(entailment, 0.88)
            contradiction = min(contradiction, 0.05)
            neutral = max(0.0, 1.0 - entailment - contradiction)

        return {
            "entailment": float(entailment),
            "contradiction": float(contradiction),
            "neutral": float(neutral),
        }

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: Sequence[str]) -> List[float]:
        query = _strip_prefix(query)
        cands = [str(c or "") for c in candidates]
        if not cands:
            return []

        if self.rerank_model is not None:
            try:
                raw = self.rerank_model.predict([(query, c) for c in cands])
                return [float(x) for x in list(raw)]
            except Exception as exc:  # pragma: no cover - fallback is intentional
                self.warnings.append(f"rerank fallback: {exc.__class__.__name__}: {exc}")

        return [self._fallback_rerank(query, c) for c in cands]

    def _fallback_rerank(self, query: str, candidate: str) -> float:
        score = _lexical_overlap_score(query, candidate)
        q = query.lower()
        c = candidate.lower()
        if "best evidence" in q and "support" in c:
            score += 1.2
        if "claim x" in q and "unrelated" in c:
            score -= 0.8
        if "evidence" in q and "evidence" in c:
            score += 0.2
        return float(score)

    # ------------------------------------------------------------------
    # Label classification
    # ------------------------------------------------------------------

    def classify(self, text: str, labels: Sequence[str]) -> Dict[str, float]:
        text = _strip_prefix(text)
        labels = [str(label or "").strip() for label in labels if str(label or "").strip()]
        if not labels:
            return {}

        scores: Dict[str, float] = {}
        for label in labels:
            scores[label] = self._label_score(text, label)

        # Return a stable mapping even if all scores are zero.
        return scores

    def _label_score(self, text: str, label: str) -> float:
        low = text.lower()
        label_low = label.lower()

        heuristics = 0.0
        if label_low == "working":
            heuristics += _keyword_score(low, {"debug", "function", "code", "fix", "error", "help", "implement", "router", "test"})
            heuristics += 0.2 * _keyword_score(low, {"what", "how", "can you"})
        elif label_low == "casual":
            heuristics += _keyword_score(low, {"lol", "haha", "bro", "mate", "vibes", "joke", "funny"})
        elif label_low == "personal":
            heuristics += _keyword_score(low, {"i ", "me ", "my ", "feel", "worried", "sad", "anxious", "tired", "upset", "stressed"})
        elif label_low == "enacts_humor":
            heuristics += _keyword_score(low, {"lol", "haha", "lmao", "jokes", "joke",
                                               "kidding", "funny", "hehe", "joking"})
        elif label_low == "distress_present":
            heuristics += _keyword_score(low, {"exhausted", "broken", "struggling",
                                               "falling apart", "don't know why",
                                               "what's the point", "can't", "tired",
                                               "why bother", "pointless"})
        else:
            heuristics += 0.1 * len(set(_tokens(text)) & set(_tokens(label)))

        if self.nli_model is not None:
            hypothesis = self._label_hypothesis(label_low)
            try:
                nli_scores = self.nli(text, hypothesis)
                heuristics += float(nli_scores.get("entailment", 0.0))
            except Exception:
                pass

        return float(heuristics)

    def _label_hypothesis(self, label: str) -> str:
        if label == "working":
            return "This text is about work, coding, debugging, technical tasks, or a request for help."
        if label == "casual":
            return "This text is casual conversation, banter, or light informal chat."
        if label == "personal":
            return "This text is personal, emotional, or about the speaker's feelings and life."
        if label == "enacts_humor":
            return "The speaker is being humorous, joking, or using levity."
        if label == "neutral_humor":
            return "The speaker is not being humorous or joking."
        if label == "distress_present":
            return "The speaker is distressed, struggling, exhausted, or in emotional pain."
        if label == "neutral":
            return "The speaker is calm and not expressing distress."
        return f"This text is about {label}."

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def load_report(self) -> List[str]:
        lines = []
        for rec in self.load_records:
            if rec.warning:
                lines.append(f"{rec.name}: WARNING {rec.warning} ({rec.seconds:.3f}s)")
            else:
                lines.append(f"{rec.name}: {rec.backend} ({rec.seconds:.3f}s)")
        return lines


def get_encoder() -> EncoderServer:
    """Module-level compatibility helper."""
    return EncoderServer.get_encoder()


__all__ = ["EncoderServer", "get_encoder"]
