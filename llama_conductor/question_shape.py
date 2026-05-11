"""Standalone question-shape classifier.

This module is intentionally narrow:
- classify the shape of a user question
- keep the logic pure and deterministic
- do not read router state, Codex state, or retrieval outputs
- do not assemble prompts or emit footers

Caller policy for ``unknown``:
- treat it as the existing hard factual gate unless the caller explicitly
  upgrades it with stronger evidence

This module is a shared leaf dependency. It is owned by neither the router nor
the Codex runtime, and can be imported by both without coupling them to each
other.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Tuple


QuestionKind = Literal["factual", "analogical", "speculative", "mechanistic", "unknown"]


@dataclass(frozen=True, slots=True)
class QuestionShape:
    kind: QuestionKind
    confidence: Literal["low", "medium", "high"]
    signals: Tuple[str, ...]


_WH_FACTUAL_RE = re.compile(
    r"^\s*(?:what|when|who|where|which|how\s+many|how\s+much|define|define\s+the)\b",
    re.IGNORECASE,
)
_FACTUAL_ANCHOR_RE = re.compile(
    r"\b("
    r"year|date|date\s+did|did\s+.*\s+go|bankrupt|invented|introduced|released|"
    r"what\s+year|what\s+date|who\s+was|where\s+was|when\s+did|how\s+many|how\s+much"
    r")\b",
    re.IGNORECASE,
)
_HISTORICAL_TEMPORAL_RE = re.compile(
    r"\b(earlier than they were|later than|before they were|in what year|what year|when did)\b",
    re.IGNORECASE,
)
_MODAL_RE = re.compile(r"\b(could|would|might|should)\b", re.IGNORECASE)
_COUNTERFACTUAL_RE = re.compile(
    r"\b("
    r"what if|suppose|imagine if|in a world where|had .*?would|"
    r"would have|could have|might have|if .*? had"
    r")\b",
    re.IGNORECASE,
)
_MODAL_COUNTERFACTUAL_FRAME_RE = re.compile(
    r"\b(?:could|would|might|should|shall|will|may|must)\b(?:\W+\w+){0,4}\W+\bhave\b",
    re.IGNORECASE,
)
_ANALOGY_PHRASE_RE = re.compile(
    r"\b("
    r"something like|similar to|as if|sort of like|kind of like|like a|like an|"
    r"would .*?\bhelp(?:ed|ing)?\b|could .*?\bhelp(?:ed|ing)?\b|might .*?\bhelp(?:ed|ing)?\b|"
    r"benefit(?:ed|ing)?|improv(?:e|ed|ing)|split-worker|specialist worker|worker split|"
    r"worker decomposition|decomposition"
    r")\b",
    re.IGNORECASE,
)
_ANALOGY_STRUCTURE_RE = re.compile(
    r"\b("
    r"help(?:ed|ing)?|benefit(?:ed|ing)?|improv(?:e|ed|ing)|"
    r"specialist|workers?|architecture|design|decomposition|split"
    r")\b",
    re.IGNORECASE,
)
_TARGET_ENTITY_RE = re.compile(r"\b(amiga|commodore|swarm|llama-conductor|llama conductor)\b", re.IGNORECASE)
_MECHANISTIC_DEF_ANCHOR_RE = re.compile(
    r"\b(?:what is|what are|what does|how does|how do)\b",
    re.IGNORECASE,
)
_MECHANISTIC_CAUSAL_CLAUSE_RE = re.compile(
    r"\b(?:and|,|;)\s+(?:why does|why do|why is|how does|what makes|what causes)\b",
    re.IGNORECASE,
)
_QUESTION_MARK_RE = re.compile(r"\?")


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9\-']{1,}", _norm(text).lower())
    )


def _is_analogical(text: str) -> tuple[bool, Tuple[str, ...], Literal["low", "medium", "high"]]:
    raw = _norm(text)
    if not raw:
        return False, tuple(), "low"

    signals = []
    low = raw.lower()

    if _ANALOGY_PHRASE_RE.search(raw):
        signals.append("analogical_phrase")
    if _TARGET_ENTITY_RE.search(raw):
        signals.append("target_entity")
    if _ANALOGY_STRUCTURE_RE.search(raw):
        signals.append("comparison_structure")
    if _MODAL_RE.search(raw):
        signals.append("modal")
    if re.search(r"\b(help(?:ed|ing)?|benefit(?:ed|ing)?|improv(?:e|ed|ing)?)\b", low):
        signals.append("support_verb")
    if re.search(r"\b(something|somebody|someone|somewhere|somehow)\b", low) and re.search(r"\blike\b", low):
        signals.append("approximation_marker")
    if re.search(r"\b(would|could|might)\b", low) and re.search(r"\b(help|helped|benefit|benefited|improve|improved)\b", low):
        signals.append("modal_support_pair")

    strong_phrase = bool(_ANALOGY_PHRASE_RE.search(raw))
    structure_score = len(set(signals))
    if strong_phrase or structure_score >= 3:
        conf: Literal["low", "medium", "high"] = "high"
    elif structure_score == 2:
        conf = "medium"
    else:
        conf = "low"

    if strong_phrase or ("modal_support_pair" in signals and ("comparison_structure" in signals or "target_entity" in signals)):
        return True, tuple(dict.fromkeys(signals)), conf

    if ("comparison_structure" in signals and "target_entity" in signals and "modal" in signals):
        return True, tuple(dict.fromkeys(signals)), conf

    return False, tuple(dict.fromkeys(signals)), conf


def _is_mechanistic(text: str) -> tuple[bool, Tuple[str, ...], Literal["low", "medium", "high"]]:
    raw = _norm(text)
    if not raw:
        return False, tuple(), "low"

    signals = []
    low = raw.lower()

    if _MECHANISTIC_DEF_ANCHOR_RE.search(raw):
        signals.append("mechanistic_anchor")
    if _MECHANISTIC_CAUSAL_CLAUSE_RE.search(raw):
        signals.append("mechanistic_causal_clause")
    if re.search(
        r"\b(?:what is|what are|what does|how does|how do)\b.*\b(?:and|,|;)\s+(?:why does|why do|why is|how does|what makes|what causes)\b",
        low,
    ):
        signals.append("mechanistic_compound")

    if "mechanistic_anchor" in signals and "mechanistic_causal_clause" in signals:
        conf: Literal["low", "medium", "high"] = "high"
        return True, tuple(dict.fromkeys(signals)), conf
    return False, tuple(dict.fromkeys(signals)), "low"


def _is_speculative(text: str) -> tuple[bool, Tuple[str, ...], Literal["low", "medium", "high"]]:
    raw = _norm(text)
    if not raw:
        return False, tuple(), "low"

    signals = []
    low = raw.lower()

    if _COUNTERFACTUAL_RE.search(raw):
        signals.append("counterfactual_frame")
    if _MODAL_COUNTERFACTUAL_FRAME_RE.search(raw):
        signals.append("modal_counterfactual_frame")
    if re.search(r"\bwhat if\b", low):
        signals.append("what_if")
    if re.search(r"\bsuppose\b", low):
        signals.append("suppose")
    if re.search(r"\bimagine\b", low):
        signals.append("imagine")

    if "modal_counterfactual_frame" in signals and _QUESTION_MARK_RE.search(raw):
        return True, tuple(dict.fromkeys(signals)), "high"
    if len(signals) >= 2:
        return True, tuple(dict.fromkeys(signals)), "high"
    if "counterfactual_frame" in signals:
        return True, tuple(dict.fromkeys(signals)), "medium"
    if "modal_counterfactual_frame" in signals:
        return True, tuple(dict.fromkeys(signals)), "medium"
    return False, tuple(dict.fromkeys(signals)), "low"


def _is_factual(text: str) -> tuple[bool, Tuple[str, ...], Literal["low", "medium", "high"]]:
    raw = _norm(text)
    if not raw:
        return False, tuple(), "low"

    signals = []
    low = raw.lower()

    if _WH_FACTUAL_RE.search(raw):
        signals.append("factual_interrogative")
    if _FACTUAL_ANCHOR_RE.search(raw):
        signals.append("factual_anchor")
    if _HISTORICAL_TEMPORAL_RE.search(raw):
        signals.append("historical_temporal_anchor")
    if _QUESTION_MARK_RE.search(raw):
        signals.append("question_mark")
    if re.search(r"\b\d{3,4}\b", raw):
        signals.append("numeric_anchor")
    if re.search(r"\b(go bankrupt|born|born in|released in|introduced in|invented earlier)\b", low):
        signals.append("historical_fact_pattern")

    if "factual_interrogative" in signals and ("factual_anchor" in signals or "historical_temporal_anchor" in signals):
        return True, tuple(dict.fromkeys(signals)), "high"
    if "historical_fact_pattern" in signals and "question_mark" in signals:
        return True, tuple(dict.fromkeys(signals)), "high"
    if "factual_interrogative" in signals and "question_mark" in signals and len(signals) >= 2:
        return True, tuple(dict.fromkeys(signals)), "high"
    if "question_mark" in signals and ("factual_anchor" in signals or "historical_temporal_anchor" in signals):
        return True, tuple(dict.fromkeys(signals)), "medium"
    return False, tuple(dict.fromkeys(signals)), "low"


def classify_question_shape(text: str) -> QuestionShape:
    """Classify the shape of a user question.

    The detector is intentionally conservative:
    - analogical beats speculative when the prompt asks for a comparison or
      a "would this help that" style synthesis
    - factual is preferred when the prompt is a direct lookup with a concrete
      historical or entity anchor
    - speculative is reserved for explicit counterfactual framing
    - unknown is returned when the prompt is ambiguous or too weakly shaped

    Caller policy for `unknown`:
    - treat it as the existing hard factual gate unless the caller explicitly
      upgrades it with stronger evidence.
    """
    raw = _norm(text)
    if not raw:
        return QuestionShape(kind="unknown", confidence="low", signals=tuple())

    analogical, analogical_signals, analogical_conf = _is_analogical(raw)
    if analogical:
        return QuestionShape(kind="analogical", confidence=analogical_conf, signals=analogical_signals)

    mechanistic, mechanistic_signals, mechanistic_conf = _is_mechanistic(raw)
    if mechanistic:
        return QuestionShape(kind="mechanistic", confidence=mechanistic_conf, signals=mechanistic_signals)

    factual, factual_signals, factual_conf = _is_factual(raw)
    if factual:
        return QuestionShape(kind="factual", confidence=factual_conf, signals=factual_signals)

    speculative, speculative_signals, speculative_conf = _is_speculative(raw)
    if speculative:
        return QuestionShape(kind="speculative", confidence=speculative_conf, signals=speculative_signals)

    signals = []
    if _MODAL_RE.search(raw):
        signals.append("modal")
    if _QUESTION_MARK_RE.search(raw):
        signals.append("question_mark")
    if _TARGET_ENTITY_RE.search(raw):
        signals.append("target_entity")
    if len(signals) >= 2:
        return QuestionShape(kind="unknown", confidence="low", signals=tuple(dict.fromkeys(signals)))
    return QuestionShape(kind="unknown", confidence="low", signals=tuple(dict.fromkeys(signals)))
