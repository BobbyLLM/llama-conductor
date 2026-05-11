"""KAIOKEN vNext control plane.

Authoritative spec: docs/planning/KAIOKEN-vNEXT.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from .text_metrics import content_word_count, tokens

logger = logging.getLogger(__name__)
_GREETINGS = {
    "sup",
    "hey",
    "heya",
    "yo",
    "hi",
    "hello",
    "hullo",
    "howdy",
}
_PHATIC_SHORT_WORDS = {
    "aight",
    "cool",
    "nice",
    "lol",
    "yeah",
    "fair",
    "right",
    "got",
    "exactly",
    "true",
}
_PHATIC_PHRASES = {
    "thanks",
    "thank you",
    "thx",
    "appreciate it",
    "fair enough",
    "got it",
    "makes sense",
}
_PHATIC = {
    "thanks",
    "thank you",
    "thx",
    "appreciate it",
    "fair enough",
    "right",
    "sure",
    "yeah",
    "haha",
    "true",
    "exactly",
    "got it",
    "makes sense",
}
_REPAIR_PATTERNS = (
    "no that's wrong",
    "you misread",
    "not what i meant",
    "you got that wrong",
    "that's not what i said",
    "that's not what i asked",
    "you missed the point",
    "you're wrong about",
    "incorrect,",
    "wrong,",
    "no, that's not",
    "that's incorrect",
    "you misunderstood",
    "i didn't say that",
    "that's not right",
    "you've got it wrong",
)
_QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
    "is",
    "are",
    "was",
    "were",
}
_COMMAND_PREFIXES = (
    "please ",
    "can you ",
    "could you ",
    "would you ",
    "help me ",
    "show me ",
    "tell me ",
    "compare ",
    "explain ",
    "describe ",
    "analyze ",
    "debug ",
    "list ",
    "give me ",
    "do ",
)


def _looks_like_greeting(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if raw in _GREETINGS or raw in _PHATIC:
        return True
    toks = tokens(raw)
    if len(toks) <= 2 and any(tok in _GREETINGS for tok in toks):
        return True
    return False


def _looks_like_phatic(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if raw in _PHATIC:
        return True
    if raw.startswith(("can you ", "could you ", "would you ", "will you ")):
        return False
    toks = tokens(raw)
    if len(toks) <= 4 and (
        any(tok in _PHATIC_SHORT_WORDS for tok in toks)
        or any(phrase in raw for phrase in _PHATIC_PHRASES)
    ):
        return True
    return False


_PERSONAL_STRONG_PHRASES = (
    "feel like a fraud",
    "too old",
    "too broken",
    "irrelevant",
    "pointless",
    "why bother",
    "what's the point of me",
    "what's even the point of me",
    "what is the point of me",
    "meltdown",
    "asd",
    "a fraud",
    "broken",
    "l5-s1",
    "disc herniation",
)
_PERSONAL_WEAK_PHRASES = (
    "i don't know why",
    "i'm so",
    "i can't",
    "i just",
    "maybe i'm",
    "i'm not sure i",
    "i don't know what",
    "i don't know",
    "exhausted",
    "struggling",
    "same old",
    "hell of a",
)
_BODY_PART_RE = re.compile(
    r"\bmy\s+(?:back|health|spine|arm|arms|leg|legs|hand|hands|head|heart|"
    r"knee|knees|hip|hips|neck|shoulder|shoulders|foot|feet|ankle|ankles|"
    r"wrist|wrists|body|brain|stomach|gut|lung|lungs|eye|eyes|ear|ears|"
    r"teeth|tooth|jaw|bone|bones)\b",
    re.IGNORECASE,
)
_BANTER_ACK_PHRASES = (
    "well that was",
    "a bit",
    "you're a",
    "rude",
    "cheeky",
    "bit much",
    "harsh",
    "lol what",
)


def _count_phrase_hits(raw: str, phrases: Sequence[str]) -> int:
    return sum(1 for phrase in phrases if phrase in raw)


def _personal_signal_counts(text: str) -> Tuple[int, int]:
    raw = str(text or "").strip().lower()
    strong = _count_phrase_hits(raw, _PERSONAL_STRONG_PHRASES)
    if _BODY_PART_RE.search(raw):
        strong += 1
    weak = _count_phrase_hits(raw, _PERSONAL_WEAK_PHRASES)
    return strong, weak


def _looks_like_personal(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if _BODY_PART_RE.search(raw) and "?" in raw:
        return True
    strong, weak = _personal_signal_counts(text)
    return strong > 0 and (strong + weak) >= 2


def _looks_like_banter_ack(text: str) -> bool:
    raw = str(text or "").strip().lower()
    return any(phrase in raw for phrase in _BANTER_ACK_PHRASES)


def _looks_like_distress_hint(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if _BODY_PART_RE.search(raw):
        return True
    return any(
        marker in raw
        for marker in (
            "broken",
            "fraud",
            "irrelevant",
            "pointless",
            "too old",
            "can't do this",
            "what's the point",
            "what is the point",
            "i don't know why i bother",
            "exhausted",
            "struggling",
            "meltdown",
            "l5-s1",
            "disc herniation",
            "back hurt",
        )
    )


def _looks_like_playful(text: str) -> bool:
    raw = str(text or "").strip().lower()
    return any(
        marker in raw
        for marker in (
            "haha",
            "lol",
            "lmao",
            "fair enough",
            "tbh",
            "ngl",
            "my code is",
            "vibes and prayer",
            "just winging it",
        )
    )


def _looks_like_repair(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    clauses = re.split(r"(?<=[.!?])\s+", raw)
    for clause in clauses:
        clause = clause.strip(" \t\r\n\"'“”‘’([{")
        if not clause:
            continue
        for pattern in _REPAIR_PATTERNS:
            if clause.startswith(pattern):
                return True
    return False


def _looks_like_command(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if raw.startswith(">>"):
        return True
    if raw.startswith(("can you ", "could you ", "would you ", "will you ")):
        return False
    return any(raw.startswith(prefix) for prefix in _COMMAND_PREFIXES)


def _looks_like_working(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    toks = tokens(raw)
    if any(tok in _QUESTION_WORDS for tok in toks):
        return True
    if any(marker in raw for marker in (" debug ", " compare ", " explain ", " analyze ", " define ", " mean", " meaning", " help ", " how do i", " what does", " what is")):
        return True
    if "?" in raw:
        return True
    if raw.startswith("do a search"):
        return True
    return False


def _is_short_continuation(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if content_word_count(raw) > 2:
        return False
    toks = tokens(raw)
    if len(toks) > 4:
        return False
    return True


def _fallback_overlap(query: str, candidate: str) -> float:
    q = set(tokens(query))
    c = set(tokens(candidate))
    if not q or not c:
        return 0.0
    return len(q & c) / max(1.0, (len(q) * len(c)) ** 0.5)


def _pick_e5_register(text: str) -> Tuple[str, float, float]:
    exemplar_map = {
        "casual": [
            "sup",
            "how's it going",
            "haha yeah",
            "fair enough",
            "lol okay",
            "aight cool",
            "that's fair",
            "yeah nah",
            "nice one",
            "haha fair",
            "oh right",
            "makes sense I guess",
        ],
        "personal": [
            "I've been struggling",
            "feeling rough today",
            "just venting",
            "I don't know why I bother",
            "maybe I'm just too broken",
            "same old me",
            "I feel like a fraud",
            "it's been a rough stretch",
            "I'm exhausted",
            "I don't know what I'm doing anymore",
            "everything feels pointless",
            "I just needed to say that",
        ],
        "working": [
            "can you help me debug this function",
            "what does X mean",
            "compare these two options",
            "how do I fix this error",
            "explain how this works",
            "what's the difference between",
            "write me a function that",
            "help me understand",
            "is this the right approach",
            "what would you recommend",
            "how does this interact with",
            "walk me through this",
        ],
    }

    scores: Dict[str, float] = {}
    try:
        from .encoder_server import EncoderServer

        enc = EncoderServer.get_encoder()
        for register, exemplars in exemplar_map.items():
            sims = enc.similarity(text, exemplars)
            scores[register] = max(sims) if sims else float("-inf")
    except Exception:
        for register, exemplars in exemplar_map.items():
            scores[register] = max((_fallback_overlap(text, ex) for ex in exemplars), default=float("-inf"))

    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_register, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else top_score
    ambiguity_gap = float(top_score - second_score) if top_score != float("-inf") else 0.0
    return top_register, float(top_score), ambiguity_gap


@dataclass
class KaiokenControlObject:
    # Core routing
    macro_register: str
    response_mode: str
    speech_act: str

    # Behavioral signals
    playful: bool
    distress_hint: bool
    vuh: bool
    literal_followup: bool
    literal_anchor: str
    topic_switch: bool
    resolved_topics: list = field(default_factory=list)

    # Constraints
    advice_allowed: bool = False
    encouragement_allowed: bool = True
    repair_mode: bool = False
    banter_mode: bool = False
    phatic_mode: bool = False

    # Routing metadata
    routing_method: str = "heuristic"
    e5_score: Optional[float] = None
    reranker_score: Optional[float] = None
    ambiguity_gap: Optional[float] = None
    calibration_flag: bool = False

    # Phrase planning
    selected_phrase_ids: dict = field(default_factory=dict)

    # Swarm sockets — do not rename or repurpose these fields
    advisor_outputs: dict = field(default_factory=dict)
    evaluator_verdicts: dict = field(default_factory=dict)

    # W3 full-pass outputs - advisory unless noted
    refusal: bool = False                  # AUTHORITATIVE - hard gate, blocks W5
    risk_level: str = "none"              # values: "none" | "low" | "medium" | "high"
    confidence_floor: float = 1.0         # floor on answer confidence, 0.0-1.0
    empathy_override: bool = False        # AUTHORITATIVE - binding on W5
    profile_floor: Optional[str] = None  # "empathetic" | None


# W3 fast pass - advisory parallel path, not a replacement for classify_turn().
# Intentionally replicates heuristic+e5 logic so W4 can receive macro early.
# Phase 5 will consume this output directly. Do not merge with classify_turn().
def classify_fast_pass(turn_text: str) -> str:
    """
    W3 fast pass - macro label only, <15ms target.
    Fires before full pass to unblock W4 retrieval cascade.
    Returns macro_register: 'working' | 'casual' | 'personal'
    No KCO created. No side effects.
    """
    text = str(turn_text or "").strip()
    if not text:
        return "working"
    if _looks_like_repair(text):
        return "working"
    if _looks_like_personal(text):
        return "personal"
    if _looks_like_phatic(text) or _looks_like_greeting(text) or _looks_like_banter_ack(text):
        return "casual"
    if _looks_like_command(text) or _looks_like_working(text):
        return "working"
    # e5 fallback for ambiguous turns
    try:
        top_register, _, _ = _pick_e5_register(text)
        return top_register
    except Exception:
        return "working"


def _check_refusal_heuristic(text: str) -> tuple[bool, str, float]:
    """
    Stub - always fails open. Returns (False, 'none', 1.0).
    DeBERTa seam: replace when W3-ADV-001 regression set is built.
    """
    return False, "none", 1.0


def classify_turn(
    turn_text,
    prior_turn_text=None,
    prior_assistant_text=None,
    prior_macro_register=None,
) -> KaiokenControlObject:
    text = str(turn_text or "").strip()
    lower = text.lower()
    content_count = content_word_count(text)
    is_question = "?" in text or any(tok in _QUESTION_WORDS for tok in tokens(text))
    kco = KaiokenControlObject(
        macro_register="working",
        response_mode="normal",
        speech_act="query" if is_question else "followup",
        playful=False,
        distress_hint=False,
        vuh=False,
        literal_followup=False,
        literal_anchor="",
        topic_switch=False,
    )

    # Stage 1: repair is terminal and always wins.
    if _looks_like_repair(text):
        kco.response_mode = "repair"
        kco.repair_mode = True
        kco.speech_act = "repair"
        if prior_macro_register in {"casual", "personal", "working"}:
            kco.macro_register = prior_macro_register
        else:
            kco.macro_register = "working"
        return kco

    # Stage 2: personal register before working heuristics.
    is_personal = _looks_like_personal(text)
    if is_personal:
        kco.macro_register = "personal"

    # Stage 3: working/casual register resolution.
    if not is_personal:
        if prior_macro_register == "casual" and _is_short_continuation(text) and not _looks_like_command(text) and not _looks_like_working(text):
            kco.macro_register = "casual"
        elif _looks_like_phatic(text):
            kco.macro_register = "casual"
            kco.banter_mode = True
        elif _looks_like_greeting(text):
            kco.macro_register = "casual"
            kco.banter_mode = True
        elif _looks_like_banter_ack(text):
            kco.macro_register = "casual"
            kco.banter_mode = True
        elif _looks_like_command(text):
            kco.macro_register = "working"
        elif _looks_like_working(text):
            kco.macro_register = "working"
        else:
            top_register, top_score, ambiguity_gap = _pick_e5_register(text)
            kco.macro_register = top_register
            kco.routing_method = "e5"
            kco.e5_score = top_score
            kco.ambiguity_gap = ambiguity_gap
            kco.calibration_flag = ambiguity_gap < 0.20

    # Stage 4: speech act resolution after the register is settled.
    if kco.macro_register == "personal":
        kco.speech_act = "checkin" if "?" in text else "vent"
    elif _looks_like_phatic(text):
        kco.speech_act = "phatic"
        kco.banter_mode = True
    elif _looks_like_greeting(text):
        kco.speech_act = "greeting"
        kco.banter_mode = True
    elif _looks_like_banter_ack(text):
        kco.speech_act = "banter_ack"
        kco.banter_mode = True
    elif _looks_like_command(text):
        kco.speech_act = "directive"
    elif kco.macro_register == "working":
        kco.speech_act = "query" if is_question else "followup"
    else:
        kco.speech_act = "query" if is_question else "followup"

    # Stage 5: behavioral signals last.
    playful_mark = _looks_like_playful(text)
    kco.playful = playful_mark or (kco.macro_register == "casual" and "!" in text and content_count <= 6)
    kco.distress_hint = kco.macro_register == "personal" and _looks_like_distress_hint(text)
    try:
        _VUH_EXEMPLARS = [
            "haha yeah I'm totally fine, just haven't slept in three days",
            "lol why do I even bother",
            "jokes aside I don't know what I'm doing anymore",
            "haha same old me, totally not falling apart",
            "lmao it's fine everything's fine",
            "haha yeah nah I'm good just exhausted and kind of broken",
        ]
        from .encoder_server import EncoderServer
        enc = EncoderServer.get_encoder()

        humor_scores = enc.classify(text, ["enacts_humor", "neutral_humor"])
        humor_present = float(humor_scores.get("enacts_humor", 0.0)) > 0.7
        if humor_present:
            e5_scores = enc.similarity(text, _VUH_EXEMPLARS)
            e5_gate = max(e5_scores) >= 0.87 if e5_scores else False
            kco.vuh = e5_gate
        else:
            kco.vuh = False

    except Exception as exc:
        logger.warning("VUH encoder gate fallback: %s", exc)
        kco.vuh = bool(kco.playful and kco.distress_hint)

    # Stage 6: W3 full-pass outputs - refusal gate, risk, empathy_override, profile_floor
    kco.refusal, kco.risk_level, kco.confidence_floor = _check_refusal_heuristic(text)
    if kco.vuh:
        # advisory only - downstream consumers read this as a ceiling on stated confidence.
        # Not a hard gate at this stage. Will become a gating input when W6 is wired.
        kco.risk_level = "high"
        kco.empathy_override = True
        kco.profile_floor = "empathetic"
        kco.confidence_floor = min(kco.confidence_floor, 0.9)
    elif kco.distress_hint:
        # advisory only - downstream consumers read this as a ceiling on stated confidence.
        # Not a hard gate at this stage. Will become a gating input when W6 is wired.
        kco.risk_level = "medium"
        kco.empathy_override = True
        kco.profile_floor = "empathetic"
        kco.confidence_floor = min(kco.confidence_floor, 0.95)
    elif kco.macro_register == "personal":
        kco.risk_level = "low"

    return kco


def log_kco_telemetry(kco: KaiokenControlObject, turn_index: int):
    payload = asdict(kco)
    payload["turn_index"] = turn_index
    logger.info("kco_telemetry=%s", payload)


def assert_kaioken_mode_exclusion(legacy_enabled: bool, serious_enabled: bool, vnext_enabled: bool):
    enabled = [bool(legacy_enabled), bool(serious_enabled), bool(vnext_enabled)]
    if sum(enabled) > 1:
        raise RuntimeError(
            "Kaioken mode exclusion violated: legacy_enabled=%s serious_enabled=%s vnext_enabled=%s"
            % (legacy_enabled, serious_enabled, vnext_enabled)
        )


__all__ = [
    "KaiokenControlObject",
    "classify_fast_pass",
    "classify_turn",
    "log_kco_telemetry",
    "assert_kaioken_mode_exclusion",
]
