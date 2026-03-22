"""KAIOKEN Phase 0 telemetry helpers (sensor-only, fail-open).

Phase 0 contract:
- classify register features for observability
- emit telemetry only (no coercion/actuation)
- never log raw user text
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Set


ROUTE_CLASSES = {"deterministic", "sidecar", "vision", "model_chat", "control"}
COERCION_ELIGIBLE = {"model_chat"}

_WORD_RE = re.compile(r"[A-Za-z0-9_']+")

_WORKING_WORDS = {
    "implement",
    "fix",
    "test",
    "patch",
    "router",
    "config",
    "command",
    "error",
    "regression",
    "contract",
    "deploy",
    "release",
    "pipeline",
}
_PERSONAL_WORDS = {
    "sad",
    "lonely",
    "anxious",
    "stressed",
    "worried",
    "afraid",
    "hurt",
    "love",
    "overwhelmed",
    "upset",
}
_CASUAL_WORDS = {
    "lol",
    "lmao",
    "haha",
    "bro",
    "joke",
    "funny",
    "meme",
    "vibes",
    "mate",
}
_DIRECTIVE_WORDS = {
    "do",
    "show",
    "tell",
    "set",
    "run",
    "use",
    "change",
    "add",
    "remove",
    "please",
    "give",
    "write",
    "rank",
    "summarize",
    "summarise",
    "pick",
    "choose",
    "prioritize",
    "prioritise",
    "need",
    "continue",
    "return",
}
_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "im", "i'm"}

_DIRECTIVE_PATTERNS = (
    "give me",
    "show me",
    "tell me",
    "write",
    "do not",
    "keep this",
    "only",
    "yes/no",
    "one-line",
    "one line",
    "top three",
    "exact",
    "command sequence",
    "i need",
    "no explanation",
    "no commentary",
    "in one sentence",
    "only the final",
    "short version",
    "choose one",
    "park this",
    "don't improvise",
)

_PERSONAL_STATE_PHRASES = (
    "at capacity",
    "overloaded",
    "bandwidth is low",
    "confidence is shaky",
    "trying to stay chill",
    "i'm tired",
    "i am tired",
    "i can keep going",
    "i feel",
    "not fine",
    "not stable",
    "masking",
    "calm-ish",
)

_PLAYFUL_MARKERS_STRONG = ("lol", "lmao", "haha", "hehe")
_PLAYFUL_MARKERS_LIGHT = (
    "bro",
    "mate",
    "vibes",
    "meme",
    "gremlin",
    "cheeky",
    "joking",
    "banter",
    "unserious",
    "funny",
)
_PLAYFUL_TONE_MARKERS = (
    "cursed",
    "theatrically",
    "charming",
    "incredible",
    "audacity",
    "chaotic good",
    "clown-car",
    "jank",
    "spicy nonsense",
    "i'll allow it",
    "kind of art",
    "respect the commitment",
    "illegal but efficient",
    "wrong, but",
    "messy but",
    "mess works",
)
_META_HUMOR_REFERENCE_MARKERS = (
    "still joking",
    "making jokes",
    "cracking jokes",
    "joking through",
    "joking so",
    "banter on the surface",
    "banter outside",
    "doing the bit",
)
_FRUSTRATION_MARKERS = (
    "wtf",
    "not what i asked",
    "timeout",
    "failed",
    "fail",
    "error",
    "regression",
    "choke",
    "why did",
)

_VUH_MARKERS = (
    "spiral",
    "combust",
    "half banter and half panic",
    "joking, but also",
    "before i say something cursed",
    "calibrating loudly",
)

_VUH_STRESS_MARKERS = (
    "imploding",
    "stress response",
    "stress signal",
    "stress",
    "not okay",
    "panic",
    "spiral",
    "spiralling",
    "overloaded",
    "cooked",
    "snapping",
    "breakdown",
    "feral",
    "scared",
    "stretched thin",
    "white-knuckling",
    "melting",
    "fraying",
    "running on fumes",
    "bandwidth is shot",
    "bandwidth is gone",
    "at limit",
    "crash",
    "fumes",
    "fear core",
    "panic inside",
    "quietly melting",
    "chest is tight",
    "at capacity",
    "triage",
    "nervous",
    "lifeline",
    "brain fries",
    "not fine",
    "not stable",
    "masking",
    "vulnerability",
    "choking",
    "medically dramatic",
    "cry",
    "masking",
    "stretched thin",
    "hard things",
    "need sleep",
    "herniation",
    "disc herniation",
    "l5-s1",
    "spine",
    "back pain",
    "chronic pain",
    "too broken",
    "too old and too broken",
    "feel like a fraud",
    "i feel like a fraud",
)

_VUH_HUMOR_FRAMING = (
    "joking",
    "joke",
    "banter",
    "clowning",
    "doing the bit",
    "stand-up",
    "this is fine",
    "smiling in text",
    "smiling",
    "grinning",
    "chirpy",
    "fun fact",
    "making jokes",
    "jokes",
    "crack jokes",
    "jokes first",
    "banter shell",
    "humour outside",
    "humor outside",
    "laughing",
    "this is funny",
)
_DISTRESS_HINT_MARKERS = (
    "not okay",
    "running on fumes",
    "fumes",
    "bandwidth is shot",
    "bandwidth is gone",
    "at limit",
    "fraying",
    "cooked",
    "low and tired",
    "need sleep",
    "not fine",
    "not stable",
    "at capacity",
    "masking",
    "overwhelmed",
    "anxious",
    "panic",
    "scared",
    "spiral",
    "white-knuckling",
    "sad",
    "lonely",
    "hurt",
    "broken",
    "feel like a fraud",
    "i feel like a fraud",
    "too old and too broken",
    "mass-irrelevant",
    "mass irrelevant",
)

_VUH_CONTRAST_MARKERS = (
    "but",
    "except",
    "while",
    "still",
    "until",
    "apart from",
    "outside",
    "inside",
)

_CASUAL_GREETINGS = {"oi", "yo", "hey", "sup", "hiya"}


def _count_markers(text_l: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in text_l)


def _classify_subsignals(f: Dict[str, Any], macro: str) -> Set[str]:
    subs: Set[str] = set()
    directive_candidate = (
        f["directive_hits"] > 0
        or f["imperative_like"]
        or f["directive_pattern_hits"] > 0
    )
    directive_guard_personal_reflection = (
        not f["imperative_like"]
        and f["working_hits"] == 0
        and f["first_person_hits"] > 0
        and f["directive_pattern_hits"] == 0
    )
    if directive_candidate and not directive_guard_personal_reflection:
        subs.add("directive")
    if f["distress_hint_hits"] > 0:
        subs.add("distress_hint")
    light_banter = (
        f["playful_light"] > 0
        and f["frustration_markers"] == 0
        and f["directive_hits"] == 0
        and f["directive_pattern_hits"] == 0
        and not f["imperative_like"]
        and f["working_hits"] == 0
    )
    tone_playful = f["playful_tone_hits"] > 0 and f["frustration_markers"] == 0
    meta_humor_reference = f["meta_humor_ref_hits"] > 0 and (
        f["vuh_stress_hits"] > 0
        or f["personal_state_hits"] > 0
        or f["disclosure_markers"] > 0
    )
    if (f["playful_strong"] > 0 or light_banter or tone_playful) and not meta_humor_reference:
        subs.add("playful")
    # Humour explicitly masking stress/state.
    has_humor = (
        f["playful_signal"] > 0
        or f["vuh_markers"] > 0
        or f["vuh_humor_hits"] > 0
    )
    has_stress = (
        f["vuh_stress_hits"] > 0
        or f["personal_state_hits"] > 0
        or f["distress_hint_hits"] > 0
    )
    masking_pattern = (
        f["vuh_humor_hits"] > 0
        and f["vuh_stress_hits"] > 0
    )
    if has_humor and (has_stress or masking_pattern) and (
        f["has_first_person"]
        or f["disclosure_markers"] > 0
        or f["vuh_contrast_hits"] > 0
    ):
        subs.add("vulnerable_under_humour")
        subs.discard("distress_hint")
    return subs


@dataclass(frozen=True)
class KaiokenClassification:
    macro: str
    confidence: str
    subsignals: List[str]
    scores: Dict[str, int]
    features: Dict[str, Any]


def _safe_route_class(route_class: str) -> str:
    rc = str(route_class or "").strip().lower()
    return rc if rc in ROUTE_CLASSES else "control"


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(str(text or ""))]


def extract_features(text: str) -> Dict[str, Any]:
    toks = _tokens(text)
    tok_set = set(toks)
    first_token = toks[0] if toks else ""
    low = str(text or "").lower()

    working_hits = sum(1 for t in toks if t in _WORKING_WORDS)
    personal_hits = sum(1 for t in toks if t in _PERSONAL_WORDS)
    casual_hits = sum(1 for t in toks if t in _CASUAL_WORDS)
    directive_hits = sum(1 for t in toks if t in _DIRECTIVE_WORDS)
    first_person_hits = sum(1 for t in toks if t in _FIRST_PERSON)

    disclosure_markers = 0
    for marker in ("i feel", "i think", "i am", "i'm", "for me", "about me"):
        if marker in low:
            disclosure_markers += 1

    directive_pattern_hits = _count_markers(low, _DIRECTIVE_PATTERNS)
    personal_state_hits = _count_markers(low, _PERSONAL_STATE_PHRASES)
    playful_strong = _count_markers(low, _PLAYFUL_MARKERS_STRONG)
    playful_light = _count_markers(low, _PLAYFUL_MARKERS_LIGHT)
    playful_tone_hits = _count_markers(low, _PLAYFUL_TONE_MARKERS)
    meta_humor_ref_hits = _count_markers(low, _META_HUMOR_REFERENCE_MARKERS)
    frustration_markers = _count_markers(low, _FRUSTRATION_MARKERS)
    vuh_markers = _count_markers(low, _VUH_MARKERS)
    vuh_stress_hits = _count_markers(low, _VUH_STRESS_MARKERS)
    vuh_humor_hits = _count_markers(low, _VUH_HUMOR_FRAMING)
    vuh_contrast_hits = _count_markers(low, _VUH_CONTRAST_MARKERS)
    distress_hint_hits = _count_markers(low, _DISTRESS_HINT_MARKERS)

    playful_signal = (2 * playful_strong) + playful_light
    if frustration_markers > 0 and playful_strong == 0:
        playful_signal = max(0, playful_signal - frustration_markers)

    return {
        "char_count": len(str(text or "")),
        "token_count": len(toks),
        "question_count": str(text or "").count("?"),
        "exclaim_count": str(text or "").count("!"),
        "working_hits": working_hits,
        "personal_hits": personal_hits,
        "casual_hits": casual_hits,
        "directive_hits": directive_hits,
        "directive_pattern_hits": directive_pattern_hits,
        "first_person_hits": first_person_hits,
        "disclosure_markers": disclosure_markers,
        "personal_state_hits": personal_state_hits,
        "playful_signal": playful_signal,
        "playful_strong": playful_strong,
        "playful_light": playful_light,
        "playful_tone_hits": playful_tone_hits,
        "meta_humor_ref_hits": meta_humor_ref_hits,
        "frustration_markers": frustration_markers,
        "vuh_markers": vuh_markers,
        "vuh_stress_hits": vuh_stress_hits,
        "vuh_humor_hits": vuh_humor_hits,
        "vuh_contrast_hits": vuh_contrast_hits,
        "distress_hint_hits": distress_hint_hits,
        "imperative_like": bool(first_token in _DIRECTIVE_WORDS),
        "contains_command_prefix": bool(low.strip().startswith((">>", "##"))),
        "has_first_person": bool(tok_set.intersection(_FIRST_PERSON)),
        "greeting_short": bool(
            len(toks) <= 3 and any(t in _CASUAL_GREETINGS for t in tok_set)
        ),
    }


def classify_register(text: str) -> KaiokenClassification:
    f = extract_features(text)

    score_working = int(
        (2 * f["working_hits"])
        + f["directive_hits"]
        + (1 if f["imperative_like"] else 0)
    )
    score_personal = int(
        (2 * f["personal_hits"])
        + f["first_person_hits"]
        + (2 * f["disclosure_markers"])
        + f["personal_state_hits"]
    )
    score_casual = int((2 * f["casual_hits"]) + f["playful_signal"])

    scores = {
        "working": score_working,
        "personal": score_personal,
        "casual": score_casual,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1]
    gap = top_score - second_score

    if top_score <= 1:
        top_label = "casual" if f["greeting_short"] else "working"
        conf = "low"
    elif gap >= 3 or (top_score >= 5 and gap >= 2):
        conf = "high"
    elif gap >= 1:
        conf = "medium"
    else:
        conf = "low"

    subsignals = list(_classify_subsignals(f, top_label))

    return KaiokenClassification(
        macro=top_label,
        confidence=conf,
        subsignals=sorted(set(subsignals)),
        scores=scores,
        features=f,
    )


def _session_hash(session_id: str, salt: str) -> str:
    payload = f"{salt}|{session_id}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_session_filename(session_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id or "").strip())
    return s or "unknown_session"


def _default_kaioken_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "total_recall" / "kaioken"


def append_kaioken_telemetry(
    *,
    session_id: str,
    turn_index: int,
    user_text: str,
    route_class: str,
    enabled: bool = True,
    mode: str = "log_only",
    log_all_routes: bool = True,
    session_hash_salt: str = "kaioken-v1",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not enabled:
        return None

    rc = _safe_route_class(route_class)
    if (not log_all_routes) and rc != "model_chat":
        return None

    c = classify_register(user_text or "")
    coercion_eligible = rc in COERCION_ELIGIBLE
    if coercion_eligible and str(mode).strip().lower() == "log_only":
        decision = "deferred"
        reason = "phase0_log_only"
    elif coercion_eligible:
        decision = "deferred"
        reason = "phase0_no_actuation"
    else:
        decision = "suppressed"
        reason = "exempt_route_class"

    row: Dict[str, Any] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "session_id_hash": _session_hash(session_id, session_hash_salt),
        "turn_index": int(max(1, turn_index)),
        "route_class": rc,
        "coercion_eligible": bool(coercion_eligible),
        "predicted_macro": c.macro,
        "predicted_confidence": c.confidence,
        "predicted_subsignals": c.subsignals,
        "scores": c.scores,
        "features": c.features,
        "coercion_decision": decision,
        "suppression_reason": reason,
    }
    if extra:
        row["outcome"] = dict(extra)

    out_dir = _default_kaioken_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_session_filename(session_id)}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return row
