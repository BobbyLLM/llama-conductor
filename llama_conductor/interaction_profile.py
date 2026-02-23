"""Deterministic per-session interaction profile (ephemeral)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


FIELD_ENUMS: Dict[str, Set[str]] = {
    "correction_style": {"direct", "neutral", "softened"},
    "failure_tolerance": {"ack_reframe", "minimal", "detailed_postmortem"},
    "ack_reframe_style": {"plain", "sharp", "feral"},
    "sarcasm_level": {"off", "low", "medium", "high"},
    "snark_tolerance": {"low", "medium", "high"},
    "verbosity": {"compact", "standard", "expanded"},
}
FIELD_BOOLEANS: Set[str] = {"profanity_ok", "sensitive_override"}
SETTABLE_FIELDS: List[str] = [
    "correction_style",
    "failure_tolerance",
    "ack_reframe_style",
    "sarcasm_level",
    "snark_tolerance",
    "profanity_ok",
    "sensitive_override",
    "verbosity",
]

SCORE_KEYS = [
    "directness",
    "evidence_posture",
    "mode_rationale_needed",
    "sarcasm_level",
    "snark_tolerance",
    "apology_theater_rejection",
]

_SENSITIVE_PRIMARY_RE = re.compile(
    r"("
    r"\b(email|draft|write|compose|send)\b.{0,60}\b(boss|manager|client|hr|lawyer|doctor|patient)\b"
    r"|"
    r"\b(cover letter|resume|curriculum vitae|cv)\b"
    r"|"
    r"\b(legal advice|medical advice|diagnosis|prescription)\b"
    r"|"
    r"\b(hr complaint|formal complaint|compliance report)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class InteractionProfile:
    correction_style: str = "neutral"
    failure_tolerance: str = "ack_reframe"
    ack_reframe_style: str = "plain"
    sarcasm_level: str = "off"
    snark_tolerance: str = "low"
    profanity_ok: bool = False
    sensitive_override: bool = False
    verbosity: str = "standard"
    confidence: float = 0.0
    asks_for: List[str] = field(default_factory=list)
    rejects: List[str] = field(default_factory=list)
    conversation_patterns: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in SCORE_KEYS})
    trait_distinct_turns: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in SCORE_KEYS})
    trait_last_seen_turn: Dict[str, int] = field(default_factory=lambda: {k: -9999 for k in SCORE_KEYS})
    active_traits: List[str] = field(default_factory=list)
    manual_overrides: Set[str] = field(default_factory=set)
    last_updated_turn: int = 0
    last_sensitive_context: str = ""


@dataclass
class EffectiveProfile:
    correction_style: str
    failure_tolerance: str
    ack_reframe_style: str
    sarcasm_level: str
    snark_tolerance: str
    profanity_ok: bool
    verbosity: str
    sensitive_context: bool


def new_profile() -> InteractionProfile:
    return InteractionProfile()


def reset_profile(profile: InteractionProfile) -> None:
    fresh = new_profile()
    profile.__dict__.update(fresh.__dict__)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _set_add_unique(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _bool_from_text(value: str) -> Optional[bool]:
    v = (value or "").strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


def profile_set(profile: InteractionProfile, field_name: str, raw_value: str) -> Tuple[bool, str]:
    field_name = (field_name or "").strip()
    raw_value = (raw_value or "").strip()
    if field_name not in SETTABLE_FIELDS:
        allowed = ",".join(SETTABLE_FIELDS)
        return False, f"[profile] invalid field: {field_name} (allowed: {allowed})"

    if field_name in FIELD_ENUMS:
        val = raw_value.lower()
        allowed_vals = sorted(FIELD_ENUMS[field_name])
        if val not in FIELD_ENUMS[field_name]:
            return False, f"[profile] invalid value for {field_name}: {raw_value} (allowed: {','.join(allowed_vals)})"
        setattr(profile, field_name, val)
        profile.manual_overrides.add(field_name)
        return True, f"[profile] updated: {field_name}={val}"

    if field_name in FIELD_BOOLEANS:
        parsed = _bool_from_text(raw_value)
        if parsed is None:
            return False, f"[profile] invalid value for {field_name}: {raw_value} (allowed: true,false)"
        setattr(profile, field_name, parsed)
        profile.manual_overrides.add(field_name)
        val = "true" if parsed else "false"
        return True, f"[profile] updated: {field_name}={val}"

    return False, f"[profile] invalid field: {field_name}"


def _trait_confidence(score: float, distinct_turns: int) -> float:
    # Conservative confidence growth: avoid hitting 1.0 from a few heated turns.
    score_term = abs(score) / 8.0
    turn_term = min(1.0, float(distinct_turns) / 5.0)
    return min(1.0, score_term * turn_term)


def classify_sensitive_context(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    return bool(_SENSITIVE_PRIMARY_RE.search(t))


def _parse_turn_signals(user_text: str) -> Dict[str, float]:
    t = (user_text or "").lower()
    signals: Dict[str, float] = {}

    if any(k in t for k in ["direct", "be direct", "straight", "no hedging", "brutal honesty"]):
        signals["directness"] = max(signals.get("directness", 0.0), 3.0)
        signals["apology_theater_rejection"] = max(signals.get("apology_theater_rejection", 0.0), 2.0)

    if any(k in t for k in ["show source", "sources", "evidence", "ground", "citation"]):
        signals["evidence_posture"] = max(signals.get("evidence_posture", 0.0), 3.0)

    if "why this mode" in t or "why mode" in t:
        signals["mode_rationale_needed"] = max(signals.get("mode_rationale_needed", 0.0), 3.0)

    if any(k in t for k in ["snark", "sarcasm", "muppet", "unfuck", "lol"]):
        signals["sarcasm_level"] = max(signals.get("sarcasm_level", 0.0), 2.0)
        signals["snark_tolerance"] = max(signals.get("snark_tolerance", 0.0), 2.0)
    if any(k in t for k in ["fuck", "fucking", "shit", "wtf"]):
        signals["snark_tolerance"] = max(signals.get("snark_tolerance", 0.0), 2.0)
        signals["sarcasm_level"] = max(signals.get("sarcasm_level", 0.0), 1.0)

    if any(k in t for k in ["stop doing", "stop repeating", "fucking stop", "why are you repeating", "loosen up"]):
        signals["directness"] = max(signals.get("directness", 0.0), 3.0)
        signals["apology_theater_rejection"] = max(signals.get("apology_theater_rejection", 0.0), 3.0)
        signals["snark_tolerance"] = max(signals.get("snark_tolerance", 0.0), 2.0)

    if any(k in t for k in ["too harsh", "don't swear", "do not swear", "professional tone"]):
        signals["snark_tolerance"] = min(signals.get("snark_tolerance", 0.0), -2.0)
        signals["sarcasm_level"] = min(signals.get("sarcasm_level", 0.0), -2.0)

    if "safety theater" in t or "apology theater" in t:
        signals["apology_theater_rejection"] = max(signals.get("apology_theater_rejection", 0.0), 3.0)

    return signals


def update_profile_from_user_turn(profile: InteractionProfile, turn_no: int, user_text: str) -> None:
    signals = _parse_turn_signals(user_text)

    for trait in SCORE_KEYS:
        score = float(profile.scores.get(trait, 0.0))
        delta = signals.get(trait)
        if delta is None:
            if score > 0:
                score -= 0.5
            elif score < 0:
                score += 0.5
        else:
            score += float(delta)
            profile.trait_distinct_turns[trait] = int(profile.trait_distinct_turns.get(trait, 0)) + 1
            profile.trait_last_seen_turn[trait] = turn_no
        profile.scores[trait] = _clamp(score, -6.0, 6.0)

    if "evidence_posture" in signals:
        _set_add_unique(profile.asks_for, "evidence_first")
    if "mode_rationale_needed" in signals:
        _set_add_unique(profile.asks_for, "why_mode_choice")
    if "apology_theater_rejection" in signals:
        _set_add_unique(profile.rejects, "apology_theater")
    if "snark_tolerance" in signals and signals["snark_tolerance"] > 0:
        _set_add_unique(profile.conversation_patterns, "boundary_testing")

    if "correction_style" not in profile.manual_overrides:
        direct = profile.scores.get("directness", 0.0)
        if direct >= 3.0:
            profile.correction_style = "direct"
        elif direct <= -3.0:
            profile.correction_style = "softened"
        else:
            profile.correction_style = "neutral"

    if "sarcasm_level" not in profile.manual_overrides:
        s_sarc = profile.scores.get("sarcasm_level", 0.0)
        if s_sarc <= 1.5:
            profile.sarcasm_level = "off"
        elif s_sarc <= 3.0:
            profile.sarcasm_level = "low"
        elif s_sarc <= 4.5:
            profile.sarcasm_level = "medium"
        else:
            profile.sarcasm_level = "high"

    if "snark_tolerance" not in profile.manual_overrides:
        s_snark = profile.scores.get("snark_tolerance", 0.0)
        if s_snark <= 1.5:
            profile.snark_tolerance = "low"
        elif s_snark <= 4.0:
            profile.snark_tolerance = "medium"
        else:
            profile.snark_tolerance = "high"

    if "ack_reframe_style" not in profile.manual_overrides:
        if profile.correction_style == "direct" and profile.snark_tolerance in ("medium", "high"):
            profile.ack_reframe_style = "sharp"
        else:
            profile.ack_reframe_style = "plain"
        if (
            profile.snark_tolerance == "high"
            and profile.sarcasm_level in ("medium", "high")
            and profile.profanity_ok
        ):
            profile.ack_reframe_style = "feral"

    active: List[str] = []
    confs: List[float] = []
    all_confs: List[float] = []
    for trait in SCORE_KEYS:
        score = float(profile.scores.get(trait, 0.0))
        turns = int(profile.trait_distinct_turns.get(trait, 0))
        last_seen = int(profile.trait_last_seen_turn.get(trait, -9999))
        if turns > 0 and abs(score) > 0:
            all_confs.append(_trait_confidence(score, turns))
        if abs(score) >= 4.0 and turns >= 2 and (turn_no - last_seen) <= 8:
            active.append(trait)
            confs.append(_trait_confidence(score, turns))
        elif abs(score) < 2.0 and trait in profile.active_traits:
            pass
    profile.active_traits = sorted(active)
    profile.confidence = max(confs) if confs else (max(all_confs) if all_confs else 0.0)
    profile.last_updated_turn = turn_no
    profile.last_sensitive_context = "professional" if classify_sensitive_context(user_text) else ""


def effective_profile(profile: InteractionProfile, user_text: str) -> EffectiveProfile:
    sensitive = classify_sensitive_context(user_text)
    ack = profile.ack_reframe_style
    sarcasm = profile.sarcasm_level
    snark = profile.snark_tolerance
    profanity_ok = bool(profile.profanity_ok)

    if sensitive and not profile.sensitive_override:
        ack = "sharp" if ack == "feral" else "plain"
        sarcasm = "low" if sarcasm in ("medium", "high") else sarcasm
        snark = "low"
        profanity_ok = False

    if ack == "feral" and not profanity_ok:
        ack = "sharp"

    return EffectiveProfile(
        correction_style=profile.correction_style,
        failure_tolerance=profile.failure_tolerance,
        ack_reframe_style=ack,
        sarcasm_level=sarcasm,
        snark_tolerance=snark,
        profanity_ok=profanity_ok,
        verbosity=profile.verbosity,
        sensitive_context=sensitive,
    )


def build_constraints_block(profile: InteractionProfile, user_text: str) -> str:
    eff = effective_profile(profile, user_text)
    return (
        "Interaction profile (style only; never alter factual grounding/provenance rules):\n"
        f"- correction_style={eff.correction_style}\n"
        f"- failure_tolerance={eff.failure_tolerance}\n"
        f"- ack_reframe_style={eff.ack_reframe_style}\n"
        f"- sarcasm_level={eff.sarcasm_level}\n"
        f"- snark_tolerance={eff.snark_tolerance}\n"
        f"- profanity_ok={str(eff.profanity_ok).lower()}\n"
        f"- verbosity={eff.verbosity}\n"
        "- Do not mirror or paraphrase the user's wording unless they explicitly ask for it.\n"
        "- Prioritize concise direct answers over meta-explanations about your own behavior."
    )


def render_profile_show(profile: InteractionProfile, *, enabled: bool, effective_strength: float = 0.0) -> str:
    eff = effective_profile(profile, "")
    lines = [
        "[profile]",
        f"enabled={str(enabled).lower()}",
        f"correction_style={profile.correction_style}",
        f"failure_tolerance={profile.failure_tolerance}",
    ]
    ack_line = f"ack_reframe_style={profile.ack_reframe_style}"
    if profile.ack_reframe_style != eff.ack_reframe_style:
        ack_line += f" (effective={eff.ack_reframe_style}; profanity_ok={str(profile.profanity_ok).lower()})"
    lines.extend(
        [
            ack_line,
            f"sarcasm_level={profile.sarcasm_level}",
            f"snark_tolerance={profile.snark_tolerance}",
            f"profanity_ok={str(profile.profanity_ok).lower()}",
            f"sensitive_override={str(profile.sensitive_override).lower()}",
            f"verbosity={profile.verbosity}",
            f"confidence={profile.confidence:.2f}",
            f"effective_strength={effective_strength:.2f}",
            f"active_traits={','.join(profile.active_traits)}",
            f"last_updated_turn={profile.last_updated_turn}",
        ]
    )
    return "\n".join(lines)


def has_non_default_style(profile: InteractionProfile) -> bool:
    if profile.correction_style != "neutral":
        return True
    if profile.sarcasm_level != "off":
        return True
    if profile.snark_tolerance != "low":
        return True
    if profile.profanity_ok:
        return True
    if profile.verbosity != "standard":
        return True
    if profile.sensitive_override:
        return True
    return False


def compute_effective_strength(
    profile: InteractionProfile,
    *,
    enabled: bool,
    output_compliance: float = 0.0,
) -> float:
    """Estimate how strongly profile constraints should be applied at runtime."""
    if not enabled:
        return 0.0
    trait_term = min(1.0, float(len(profile.active_traits)) / 3.0)
    style_term = 0.15 if has_non_default_style(profile) else 0.0
    compliance = _clamp(float(output_compliance or 0.0), 0.0, 1.0)
    out = (profile.confidence * 0.45) + (trait_term * 0.25) + (compliance * 0.30) + style_term
    return _clamp(out, 0.0, 1.0)
