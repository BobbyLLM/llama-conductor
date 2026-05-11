from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .encoder_server import EncoderServer
from .text_metrics import content_word_count

if TYPE_CHECKING:
    from .kaioken_vnext import KaiokenControlObject


logger = logging.getLogger(__name__)

_ECHO_THRESHOLD = 0.90
_REPEAT_SOFT_THRESHOLD = 0.75
_CONTRADICTION_THRESHOLD = 0.92  # provisional: NLI model not calibrated for conversational register; revisit when DeBERTa W3 AND-gate is built


def run_w2a(
    current_turn: str,
    recent_user_turns: List[str],
    kco: "KaiokenControlObject",
) -> None:
    """
    W2a session continuity lane.

    Computes echo/repeat and contradiction advisory signals against the
    rolling session window only. Fail-open on any encoder or state error.
    """
    current_turn_text = str(current_turn or "").strip()
    recent_turn_texts = [str(turn or "").strip() for turn in recent_user_turns or [] if str(turn or "").strip()]

    if current_turn_text.startswith(">>"):
        existing_outputs = getattr(kco, "advisor_outputs", None)
        if not isinstance(existing_outputs, dict):
            existing_outputs = {}
        else:
            existing_outputs = dict(existing_outputs)
        existing_outputs["w2a"] = {
            "echo_risk": False,
            "repeat_risk": 0,
            "contradiction_risk": False,
            "conflict_candidates": [],
            "protected_session_facts": [],
        }
        kco.advisor_outputs = existing_outputs
        return

    try:
        enc = EncoderServer.get_encoder()
        similarity_scores = enc.similarity(current_turn_text, recent_turn_texts) if recent_turn_texts else []
        max_similarity = max(similarity_scores) if similarity_scores else 0.0

        echo_risk = max_similarity >= _ECHO_THRESHOLD
        repeat_risk = 2 if echo_risk else 1 if max_similarity >= _REPEAT_SOFT_THRESHOLD else 0

        contradiction_risk = False
        conflict_candidates: List[Dict[str, Any]] = []

        current_turn_content_words = content_word_count(current_turn_text)
        nli_candidate_turns = [
            prior_turn
            for prior_turn in recent_turn_texts
            if content_word_count(prior_turn) >= 6
        ]

        # Only spend NLI calls if the current turn is at least somewhat related
        # to something in the session window.
        if recent_turn_texts and max_similarity >= 0.50 and current_turn_content_words >= 6:
            for prior_turn in nli_candidate_turns:
                nli_scores = enc.nli(prior_turn, current_turn_text)
                contradiction_score = float(nli_scores.get("contradiction", 0.0) or 0.0)
                if contradiction_score >= _CONTRADICTION_THRESHOLD:
                    contradiction_risk = True
                    conflict_candidates.append(
                        {
                            "prior_turn": prior_turn,
                            "contradiction_score": contradiction_score,
                        }
                    )

        w2a_payload = {
            "echo_risk": bool(echo_risk),
            "repeat_risk": int(repeat_risk),
            "contradiction_risk": bool(contradiction_risk),
            "conflict_candidates": conflict_candidates if contradiction_risk else [],
            "protected_session_facts": [],
        }

        existing_outputs = getattr(kco, "advisor_outputs", None)
        if not isinstance(existing_outputs, dict):
            existing_outputs = {}
        else:
            existing_outputs = dict(existing_outputs)
        existing_outputs["w2a"] = w2a_payload
        kco.advisor_outputs = existing_outputs
    except Exception as exc:
        logger.exception("W2a session continuity failed")
        return


__all__ = ["run_w2a"]
