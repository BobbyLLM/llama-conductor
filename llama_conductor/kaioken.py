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

_REFLECTIVE_SADNESS_EXEMPLARS = (
    "I keep thinking about how things used to be and it makes me sad.",
    "I'm getting older and it feels like time is moving too fast.",
    "Things are slipping away and I can't quite stop it.",
    "It feels less magical than it used to.",
    "I miss being young.",
    "There are fewer tomorrows ahead than there used to be.",
    "The world feels changed in a way I can't unsee.",
    "I feel nostalgic and a little sad about it.",
)
_REFLECTIVE_SADNESS_CONTROL_EXEMPLARS = (
    "I'm getting older and I need to make a better plan for the future.",
    "I'm getting older and I need to take my health more seriously.",
    "I'm getting older, which is fine.",
    "I'm getting older and I'm okay with it.",
    "Things are changing and that is normal.",
)
_REFLECTIVE_SADNESS_E5_CANDIDATE_K = 3
_REFLECTIVE_SADNESS_THRESHOLD = -4.0
_REFLECTIVE_SADNESS_MARGIN = 0.5

_CLARIFICATION_CONTINUITY_EXEMPLARS = (
    (
        "What do you mean?",
        "I mean the future still exists, but it is not the same endless open field it used to feel like.",
    ),
    (
        "No I mean what do you mean by that statement",
        "I mean the statement is still about the same idea, but with the corrected premise in view.",
    ),
    (
        "What are you saying?",
        "I mean the same claim from the last turn, not a new topic.",
    ),
    (
        "Can you explain that?",
        "I mean the comparison should stay on the same criteria unless the user explicitly changes them.",
    ),
    (
        "What do you mean?",
        "I mean the clarification is about the same statement, so the answer should stay anchored to it.",
    ),
)
_CLARIFICATION_CONTINUITY_CONTROL_EXEMPLARS = (
    (
        "What do you mean?",
        "Sorry, I lost the thread there - tell me again?",
    ),
    (
        "What do you mean?",
        "Can you restate the question more directly?",
    ),
    (
        "What are you saying?",
        "I do not know what you are asking.",
    ),
    (
        "No I mean what do you mean by that statement",
        "Let us move on to a different topic.",
    ),
    (
        "Can you explain that?",
        "I need more context before I can answer.",
    ),
)
_CLARIFICATION_CONTINUITY_E5_CANDIDATE_K = 3
_CLARIFICATION_CONTINUITY_THRESHOLD = -4.0
_CLARIFICATION_CONTINUITY_MARGIN = 0.5

_EMOTIONAL_CONTINUITY_EXEMPLARS = (
    (
        "reflective_sadness",
        "I feel melancholy",
        "Yeah, that's a real thing. I wonder what's actually weighing on you right now.",
    ),
    (
        "reflective_sadness",
        "I'm getting older and the things and places that shaped me as a kid are all slipping away",
        "Old memories don't disappear, they just stop being the only version of the story you know.",
    ),
    (
        "reflective_sadness",
        "I miss being young and the endless possibility of tomorrow",
        "Yeah, that loss of open-ended tomorrow is real.",
    ),
    (
        "sadness_clarification",
        "What do you mean?",
        "I mean the future used to feel wide open, and now it feels smaller and harder to ignore.",
    ),
    (
        "sadness_clarification",
        "No I mean what do you mean by that statement",
        "I mean the feeling underneath it: the loss of open-ended future space.",
    ),
    (
        "sadness_clarification",
        "I mean about my sadness",
        "I mean the sadness itself, not a new topic.",
    ),
    (
        "emotional_clarification",
        "I mean about my sadness",
        "I mean your sadness is not a glitch. It's the old code running again.",
    ),
    (
        "emotional_clarification",
        "No I mean what do you mean by that statement",
        "I mean the same emotional thread, not a new problem.",
    ),
    (
        "emotional_clarification",
        "What do you mean?",
        "I mean the feeling underneath what you just said.",
    ),
    (
        "advice_request",
        "What should I do?",
        "Start with one small thing you can do right now.",
    ),
    (
        "advice_request",
        "What do I do about this?",
        "Take one small step first.",
    ),
    (
        "advice_request",
        "Any advice?",
        "We can start with one concrete thing.",
    ),
    (
        "topic_shift",
        "Can we move on?",
        "Sure, let's switch gears.",
    ),
    (
        "topic_shift",
        "Let's talk about something else",
        "Yep, we can change topics.",
    ),
    (
        "topic_shift",
        "New question",
        "Okay, what's the new topic?",
    ),
)
_EMOTIONAL_CONTINUITY_CONTROL_EXEMPLARS = (
    (
        "neutral",
        "I'm getting older and I'm okay with it",
        "That's fair.",
    ),
    (
        "neutral",
        "Things are changing and that is normal",
        "Yep, that happens.",
    ),
    (
        "neutral",
        "I am fine and just checking in",
        "Sure.",
    ),
    (
        "neutral",
        "What do you mean?",
        "Sorry, I lost the thread there - tell me again?",
    ),
    (
        "neutral",
        "What should I do?",
        "Can you clarify your question?",
    ),
    (
        "neutral",
        "I feel sad",
        "Okay.",
    ),
)
_EMOTIONAL_CONTINUITY_E5_CANDIDATE_K = 4
_EMOTIONAL_CONTINUITY_THRESHOLD = -4.0
_EMOTIONAL_CONTINUITY_MARGIN = 0.5

_EMOTIONAL_REPEAT_EXEMPLARS = (
    (
        "repeat",
        "user: What do you mean?\nprior: I mean the future used to feel like an open road, and now it feels narrower.\ndraft: I mean the future used to feel like an open road, and now it feels narrower.",
    ),
    (
        "repeat",
        "user: No I mean what do you mean by that statement\nprior: I mean the same emotional thread, just clearer.\ndraft: I mean the same emotional thread, just clearer.",
    ),
    (
        "repeat",
        "user: I mean about my sadness\nprior: I mean the sadness is real and tied to what changed.\ndraft: I mean the sadness is real and tied to what changed.",
    ),
    (
        "tighten",
        "user: What do you mean?\nprior: I mean the future used to feel like an open road, and now it feels narrower.\ndraft: It means the future feels smaller now, so the loss is about space and possibility, not just time.",
    ),
    (
        "tighten",
        "user: No I mean what do you mean by that statement\nprior: I mean the same emotional thread, just clearer.\ndraft: I mean the same thread more plainly: the feeling is real, and it comes from how the future has changed.",
    ),
    (
        "tighten",
        "user: I mean about my sadness\nprior: I mean the sadness is real and tied to what changed.\ndraft: I mean the sadness itself, not a new problem.",
    ),
)
_EMOTIONAL_REPEAT_CONTROL_EXEMPLARS = (
    (
        "neutral",
        "user: What do you mean?\nprior: I mean the future used to feel like an open road.\ndraft: Sorry, I lost the thread there - tell me again?",
    ),
    (
        "neutral",
        "user: I mean about my sadness\nprior: I mean the sadness is real.\ndraft: Okay.",
    ),
    (
        "neutral",
        "user: No I mean what do you mean by that statement\nprior: I mean the same emotional thread.\ndraft: Let us move on to a different topic.",
    ),
)
_EMOTIONAL_REPEAT_E5_CANDIDATE_K = 3
_EMOTIONAL_REPEAT_THRESHOLD = -4.0
_EMOTIONAL_REPEAT_MARGIN = 0.5
_EMOTIONAL_REPEAT_COSINE_THRESHOLD = 0.90

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


def _semantic_reflective_sadness_hits(text: str, f: Dict[str, Any]) -> int:
    """Detect reflective / nostalgic sadness via e5 + TinyBERT, not regex soup."""
    low = str(text or "").strip()
    if not low:
        return 0
    if int(f.get("distress_hint_hits", 0) or 0) > 0:
        return 0
    if not (
        int(f.get("first_person_hits", 0) or 0) > 0
        or int(f.get("disclosure_markers", 0) or 0) > 0
        or int(f.get("personal_state_hits", 0) or 0) > 0
        or int(f.get("personal_hits", 0) or 0) > 0
    ):
        return 0
    if (
        int(f.get("working_hits", 0) or 0) > 0
        or int(f.get("directive_hits", 0) or 0) > 0
        or int(f.get("directive_pattern_hits", 0) or 0) > 0
        or bool(f.get("imperative_like", False))
    ):
        return 0

    try:
        from .rag import get_embed_model, get_rerank_model
    except Exception:
        return 0

    try:
        embed_model = get_embed_model()
        rerank_model = get_rerank_model()
    except Exception:
        return 0
    if rerank_model is None:
        return 0

    exemplar_embeddings = embed_model.encode(
        [f"passage: {s}" for s in _REFLECTIVE_SADNESS_EXEMPLARS],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    control_embeddings = embed_model.encode(
        [f"passage: {s}" for s in _REFLECTIVE_SADNESS_CONTROL_EXEMPLARS],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    text_embedding = embed_model.encode(
        f"query: {low}",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    try:
        cosine_scores = exemplar_embeddings @ text_embedding
    except Exception:
        return 0

    top_k_indices = list(reversed(cosine_scores.argsort()[-_REFLECTIVE_SADNESS_E5_CANDIDATE_K:]))
    pairs = [(low, _REFLECTIVE_SADNESS_EXEMPLARS[i]) for i in top_k_indices]
    control_scores = []
    try:
        control_cosine_scores = control_embeddings @ text_embedding
        control_top_k_indices = list(
            reversed(control_cosine_scores.argsort()[-_REFLECTIVE_SADNESS_E5_CANDIDATE_K:])
        )
        control_pairs = [(low, _REFLECTIVE_SADNESS_CONTROL_EXEMPLARS[i]) for i in control_top_k_indices]
        control_scores = rerank_model.predict(control_pairs)
    except Exception:
        control_scores = []
    try:
        scores = rerank_model.predict(pairs)
    except Exception:
        return 0
    if scores is None:
        return 0
    try:
        scores = list(scores)
    except TypeError:
        scores = [scores]
    if len(scores) == 0:
        return 0
    if control_scores is None:
        control_scores = []
    else:
        try:
            control_scores = list(control_scores)
        except TypeError:
            control_scores = [control_scores]
    try:
        best_score = float(max(scores))
        control_best = float(max(control_scores)) if len(control_scores) > 0 else 0.0
    except Exception:
        return 0
    if best_score < _REFLECTIVE_SADNESS_THRESHOLD:
        return 0
    if (best_score - control_best) < _REFLECTIVE_SADNESS_MARGIN:
        return 0
    return 1


def _semantic_clarification_continuity_hits(user_text: str, prior_assistant_text: str) -> int:
    """Detect same-thread clarification continuity via e5 + TinyBERT.

    Step 1 helper only: keep the output semantic and additive, but do not
    wire routing/fallback decisions here yet.
    """
    user = str(user_text or "").strip()
    prior = str(prior_assistant_text or "").strip()
    if not user or not prior:
        return 0

    try:
        from .rag import get_embed_model, get_rerank_model
    except Exception:
        return 0

    try:
        embed_model = get_embed_model()
        rerank_model = get_rerank_model()
    except Exception:
        return 0
    if rerank_model is None:
        return 0

    probe = f"user: {user}\nassistant: {prior}"
    exemplar_texts = [
        f"user: {u}\nassistant: {a}"
        for (u, a) in _CLARIFICATION_CONTINUITY_EXEMPLARS
    ]
    control_texts = [
        f"user: {u}\nassistant: {a}"
        for (u, a) in _CLARIFICATION_CONTINUITY_CONTROL_EXEMPLARS
    ]

    exemplar_embeddings = embed_model.encode(
        [f"passage: {s}" for s in exemplar_texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    control_embeddings = embed_model.encode(
        [f"passage: {s}" for s in control_texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    probe_embedding = embed_model.encode(
        f"query: {probe}",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    try:
        exemplar_cosine_scores = exemplar_embeddings @ probe_embedding
    except Exception:
        return 0

    top_k_indices = list(
        reversed(exemplar_cosine_scores.argsort()[-_CLARIFICATION_CONTINUITY_E5_CANDIDATE_K:])
    )
    candidate_pairs = [(probe, exemplar_texts[i]) for i in top_k_indices]

    control_scores = []
    try:
        control_cosine_scores = control_embeddings @ probe_embedding
        control_top_k_indices = list(
            reversed(control_cosine_scores.argsort()[-_CLARIFICATION_CONTINUITY_E5_CANDIDATE_K:])
        )
        control_pairs = [(probe, control_texts[i]) for i in control_top_k_indices]
        control_scores = rerank_model.predict(control_pairs)
    except Exception:
        control_scores = []

    try:
        scores = rerank_model.predict(candidate_pairs)
    except Exception:
        return 0
    if scores is None:
        return 0
    try:
        scores = list(scores)
    except TypeError:
        scores = [scores]
    if len(scores) == 0:
        return 0
    if control_scores is None:
        control_scores = []
    else:
        try:
            control_scores = list(control_scores)
        except TypeError:
            control_scores = [control_scores]

    try:
        best_score = float(max(scores))
        control_best = float(max(control_scores)) if len(control_scores) > 0 else 0.0
    except Exception:
        return 0
    if best_score < _CLARIFICATION_CONTINUITY_THRESHOLD:
        prior_low = prior.lower()
        prior_tokens = len(prior.split())
        repair_like = bool(
            re.search(
                r"\b(lost the thread|tell me again|what am i missing here|i do not know what you are asking|can you restate)\b",
                prior_low,
            )
        )
        if prior_tokens >= 8 and not repair_like:
            return 1
        return 0
    if (best_score - control_best) < _CLARIFICATION_CONTINUITY_MARGIN:
        prior_low = prior.lower()
        prior_tokens = len(prior.split())
        repair_like = bool(
            re.search(
                r"\b(lost the thread|tell me again|what am i missing here|i do not know what you are asking|can you restate)\b",
                prior_low,
            )
        )
        if prior_tokens >= 8 and not repair_like:
            return 1
        return 0
    return 1


def _semantic_emotional_continuity_label(
    user_text: str,
    prior_assistant_text: str,
    prior_user_text: str = "",
) -> str:
    """Classify emotional continuity with a narrow semantic label.

    Labels:
    - reflective_sadness
    - sadness_clarification
    - emotional_clarification
    - advice_request
    - topic_shift
    - neutral
    """
    user = str(user_text or "").strip()
    prior = str(prior_assistant_text or "").strip()
    prev_user = str(prior_user_text or "").strip()
    if not user or not prior:
        return "neutral"

    try:
        from .rag import get_embed_model, get_rerank_model
    except Exception:
        return "neutral"

    try:
        embed_model = get_embed_model()
        rerank_model = get_rerank_model()
    except Exception:
        return "neutral"
    if rerank_model is None:
        return "neutral"

    user_low = user.lower()
    emotional_seed = (
        int(_count_markers(user_low, ("sad", "sadness", "melancholy", "nostalg", "young", "tomorrow", "magical"))) > 0
        or int(_count_markers(user_low, ("what should i do", "what do i do", "any advice", "need advice"))) > 0
        or bool(re.search(r"\b(what do you mean|what do you mean by that statement|i mean about my sadness)\b", user_low))
        or bool(re.search(r"\b(move on|different topic|new question|switch gears)\b", user_low))
        or bool(
            int(_count_markers(user_low, ("i", "i'm", "im", "my", "mine"))) > 0
            and int(_count_markers(user_low, ("feel", "missing", "slipping", "less magical", "fewer tomorrows", "older"))) > 0
        )
    )
    if not emotional_seed:
        return "neutral"

    probe = f"user: {user}\nassistant: {prior}"
    if prev_user:
        probe = f"{probe}\nprev_user: {prev_user}"

    exemplar_texts = [
        (label, f"user: {u}\nassistant: {a}")
        for (label, u, a) in _EMOTIONAL_CONTINUITY_EXEMPLARS
    ]
    control_texts = [
        f"user: {u}\nassistant: {a}"
        for (_label, u, a) in _EMOTIONAL_CONTINUITY_CONTROL_EXEMPLARS
    ]

    try:
        exemplar_embeddings = embed_model.encode(
            [f"passage: {s}" for (_label, s) in exemplar_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        control_embeddings = embed_model.encode(
            [f"passage: {s}" for s in control_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        probe_embedding = embed_model.encode(
            f"query: {probe}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception:
        return "neutral"

    try:
        exemplar_cosine_scores = exemplar_embeddings @ probe_embedding
    except Exception:
        return "neutral"

    top_k_indices = list(reversed(exemplar_cosine_scores.argsort()[-_EMOTIONAL_CONTINUITY_E5_CANDIDATE_K:]))
    candidate_pairs = [(probe, exemplar_texts[i][1]) for i in top_k_indices]
    candidate_labels = [exemplar_texts[i][0] for i in top_k_indices]

    control_scores = []
    try:
        control_cosine_scores = control_embeddings @ probe_embedding
        control_top_k_indices = list(reversed(control_cosine_scores.argsort()[-_EMOTIONAL_CONTINUITY_E5_CANDIDATE_K:]))
        control_pairs = [(probe, control_texts[i]) for i in control_top_k_indices]
        control_scores = rerank_model.predict(control_pairs)
    except Exception:
        control_scores = []

    try:
        scores = rerank_model.predict(candidate_pairs)
    except Exception:
        return "neutral"
    if scores is None:
        return "neutral"
    try:
        scores = list(scores)
    except TypeError:
        scores = [scores]
    if len(scores) == 0:
        return "neutral"
    if control_scores is None:
        control_scores = []
    else:
        try:
            control_scores = list(control_scores)
        except TypeError:
            control_scores = [control_scores]

    try:
        best_idx = int(max(range(len(scores)), key=lambda i: float(scores[i])))
        best_score = float(scores[best_idx])
        control_best = float(max(control_scores)) if len(control_scores) > 0 else 0.0
        label = str(candidate_labels[best_idx] or "neutral").strip().lower()
    except Exception:
        return "neutral"
    if best_score < _EMOTIONAL_CONTINUITY_THRESHOLD:
        return "neutral"
    if (best_score - control_best) < _EMOTIONAL_CONTINUITY_MARGIN:
        return "neutral"
    if label == "emotional_clarification":
        same_thread_clarify = bool(
            re.search(r"\b(what do you mean|what do you mean by that statement|i mean about my sadness)\b", user_low)
        )
        sadness_context = bool(
            _count_markers(user_low, ("sad", "sadness", "melancholy", "nostalg", "young", "tomorrow", "magical", "older")) > 0
            or _count_markers(prior.lower(), ("sad", "sadness", "melancholy", "nostalg", "young", "tomorrow", "magical")) > 0
        )
        if same_thread_clarify and sadness_context:
            label = "sadness_clarification"
    elif label == "neutral":
        same_thread_clarify = bool(
            re.search(r"\b(what do you mean|what do you mean by that statement|i mean about my sadness)\b", user_low)
        )
        sadness_context = bool(
            _count_markers(user_low, ("sad", "sadness", "melancholy", "nostalg", "young", "tomorrow", "magical", "older")) > 0
            or _count_markers(prior.lower(), ("sad", "sadness", "melancholy", "nostalg", "young", "tomorrow", "magical", "older")) > 0
        )
        if same_thread_clarify and sadness_context:
            label = "sadness_clarification"
    return label if label in {"reflective_sadness", "sadness_clarification", "emotional_clarification", "advice_request", "topic_shift"} else "neutral"


def _semantic_emotional_repeat_risk(
    user_text: str,
    prior_assistant_text: str,
    draft_text: str,
    continuity_label: str = "",
) -> int:
    """Detect repeated emotional clarification phrasing versus a tighter clarification rewrite."""
    label = str(continuity_label or "").strip().lower()
    if label not in {"sadness_clarification", "emotional_clarification"}:
        return 0
    user = str(user_text or "").strip()
    prior = str(prior_assistant_text or "").strip()
    draft = str(draft_text or "").strip()
    if not user or not prior or not draft:
        return 0

    try:
        from .rag import get_embed_model, get_rerank_model
    except Exception:
        return 0

    try:
        embed_model = get_embed_model()
        rerank_model = get_rerank_model()
    except Exception:
        return 0
    if rerank_model is None:
        return 0

    probe = f"user: {user}\nprior: {prior}\ndraft: {draft}\nlabel: {label}"
    repeat_texts = [s for (_label, s) in _EMOTIONAL_REPEAT_EXEMPLARS if _label == "repeat"]
    tighten_texts = [s for (_label, s) in _EMOTIONAL_REPEAT_EXEMPLARS if _label == "tighten"]
    control_texts = [s for (_label, s) in _EMOTIONAL_REPEAT_CONTROL_EXEMPLARS]

    try:
        repeat_embeddings = embed_model.encode(
            [f"passage: {s}" for s in repeat_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        tighten_embeddings = embed_model.encode(
            [f"passage: {s}" for s in tighten_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        control_embeddings = embed_model.encode(
            [f"passage: {s}" for s in control_texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        probe_embedding = embed_model.encode(
            f"query: {probe}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception:
        return 0

    try:
        prior_draft_embeddings = embed_model.encode(
            [f"passage: {prior}", f"passage: {draft}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        direct_cosine = float(prior_draft_embeddings[0] @ prior_draft_embeddings[1])
    except Exception:
        direct_cosine = 0.0

    if label == "sadness_clarification" and direct_cosine >= _EMOTIONAL_REPEAT_COSINE_THRESHOLD:
        return 1

    try:
        repeat_cosine_scores = repeat_embeddings @ probe_embedding
        tighten_cosine_scores = tighten_embeddings @ probe_embedding
        control_cosine_scores = control_embeddings @ probe_embedding
    except Exception:
        return 0

    repeat_top = list(reversed(repeat_cosine_scores.argsort()[-_EMOTIONAL_REPEAT_E5_CANDIDATE_K:]))
    tighten_top = list(reversed(tighten_cosine_scores.argsort()[-_EMOTIONAL_REPEAT_E5_CANDIDATE_K:]))
    control_top = list(reversed(control_cosine_scores.argsort()[-_EMOTIONAL_REPEAT_E5_CANDIDATE_K:]))

    candidate_pairs = [(probe, repeat_texts[i]) for i in repeat_top] + [(probe, tighten_texts[i]) for i in tighten_top]
    candidate_labels = ["repeat"] * len(repeat_top) + ["tighten"] * len(tighten_top)

    try:
        control_pairs = [(probe, control_texts[i]) for i in control_top]
        control_scores = rerank_model.predict(control_pairs)
    except Exception:
        control_scores = []

    try:
        scores = rerank_model.predict(candidate_pairs)
    except Exception:
        return 0
    if scores is None:
        return 0
    try:
        scores = list(scores)
    except TypeError:
        scores = [scores]
    if len(scores) == 0:
        return 0
    if control_scores is None:
        control_scores = []
    else:
        try:
            control_scores = list(control_scores)
        except TypeError:
            control_scores = [control_scores]

    try:
        best_idx = int(max(range(len(scores)), key=lambda i: float(scores[i])))
        best_score = float(scores[best_idx])
        control_best = float(max(control_scores)) if len(control_scores) > 0 else 0.0
        label_best = str(candidate_labels[best_idx] or "neutral").strip().lower()
    except Exception:
        return 0
    if best_score < _EMOTIONAL_REPEAT_THRESHOLD:
        return 0
    if (best_score - control_best) < _EMOTIONAL_REPEAT_MARGIN:
        return 0
    if label_best == "repeat":
        return 1
    return 1 if label == "sadness_clarification" and direct_cosine >= (_EMOTIONAL_REPEAT_COSINE_THRESHOLD - 0.02) else 0


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
        or f.get("reflective_sadness_hits", 0) > 0
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
    reflective_sadness_hits = _semantic_reflective_sadness_hits(
        low,
        {
            "first_person_hits": first_person_hits,
            "disclosure_markers": disclosure_markers,
            "personal_state_hits": personal_state_hits,
            "personal_hits": personal_hits,
            "distress_hint_hits": distress_hint_hits,
        },
    )
    if reflective_sadness_hits > 0:
        distress_hint_hits += reflective_sadness_hits

    playful_signal = (2 * playful_strong) + playful_light
    if frustration_markers > 0 and playful_strong == 0:
        playful_signal = max(0, playful_signal - frustration_markers)

    return {
        "char_count": len(str(text or "")),
        "token_count": len(toks),
        "greeting_hit": bool(any(t in _CASUAL_GREETINGS for t in tok_set)),
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
        "reflective_sadness_hits": reflective_sadness_hits,
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
        _greeting_extended = (
            f["greeting_hit"]
            and f["token_count"] <= 8
            and f["working_hits"] == 0
            and f["directive_hits"] == 0
        )
        is_low_signal_casual = bool(
            f["greeting_short"]
            or _greeting_extended
            or (
                f["first_person_hits"] > 0
                and f["working_hits"] == 0
                and f["personal_state_hits"] == 0
                and not f["imperative_like"]
                and f["directive_hits"] == 0
            )
        )
        top_label = "casual" if is_low_signal_casual else "working"
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
        extra_dict = dict(extra)
        row["outcome"] = extra_dict
        row["emotional_continuity_label"] = str(extra_dict.get("emotional_continuity_label", "neutral") or "neutral")
    else:
        row["emotional_continuity_label"] = "neutral"

    out_dir = _default_kaioken_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_session_filename(session_id)}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return row
