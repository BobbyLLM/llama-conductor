"""KAIOKEN turn-intent classification helpers."""

from __future__ import annotations

import re

_KAIOKEN_CONTINUATION_RE = re.compile(
    r"^\s*(go on|and\??|continue|carry on|keep going|more|\.{2,})\s*$",
    re.IGNORECASE,
)
_KAIOKEN_CLARIFY_RE = re.compile(
    r"^\s*(?:"
    r"(?:sorry[,?.!]*\s*)?(?:um+|uh+)?[.\s]*what\??|"
    r"(?:sorry[,?.!]*\s*)?(?:what are you saying|what do you mean|lost me)\??|"
    r"(?:no\s+i\s+mean\s+)?(?:what do you mean)(?:\s+by\s+[^?]{1,120})?\??|"
    r"(?:sorry[,?.!]*\s*)?what(?:'s| is)\s+[^?]{1,160}\??|"
    r"(?:keeping|staying)\s+what\s+direct\??|"
    r"what(?:'s| is)\s+direct\??"
    r")\s*$",
    re.IGNORECASE,
)
_KAIOKEN_TOPIC_SWITCH_RE = re.compile(
    r"\b(something else on my mind|talk about something else|change subject|different topic|new topic|move on|let's move on|lets move on)\b",
    re.IGNORECASE,
)
_KAIOKEN_TOPIC_RESOLVED_RE = re.compile(
    r"\b(already home|home now|can wait till morning|it can wait|no dramas|all good(?: now)?|resolved|sorted)\b",
    re.IGNORECASE,
)


def _is_continuation_prompt(text: str) -> bool:
    return bool(_KAIOKEN_CONTINUATION_RE.search(str(text or "").strip()))


def _is_clarification_prompt(text: str) -> bool:
    return bool(_KAIOKEN_CLARIFY_RE.search(str(text or "").strip()))


def _is_topic_switch_prompt(text: str) -> bool:
    return bool(_KAIOKEN_TOPIC_SWITCH_RE.search(str(text or "")))


def _is_resolution_signal(text: str) -> bool:
    return bool(_KAIOKEN_TOPIC_RESOLVED_RE.search(str(text or "")))


def _user_explicitly_requests_advice(user_text: str) -> bool:
    t = str(user_text or "").lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\b("
            r"what should i|should i|how do i|how can i|can you help|"
            r"give me advice|advise me|what do i do|what next|how to|fix this|improve this|"
            r"(?:should|do)\s+i\s+(?:quit|stop)|"
            r"(?:i|maybe i)\s+should\s+(?:quit|stop)"
            r")\b",
            t,
        )
    )


def _user_explicitly_requests_encouragement(user_text: str) -> bool:
    t = str(user_text or "").lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\b("
            r"encouragement|words of wisdom|give me hope|need a boost|"
            r"say something encouraging|can you encourage me|reassure me"
            r")\b",
            t,
            flags=re.IGNORECASE,
        )
    )
