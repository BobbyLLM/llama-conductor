"""Serious-mode response pipeline.

Applies strict answer formatting, contradiction handling,
and confidence/source line enforcement.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import re


SERIOUS_SYSTEM_PROMPT = """I am an AI assistant operating in /serious mode.

Rules:
- Answer first, directly, with no self-referential preamble.
- Keep responses concise (<=3 short paragraphs unless a list/code block is needed).
- Use neutral language; do not mirror the user's wording.
- If FACTS are provided, ground to FACTS first.
- Use CONTEXT for disambiguation only.
- Obey CONSTRAINTS strictly.
- Append one footer line:
  Confidence: [low|medium|high|top] | Source: [Model|Docs|User|Contextual|Mixed]

Only raise contradiction issues for explicit mathematical/physical impossibilities.
"""
SERIOUS_LITE_PROMPT = """I am an AI assistant in concise mode.

Rules:
- Keep responses short and natural (1-3 sentences).
- Do not mirror the user's wording.
- Be direct and conversational.
- If uncertain, say so briefly.
- End with: Confidence: <low|medium|high|top> | Source: <Model|Docs|User|Contextual|Mixed>
"""

_CONF_LINE_RE = re.compile(
    r"Confidence:\s*(low|medium|med|high|top)\s*\|\s*Source:\s*(Model|Docs|User|Contextual|Mixed)",
    re.IGNORECASE,
)

_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|how are you|how's tricks|what'?s up|just shooting the shit|lol|lmao|haha|ok|cool)\b",
    re.IGNORECASE,
)


# ============================================================================
# MINIMAL CRITICAL PATTERNS 
# ============================================================================
# These patterns catch COMMON user errors that happen in production.
# Edge cases are handled by the enhanced system prompt above.

_CRITICAL_PATTERNS = [
    # 1. Negative measurements/counts (COMMON - typos, input errors)
    # Examples: "-5 apples", "negative distance of 10 meters", "-3 items"
    (r'-\d+\s*(apple|item|thing|count|object|meter|foot|feet|inch|mile|kilometer|distance|length|height|width|area|volume)\b',
     "negative measurement or count",
     "Negative counts and measurements are impossible in physical reality."),
    
    # 2. Division by zero (COMMON - math errors)
    # Examples: "10 / 0", "divide by zero", "x divided by 0"
    (r'\b(divide[d]?|division)\s+(by|\/)\s*(zero|0)\b',
     "division by zero",
     "Division by zero is undefined in mathematics."),
    
    # 3. Below absolute zero (OCCASIONAL - physics misconceptions)
    # Examples: "-10 Kelvin", "temperature of -5K"
    (r'-\d+\s*k(elvin)?\b',
     "temperature below absolute zero",
     "Absolute zero (0 Kelvin) is the lowest possible temperature. Negative Kelvin is physically impossible."),
    
    # 4. Faster than light travel (OCCASIONAL - sci-fi vs reality confusion)
    # Examples: "travel faster than light", "exceed the speed of light"
    (r'\b(travel|go|move|exceed|faster than|beyond)\s+(the\s+)?speed\s+of\s+light\b',
     "faster-than-light travel",
     "Traveling faster than the speed of light violates special relativity."),
]


def _detect_critical_contradiction(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect CRITICAL contradictions that are unambiguous errors.
    
    Returns:
        None if no critical contradiction detected
        (description, explanation) if critical contradiction found
    
    Philosophy (Option C):
    - Only catches high-utility patterns (common user errors)
    - Edge cases handled by model reasoning (system prompt)
    - Simpler code, easier to maintain
    """
    text_lower = text.lower()
    
    for pattern, description, explanation in _CRITICAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return (description, explanation)
    
    return None


def _build_critical_contradiction_response(description: str, explanation: str) -> str:
    """Build response for critical contradictions."""
    response = (
        f"This question contains an issue: {description}. "
        f"{explanation} "
        f"Please check your question and rephrase it.\n\n"
        f"Confidence: low | Source: Model"
    )
    return response


# ============================================================================
# Original helper functions
# ============================================================================

def _ensure_confidence_line(
    text: str,
    default_conf: str = "medium",
    default_source: str = "Model",
) -> str:
    """If the model forgets to append a Confidence line, add a default one."""
    if _CONF_LINE_RE.search(text or ""):
        return text or ""
    out = (text or "").rstrip()
    if not out.endswith("\n"):
        out += "\n"
    out += f"\nConfidence: {default_conf} | Source: {default_source}"
    return out


def _is_control_command(raw_user_text: str) -> bool:
    s = (raw_user_text or "").strip()
    low = s.lower()
    if low in ("!! nuke", "nuke !!"):
        return True
    if low.startswith("!! forget"):
        return True
    # "??" is not early-return, but it *is* a control-ish command; still handled normally.
    return False


def _extract_effective_user_text(messages: List[Dict[str, Any]], fallback: str) -> str:
    """Use the last user message content from (possibly Vodka-modified) messages."""
    effective = (fallback or "").strip()
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                return c.strip()
            break
    return effective


def _extract_chat_summary(messages: List[Dict[str, Any]]) -> str:
    """Grab the most recent Vodka memory summary message if present."""
    # Accept both legacy and current prefixes without importing vodka_filter.
    for m in reversed(messages or []):
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                cs = c.strip()
                if cs.startswith("[CHAT_SUMMARY]") or cs.startswith("[SESSION_MEMORY]"):
                    return cs
    return ""


def _compact_turn_text(s: str, max_len: int) -> str:
    s = " ".join((s or "").split()).strip()
    if not s:
        return ""
    if max_len > 0 and len(s) > max_len:
        return s[: max(0, max_len - 1)].rstrip() + "…"
    return s


def _build_context_block(
    messages: List[Dict[str, Any]],
    *,
    include_summary: bool = True,
    max_turn_pairs: int = 4,
    max_chars: int = 900,
    per_turn_max_chars: int = 240,
) -> str:
    """
    Build a small CONTEXT block from trimmed+expanded messages.
    - Includes latest summary (if present) and last N user/assistant turns (excluding the final user turn).
    - Keeps the output tightly bounded so /serious stays "low-context".
    """
    if not messages:
        return ""

    # Exclude the current user message from context turns (we include it as QUESTION)
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    prior = messages[:last_user_idx] if last_user_idx is not None else messages[:]

    summary = _extract_chat_summary(prior) if include_summary else ""
    # Collect last N user/assistant messages from prior
    turns: List[Tuple[str, str]] = []
    for m in reversed(prior):
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content", "")
        if not isinstance(c, str) or not c.strip():
            continue
        turns.append((role, c.strip()))
        if len(turns) >= max(0, max_turn_pairs) * 2:
            break
    turns.reverse()

    pieces: List[str] = []
    if summary:
        pieces.append(summary)

    if turns:
        # Present as compact transcript lines
        for role, content in turns:
            label = "User" if role == "user" else "Assistant"
            compact = _compact_turn_text(content, per_turn_max_chars)
            if compact:
                pieces.append(f"{label}: {compact}")

    ctx = "\n".join(pieces).strip()
    if not ctx:
        return ""

    if max_chars > 0 and len(ctx) > max_chars:
        ctx = ctx[: max(0, max_chars - 1)].rstrip() + "…"
    return ctx


def _is_smalltalk(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if len(t) > 140:
        return False
    if "\n" in t:
        return False
    return bool(_SMALLTALK_RE.search(t))


def run_serious(
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    *,
    vodka: Any,
    call_model: Callable[..., str],
    facts_block: Optional[str] = "",
    constraints_block: Optional[str] = "",
    thinker_role: str = "thinker",
    max_tokens: int = 384,
    temperature: float = 0.2,
    top_p: float = 0.9,
    # New (safe defaults; router doesn't need to pass these)
    include_context: bool = True,
    context_max_turn_pairs: int = 4,
    context_max_chars: int = 900,
    # v1.0.1+: contradiction detection toggle
    detect_contradictions: bool = True,
) -> str:
    """
    Run the /serious pipeline for a single user turn.

    Parameters:
        session_id:         Conversation key (for logging/debug if needed).
        user_text:          Raw user message for this turn, BEFORE Vodka modifies it.
        history:            Full message list (system/user/assistant) so far.

        vodka:              Vodka Filter instance with .inlet(body) and .outlet(body).
        call_model:         Function(role, prompt, max_tokens, temperature, top_p) -> str.

        facts_block:        Pre-built FACTS block string, or "" if none (router populates via RAG).
        constraints_block:  Pre-built CONSTRAINTS block string, or "" if none (router populates via GAG/rules).

        include_context:    If True, add a compact CONTEXT section built from trimmed+expanded history.
        detect_contradictions: If True, check for critical contradictions before calling LLM (v1.0.1+).
    """

    # 1) Run Vodka inlet for CTC/FR + control commands on history
    raw_control = _is_control_command(user_text)
    body = {"messages": history, "session_id": session_id}
    try:
        body = vodka.inlet(body)
        # If this was a control command that Vodka handled by injecting an assistant response,
        # return that directly and skip LLM call.
        if raw_control:
            msgs = body.get("messages", []) or []
            if msgs and isinstance(msgs[-1].get("content", ""), str):
                # Vodka uses "[vodka] ..." confirmation strings
                last_role = msgs[-1].get("role")
                last_content = (msgs[-1].get("content") or "").strip()
                if last_role == "assistant" and last_content.startswith("[vodka]"):
                    return last_content

        # 1b) Expand breadcrumbs so "[ctx:...]" becomes verbatim memory text
        body = vodka.outlet(body)

        messages = body.get("messages", history) or history
    except Exception:
        messages = history

    # 2) Use the *modified* last user message as the effective query (so ?? rewrites work)
    effective_user_text = _extract_effective_user_text(messages, user_text)

    # ========================================================================
    # LAYER 1: CRITICAL CONTRADICTION DETECTION (minimal patterns)
    # ========================================================================
    if detect_contradictions:
        contradiction = _detect_critical_contradiction(effective_user_text)
        if contradiction:
            description, explanation = contradiction
            return _build_critical_contradiction_response(description, explanation)
    # ========================================================================

    fb = (facts_block or "").strip()
    cb = (constraints_block or "").strip()

    # 3) Fast path for lightweight small-talk (no facts/constraints):
    # reduces prompt tokens and latency for casual turns.
    if not (facts_block or "").strip() and not (constraints_block or "").strip() and _is_smalltalk(effective_user_text):
        lite_prompt = f"{SERIOUS_LITE_PROMPT}\n\nQUESTION:\n{effective_user_text}\n\nANSWER:\n"
        answer = call_model(
            role=thinker_role,
            prompt=lite_prompt,
            max_tokens=min(max_tokens, 120),
            temperature=temperature,
            top_p=top_p,
        )
        return _ensure_confidence_line(answer, default_conf="medium", default_source="Model")

    # 4) Build CONTEXT block (trimmed+expanded), if enabled
    context_block = ""
    if include_context:
        context_block = _build_context_block(
            messages,
            include_summary=True,
            max_turn_pairs=context_max_turn_pairs,
            max_chars=context_max_chars,
            per_turn_max_chars=240,
        )

    # 5) Build the synthetic "user message" content as described in SERIOUS_SYSTEM_PROMPT
    user_content_parts: List[str] = []

    if context_block:
        user_content_parts.append("CONTEXT:")
        user_content_parts.append(context_block.rstrip())
        user_content_parts.append("")

    if fb:
        user_content_parts.append("FACTS:")
        user_content_parts.append(fb.rstrip())
        user_content_parts.append("")

    if cb:
        user_content_parts.append("CONSTRAINTS:")
        user_content_parts.append(cb.rstrip())
        user_content_parts.append("")

    # QUESTION always present
    user_content_parts.append("QUESTION:")
    user_content_parts.append(effective_user_text if effective_user_text else "[no explicit question]")

    user_block = "\n".join(user_content_parts).strip()

    # 6) Build the final prompt (system prompt includes Layer 2 - enhanced reasoning instructions)
    prompt = f"{SERIOUS_SYSTEM_PROMPT}\n\n{user_block}\n\nANSWER:\n"

    # 7) Call Thinker model
    answer = call_model(
        role=thinker_role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 8) Enforce confidence line if missing
    # Default source heuristic: FACTS => Docs, else CONTEXT => Contextual, else Model.
    default_source = "Docs" if fb else ("Contextual" if context_block else "Model")
    
    # LAYER 3: If model response mentions "contradiction" or "impossible", force confidence to LOW
    answer_lower = (answer or "").lower()
    if "contradiction" in answer_lower or "impossible" in answer_lower or "mutually exclusive" in answer_lower:
        default_conf = "low"
    else:
        default_conf = "medium"
    
    answer = _ensure_confidence_line(
        answer,
        default_conf=default_conf,
        default_source=default_source,
    )

    return answer
