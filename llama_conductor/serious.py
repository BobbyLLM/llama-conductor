# serious.py
# version 1.0.0
"""
Serious mode pipeline (/serious default posture).

This module:
- Calls Vodka inlet() for CTC/FR and control commands (including ??).
- Calls Vodka outlet() so "[ctx:...]" markers are expanded into verbatim stored text.
- Uses the *modified* last user message after Vodka (so ?? rewrites work).
- Adds a small CONTEXT section (summary + last few turns) so Serious can benefit from breadcrumbs.
- Enforces a confidence line if the model forgets to add one.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import re


SERIOUS_SYSTEM_PROMPT = """You are in /serious mode.

Tone:
- Neutral, precise, low-context.
- No filler, no small talk, no soft closure.

Output rules:
- Answer first. No preamble.
- ≤3 short paragraphs (you may add a short bullet list or code block if needed).
- Minimal emotion; no subjective experiences or fictional biography.
- End with a plain declarative sentence.

Input format:
The user message may contain optional sections:

CONTEXT:
- Excerpts from prior turns and/or rehydrated breadcrumbs from stored notes.
- Helpful for disambiguation, continuity, and "what we already established".
- Not authoritative ground truth unless also repeated in FACTS.

FACTS:
- Retrieved snippets or user-provided facts for this turn.

CONSTRAINTS:
- Hard rules to obey.

QUESTION:
- The actual request to answer.

Priority:
1) CONSTRAINTS (must obey)
2) FACTS (treat as primary ground truth when present)
3) CONTEXT (use for disambiguation, but do not treat as guaranteed truth)
4) QUESTION (answer it)

Uncertainty / self-check:
- If the question is complex, under-specified, or high-impact, briefly mention:
  - What information is missing or uncertain, and
  - The most likely way your answer could be wrong.
- Keep this to 1–2 short sentences at the end of the answer text (before the confidence line).

Confidence and source:
- At the very end of every answer, append a line:

Confidence: [low | medium | high | top] | Source: [Model | Docs | User | Contextual | Mixed]

Where:
- Confidence:
  - low   = weak support, guesswork, or major gaps.
  - medium= some support but important gaps.
  - high  = well-supported with minor uncertainty.
  - top   = directly backed by clear info (especially from FACTS), minimal doubt.
- Source:
  - Model     = mostly from pretrained knowledge.
  - Docs      = mostly from FACTS supplied in the message (e.g. RAG output).
  - User      = primarily restating/organizing user-supplied content.
  - Contextual= inferred from this conversation’s prior turns.
  - Mixed     = substantial mix of the above, none clearly dominant.

Always follow these rules.
"""

_CONF_LINE_RE = re.compile(
    r"Confidence:\s*(low|medium|med|high|top)\s*\|\s*Source:\s*(Model|Docs|User|Contextual|Mixed)",
    re.IGNORECASE,
)


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
    """Grab the most recent Vodka summary message if present."""
    # Vodka uses SUMMARY_PREFIX = "[CHAT_SUMMARY] " (see vodka_filter.py),
    # but we don't import it here; we just detect by prefix.
    for m in reversed(messages or []):
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip().startswith("[CHAT_SUMMARY]"):
                return c.strip()
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
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    # New (safe defaults; router doesn't need to pass these)
    include_context: bool = True,
    context_max_turn_pairs: int = 4,
    context_max_chars: int = 900,
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
    """

    # 1) Run Vodka inlet for CTC/FR + control commands on history
    raw_control = _is_control_command(user_text)
    body = {"messages": history}
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

    fb = (facts_block or "").strip()
    cb = (constraints_block or "").strip()

    # 3) Build CONTEXT block (trimmed+expanded), if enabled
    context_block = ""
    if include_context:
        context_block = _build_context_block(
            messages,
            include_summary=True,
            max_turn_pairs=context_max_turn_pairs,
            max_chars=context_max_chars,
            per_turn_max_chars=240,
        )

    # 4) Build the synthetic "user message" content as described in SERIOUS_SYSTEM_PROMPT
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

    # 5) Build the final prompt
    prompt = f"{SERIOUS_SYSTEM_PROMPT}\n\n{user_block}\n\nANSWER:\n"

    # 6) Call Thinker model
    answer = call_model(
        role=thinker_role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # 7) Enforce confidence line if missing
    # Default source heuristic: FACTS => Docs, else CONTEXT => Contextual, else Model.
    default_source = "Docs" if fb else ("Contextual" if context_block else "Model")
    answer = _ensure_confidence_line(
        answer,
        default_conf="medium",
        default_source=default_source,
    )

    return answer
