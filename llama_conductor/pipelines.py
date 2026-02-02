# pipelines.py
# version 1.0.2
"""
Specialized reasoning pipelines for llama-conductor.

CHANGES IN v1.0.2:
- RAW mode now has minimal system prompt to prevent:
  * Annoying preambles ("Raw mode active—no stiff formatting. I'm listening.")
  * Mid-sentence cutoffs causing loops when user says "go on"
  * Mode announcements and meta-commentary
- System prompt keeps RAW conversational but gives structure to prevent failure modes
- Increased recommended CTC limits for RAW (see vodka_filter.py settings below)

CHANGES IN v1.0.1:
- RAW mode now includes CONTEXT block (conversation history) so queries like "what have we discussed?"
  work correctly instead of hallucinating. Model now has access to prior turns for disambiguation.

Currently implements:
  - RAW mode: conversational with minimal structure, keeps harness constraints (CTC, KB grounding, CONTEXT)
  - Serious mode: structured output with confidence lines (see serious.py)
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional, Tuple


# ============================================================================
# RAW Mode System Prompt (NEW in v1.0.2)
# ============================================================================

RAW_SYSTEM_PROMPT = """Conversational mode.

Core rules:
- Answer naturally and directly (no preambles, no mode announcements)
- Complete your thoughts (don't cut off mid-sentence)
- If user says "continue" or "go on", pick up EXACTLY where you left off
- Use the CONTEXT section to understand prior conversation turns
- Use the FACTS section when present (treat as ground truth)
- Use the CONSTRAINTS section when present (must obey)

DO NOT:
- Announce "Raw mode active" or similar
- Use phrases like "I'm listening" or "no stiff formatting"
- Repeat previous message fragments when continuing
- Cut off mid-sentence (complete your current thought)

Just talk naturally like a human would."""


# ============================================================================
# Context Building (shared utility from serious.py)
# ============================================================================


def _extract_chat_summary(messages: List[Dict[str, Any]]) -> str:
    """Grab the most recent Vodka summary message if present."""
    for m in reversed(messages or []):
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip().startswith("[CHAT_SUMMARY]"):
                return c.strip()
    return ""


def _compact_turn_text(s: str, max_len: int) -> str:
    """Compact text to max_len with ellipsis."""
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
    - Includes latest summary (if present) and last N user/assistant turns (excluding final user turn).
    - Keeps output tightly bounded so raw/serious stay "low-context".
    """
    if not messages:
        return ""

    # Exclude current user message from context turns
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


# ============================================================================
# RAW Mode Pipeline
# ============================================================================


def run_raw(
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
) -> str:
    """
    RAW mode: conversational answer with minimal system prompt for structure.
    
    v1.0.2 changes:
      - Added RAW_SYSTEM_PROMPT to prevent preamble spam and continuation loops
      - Model now knows to complete thoughts and handle "go on" correctly
    
    Still uses:
      - Vodka (inlet/outlet for CTC trimming + memory expansion)
      - CONTEXT (conversation history for questions requiring prior context)
      - KB grounding (FACTS block if attached)
      - Constraints (if provided)
    
    Does NOT use:
      - Serious system prompt (no confidence/source line appended)
      - Fun/FR transforms
      - Heavy formatting
    
    Result is conversational model output with just enough structure to prevent failure modes.
    """

    # 1) Apply Vodka inlet for CTC trimming
    body = {"messages": history}
    try:
        body = vodka.inlet(body)
        body = vodka.outlet(body)
        messages = body.get("messages", history) or history
    except Exception:
        messages = history

    # 2) Extract effective user text (possibly rewritten by Vodka if ?? was used)
    effective_user_text = _extract_last_user_message(messages, user_text)

    # 3) Build CONTEXT block (needed for summary queries and conversation continuity)
    context_block = _build_context_block(
        messages,
        include_summary=True,
        max_turn_pairs=4,
        max_chars=900,
        per_turn_max_chars=240,
    )

    # 4) Build prompt with RAW system prompt + sections
    fb = (facts_block or "").strip()
    cb = (constraints_block or "").strip()

    prompt_parts: List[str] = []
    
    # Add system prompt (NEW in v1.0.2)
    prompt_parts.append(RAW_SYSTEM_PROMPT)
    prompt_parts.append("")

    # Add CONTEXT if present
    if context_block:
        prompt_parts.append("CONTEXT:")
        prompt_parts.append(context_block)
        prompt_parts.append("")

    # Add FACTS block if present
    if fb:
        prompt_parts.append("FACTS:")
        prompt_parts.append(fb)
        prompt_parts.append("")

    # Add CONSTRAINTS block if present
    if cb:
        prompt_parts.append("CONSTRAINTS:")
        prompt_parts.append(cb)
        prompt_parts.append("")

    # Add the question
    prompt_parts.append(effective_user_text if effective_user_text else "[no question provided]")

    prompt = "\n".join(prompt_parts).strip()

    # 5) Call model with structured prompt
    answer = call_model(
        role=thinker_role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return (answer or "").strip()


def _extract_last_user_message(messages: List[Dict[str, Any]], fallback: str) -> str:
    """Extract the most recent user message from history."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                return c.strip()
    return (fallback or "").strip()


# ============================================================================
# Future Pipelines (Placeholders)
# ============================================================================

# Additional specialized pipelines can be added here in the future.
# Examples:
#   - run_coder() – code generation with constraints
#   - run_research() – deep retrieval + synthesis
#   - run_creative() – creative writing with worldbuilding
