# pipelines.py
# version 1.0.0
"""
Specialized reasoning pipelines for llama-conductor.

Currently implements:
  - RAW mode: bypass Serious formatting prompt, keep harness constraints (CTC, KB grounding)
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional


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
    RAW mode: answer the user without the Serious formatting prompt.
    
    Still uses:
      - Vodka (inlet/outlet for CTC trimming + memory expansion)
      - KB grounding (FACTS block if attached)
      - Constraints (if provided)
    
    Does NOT use:
      - Serious system prompt (no confidence/source line appended)
      - Fun/FR transforms
      - Any styling
    
    Result is model output as-is, with minimal scaffolding.
    """

    # 1) Apply Vodka inlet for CTC trimming (same as Serious)
    body = {"messages": history}
    try:
        body = vodka.inlet(body)
        body = vodka.outlet(body)
        messages = body.get("messages", history) or history
    except Exception:
        messages = history

    # 2) Extract effective user text (possibly rewritten by Vodka if ?? was used)
    effective_user_text = _extract_last_user_message(messages, user_text)

    # 3) Build a minimal prompt (no Serious system prompt)
    fb = (facts_block or "").strip()
    cb = (constraints_block or "").strip()

    prompt_parts: List[str] = []

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

    # 4) Call model directly (minimal framing)
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
#   - run_coder() — code generation with constraints
#   - run_research() — deep retrieval + synthesis
#   - run_creative() — creative writing with worldbuilding
