from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


async def handle_vision_ocr_selector(
    *,
    selector: str,
    raw_messages: List[dict],
    run_sync: Callable[..., Any],
    call_model_messages: Callable[..., str],
) -> Optional[str]:
    """Handle explicit vision/ocr selector path."""
    if selector not in ("vision", "ocr"):
        return None
    return await run_sync(
        call_model_messages,
        role="vision",
        messages=raw_messages,
        max_tokens=700,
        temperature=0.2,
        top_p=0.9,
    )


def resolve_fun_mode(
    *,
    selector: str,
    state: Any,
    history_text_only: List[dict],
    has_mentats_in_recent_history: Callable[..., bool],
) -> Tuple[str, Optional[str]]:
    """Resolve effective fun mode and optional blocking message.

    Precedence order:
      1. Explicit selector (##fun) → fun
      2. fun_rewrite_sticky → fun_rewrite
      3. fun_sticky → fun
      4. serious_sticky → serious
      5. raw_sticky → raw
      6. default_mode fallthrough (empty = serious implicit)
    """
    if selector == "fun":
        state.fun_sticky = False
        state.fun_rewrite_sticky = False
        fun_mode = "fun"
    else:
        fun_mode = ""

    if not fun_mode and (state.fun_rewrite_sticky or state.fun_sticky):
        if has_mentats_in_recent_history(history_text_only, last_n=5):
            state.fun_sticky = False
            state.fun_rewrite_sticky = False
            return "", "[router] Fun/FR auto-disabled: Mentats output in recent history. Start new topic or re-enable with >>f"

    if not fun_mode and state.fun_rewrite_sticky:
        fun_mode = "fun_rewrite"
    elif not fun_mode and state.fun_sticky:
        fun_mode = "fun"

    # serious_sticky overrides default_mode (same pattern as fun_sticky).
    if not fun_mode and getattr(state, "serious_sticky", False):
        fun_mode = "serious"

    # raw_sticky overrides default_mode (existing behaviour, kept for back-compat).
    if not fun_mode and getattr(state, "raw_sticky", False):
        fun_mode = "raw"

    # default_mode fallthrough: when no sticky is set, use configured default.
    # Empty string = serious (implicit legacy default).
    if not fun_mode:
        dm = str(getattr(state, "default_mode", "") or "").strip().lower()
        if dm in ("fun", "fun_rewrite", "raw", "serious"):
            fun_mode = dm

    return fun_mode, None

