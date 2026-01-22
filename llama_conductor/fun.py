```python
# fun.py
# version 1.0.7
#
# SAFETY / COMPATIBILITY GOALS (MAX PRECAUTIONS):
# - Do NOT change any public function signatures.
# - Do NOT change how external callers invoke this module.
# - Do NOT modify upstream/downstream modules (router, mentats, vodka, image routing).
# - Fix the duplicate FUN quote issue (>>f) by ensuring run_fun() returns a seed-quote-only first line.
# - Prevent router/UI meta messages (e.g., "[router] ...", "[status] ...", "[peek] ...", "Confidence: ...")
#   from contaminating the model-facing transcript used inside Fun mode, WITHOUT removing them from
#   the actual chat history (no in-place mutation of messages).
#
# NOTE:
# - This file does NOT implement routing or command handling; it only formats prompts and post-processes outputs.
# - Any changes here are strictly inside the "Fun" generation path, and are designed to be fail-open
#   (return a sane answer rather than raise).
#
# Behavior summary:
# - [FUN] mode (run_fun): produce a correct base answer, then rewrite in a seed quote voice.
#   Returns:
#       <seed quote text ONLY on first line>
#       <blank line>
#       <rewrite body>
#   The router wraps the first line into: [FUN] "<quote>"
#
# - [FUN REWRITE] helper (run_fun_rewrite): unchanged format; included for compatibility.

from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional, Tuple
import random
import os

# -----------------------------
# THINKER (facts + continuity)
# -----------------------------

FUN_THINKER_PROMPT = """You are answering as part of an ongoing conversation.

Rules:
- Stay on the SAME topic unless the user clearly changes it.
- Use FACTS_BLOCK when present. If FACTS_BLOCK is empty, do NOT invent facts.
- If the question is ambiguous, resolve it from context.
- If still unclear, ask ONE short clarification question.
- Be concise and technically accurate.
"""

# -----------------------------
# ACTOR (style rewrite)
# -----------------------------

ACTOR_PROMPT = """You are rewriting an answer using a seed quote.

You are given:
- QUOTE_KICKER (must be first line exactly as provided)
- SEED_QUOTE (the quote without the [FUN] marker)
- ORIGINAL_ANSWER (already correct)

Rules:
- Output QUOTE_KICKER as the FIRST line exactly as provided, then a blank line, then your rewrite.
- Rewrite ORIGINAL_ANSWER in a voice that fits SEED_QUOTE.
- Style may bend grammar, tone, and voice, but never semantics.
- Attitudinal worldview may be emulated, but epistemic claims may not be altered.
- Do NOT add new facts.
- Do NOT change uncertainty or confidence.
- Do NOT change topic.
- Keep the rewrite concise and natural.
"""

FUN_REWRITE_PICK_PROMPT = """You are selecting a tone/character tag for a style rewrite.

You will be given:
- USER_TEXT
- ORIGINAL_ANSWER
- TAGS (a list of headings from quotes.md)

Task:
- Pick exactly ONE tag line from TAGS that best matches the mood/tone for a playful rewrite.
- Output EXACTLY the chosen tag line (verbatim), and nothing else.
"""

FUN_REWRITE_ACTOR_PROMPT = """You are rewriting an answer using a seed quote.

You are given:
- QUOTE_KICKER (must be first line exactly as provided)
- SEED_QUOTE (the same quote, without the mode marker)
- ORIGINAL_ANSWER (already correct)

Rules:
- Output QUOTE_KICKER as the FIRST line.
- Rewrite ORIGINAL_ANSWER in a voice that fits SEED_QUOTE.
- Style may bend grammar, tone, and voice, but never semantics.
- Attitudinal worldview may be emulated, but epistemic claims may not be altered.
- Do NOT add new facts.
- Do NOT change uncertainty or confidence.
- Do NOT change topic.
"""

# -----------------------------
# Helpers
# -----------------------------

_RECENT_KICKERS: Dict[str, List[str]] = {}


def _is_mentats_output(text: str) -> bool:
    """Heuristic detector; kept for compatibility. Not used for routing here."""
    t = (text or "")
    return "[ZARDOZ HATH SPOKEN]" in t or "Sources: Vault" in t


def _pick_kicker(quotes: List[str], session_id: str, cooldown: int = 6) -> str:
    if not quotes:
        return "Good news, everyone!"

    recent = set(q.lower() for q in _RECENT_KICKERS.get(session_id, [])[-cooldown:])
    pool = [q for q in quotes if q.lower() not in recent] or quotes
    kicker = random.choice(pool)

    _RECENT_KICKERS.setdefault(session_id, []).append(kicker)
    return kicker


# ---- Meta filtering (MAX precautions) ----

# We only filter these from the *model-facing transcript string* for Fun mode.
# We do NOT mutate history messages, and we do NOT filter user text.
_META_PREFIXES = (
    "[router]",
    "[status]",
    "[peek]",
    "[vodka]",
    "[debug]",
    "[error]",
    "confidence:",
    "source:",
)

def _looks_like_meta_line(line: str) -> bool:
    """Conservative: only treat a line as meta if it starts with known prefixes."""
    s = (line or "").strip()
    if not s:
        return False
    low = s.lower()
    return any(low.startswith(p) for p in _META_PREFIXES)


def _sanitize_message_for_transcript(role: str, content: str) -> str:
    """
    Return a version of content safe for inclusion in the model-facing transcript string.
    Max precautions:
    - Do NOT remove user content.
    - For assistant content, drop only known meta lines (router/status/peek/confidence/source etc).
    - Keep everything else intact.
    """
    txt = (content or "").strip()
    if not txt:
        return ""

    # Never strip user text (to avoid removing legitimate bracketed constraints)
    if role == "user":
        return txt

    # For assistant text, remove meta-only lines but keep substantive content.
    # This prevents "[router] fun mode ON (sticky)" from becoming conversational material.
    lines = txt.splitlines()
    kept: List[str] = []
    for ln in lines:
        if _looks_like_meta_line(ln):
            continue
        kept.append(ln)
    cleaned = "\n".join(kept).strip()
    return cleaned


def _pack_history(history: List[Dict[str, Any]], turns: int = 10) -> str:
    """
    Build a lightweight transcript for continuity.
    MAX precautions:
    - No in-place mutations.
    - Conservative meta filtering on assistant messages only.
    """
    tail = history[-turns:]
    out: List[str] = []
    for m in tail:
        role = m.get("role")
        raw = m.get("content")
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)

        cleaned = _sanitize_message_for_transcript(str(role or ""), raw)
        if not cleaned:
            continue

        if role == "user":
            out.append(f"USER: {cleaned}")
        elif role == "assistant":
            out.append(f"ASSISTANT: {cleaned}")
        # ignore other roles for transcript to keep it simple/deterministic
    return "\n".join(out)


def _format_mode_kicker(mode: str, quote: str) -> str:
    q = (quote or "").strip() or "Good news, everyone!"
    return f'[{mode}] "{q}"'


def _load_quotes_by_tag(quotes_path: str) -> List[Tuple[str, List[str]]]:
    """Parse quotes.md-style files.

    Format:
      ## tag words...
      quote line
      quote line
      ## next tags...
      ...

    Returns list of (tag_line, quotes_under_tag).
    """
    if not quotes_path:
        return []
    if not os.path.exists(quotes_path):
        return []

    tag_blocks: List[Tuple[str, List[str]]] = []
    current_tag: Optional[str] = None
    current_quotes: List[str] = []

    with open(quotes_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Check for tag headers BEFORE skipping comments
            if line.startswith("## "):
                if current_tag is not None:
                    tag_blocks.append((current_tag, current_quotes))
                current_tag = line
                current_quotes = []
                continue
            # Skip comment lines (but not tag headers)
            if line.startswith("#"):
                continue
            # quote line
            if current_tag is not None:
                current_quotes.append(line)

    if current_tag is not None:
        tag_blocks.append((current_tag, current_quotes))

    # DEBUG: Show what was loaded
    print(f"[DEBUG FUN] Loaded {len(tag_blocks)} tag blocks from quotes.md", flush=True)
    for tag, quotes in tag_blocks[:3]:  # Show first 3
        print(
            f"[DEBUG FUN]   Tag: {tag!r}, Quotes: {len(quotes)} (first: {quotes[0] if quotes else 'NONE'})",
            flush=True,
        )

    return tag_blocks


def _choose_tag_for_rewrite(
    *,
    session_id: str,
    user_text: str,
    original_answer: str,
    tag_lines: List[str],
    call_model: Callable[..., str],
) -> str:
    if not tag_lines:
        return "## meta"

    prompt = (
        f"{FUN_REWRITE_PICK_PROMPT}\n\n"
        f"USER_TEXT:\n{user_text.strip()}\n\n"
        f"ORIGINAL_ANSWER:\n{original_answer.strip()}\n\n"
        f"TAGS:\n" + "\n".join(tag_lines) + "\n"
    )

    picked = call_model(
        role="thinker",
        prompt=prompt,
        max_tokens=64,
        temperature=0.2,
        top_p=0.9,
    )

    picked_line = (picked or "").strip().splitlines()[0].strip()
    # Must be one of the provided tags
    for t in tag_lines:
        if picked_line == t:
            return t

    # fallback: pick a deterministic-ish tag (avoid repeating)
    return random.choice(tag_lines)


def _strip_leading_kicker_lines(text: str, *, mode_kicker: str, seed: str) -> str:
    """
    Remove any leading kicker/header lines that the actor model may echo.

    Used by run_fun() to prevent a second displayed quote line.
    Conservative: only strips at the very top.
    """
    if not text:
        return ""

    lines = text.splitlines()

    mk = (mode_kicker or "").strip()
    sd = (seed or "").strip()

    def is_kicker_or_blank(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return True  # treat leading blanks as removable
        if mk and s == mk:
            return True
        if sd and (s == sd or s == f'"{sd}"' or s.strip('"').strip() == sd):
            return True
        return False

    i = 0
    while i < len(lines) and is_kicker_or_blank(lines[i]):
        i += 1

    # remove one extra blank block after the kicker
    while i < len(lines) and not (lines[i] or "").strip():
        i += 1

    return "\n".join(lines[i:]).strip()


def _strip_leading_meta_lines(text: str) -> str:
    """
    Extra-hardening: if the model still emits meta lines at the start of the body,
    strip ONLY those leading meta lines. This avoids removing legitimate user content
    elsewhere in the answer.
    """
    if not text:
        return ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = (lines[i] or "").strip()
        if not ln:
            i += 1
            continue
        if _looks_like_meta_line(ln):
            i += 1
            continue
        break
    # also drop a single blank run after stripped meta
    while i < len(lines) and not (lines[i] or "").strip():
        i += 1
    return "\n".join(lines[i:]).strip()


# -----------------------------
# MAIN ENTRY: FUN (two-pass)
# -----------------------------

def run_fun(
    *,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    facts_block: str,
    quote_pool: List[str],
    vodka,
    call_model: Callable[..., str],
    thinker_role: str = "thinker",
) -> str:
    """
    Fun mode: correct answer first, then rewrite in a character voice.

    OUTPUT CONTRACT (v1.0.6+):
    - First line is the SEED QUOTE TEXT ONLY (no [FUN] prefix).
    - Router is responsible for wrapping it into [FUN] "<quote>".
    - Body contains only the styled answer (no repeated kicker line).
    """

    # --- Continuity (do not mutate history) ---
    body = {"messages": history}
    # Keep existing behavior: this module still calls vodka inlet/outlet as in v1.0.5.
    # (We do not change upstream contracts; callers may rely on this trimming.)
    body = vodka.inlet(body)
    body = vodka.outlet(body)
    trimmed = body.get("messages", history)

    # IMPORTANT: model-facing transcript is sanitized to remove router/UI meta assistant lines
    transcript = _pack_history(trimmed)

    # --- THINKER (correct content) ---
    thinker_prompt = (
        f"{FUN_THINKER_PROMPT}\n\n"
        f"CHAT HISTORY:\n{transcript or '[none]'}\n\n"
        f"FACTS_BLOCK:\n{facts_block or 'NONE'}\n\n"
        f"QUESTION:\n{user_text}\n\n"
        "ANSWER:\n"
    )

    base_answer = call_model(
        role=thinker_role,
        prompt=thinker_prompt,
        max_tokens=280,
        temperature=0.3,
        top_p=0.9,
    )
    base_answer = (base_answer or "").strip()
    if not base_answer:
        return ""

    # --- STYLE ---
    seed = _pick_kicker(quote_pool, session_id)
    # Keep mode_kicker for the actor prompt (behavior preserved); DO NOT return it directly.
    mode_kicker = _format_mode_kicker("FUN", seed)

    actor_prompt = (
        f"{ACTOR_PROMPT}\n\n"
        f"QUOTE_KICKER:\n{mode_kicker}\n\n"
        f"SEED_QUOTE:\n{seed}\n\n"
        f"ORIGINAL_ANSWER:\n{base_answer}\n\n"
        "REWRITE:\n"
    )

    styled = call_model(
        role=thinker_role,
        prompt=actor_prompt,
        max_tokens=360,
        temperature=0.85,
        top_p=0.95,
    )
    styled = (styled or "").strip()

    # If style failed, degrade to base (still honor the output contract).
    if not styled:
        return f"{seed}\n\n{base_answer}"

    # Extract rewrite body robustly
    if "REWRITE:" in styled:
        body_text = styled.split("REWRITE:", 1)[1].strip() or base_answer
    else:
        body_text = styled

    # Critical: remove echoed kicker/header lines so the quote isn't displayed twice.
    body_text = _strip_leading_kicker_lines(body_text, mode_kicker=mode_kicker, seed=seed)
    # Extra: strip any leading router/status/peek/confidence meta lines if they slip in.
    body_text = _strip_leading_meta_lines(body_text)

    if not body_text:
        body_text = base_answer

    # Return with seed-only first line (router will wrap with [FUN] "...").
    return f"{seed}\n\n{body_text}"


# -----------------------------
# FUN REWRITE (FR): rewrite an existing correct answer
# -----------------------------

def run_fun_rewrite(
    *,
    session_id: str,
    user_text: str,
    original_answer: str,
    quotes_path: str,
    call_model: Callable[..., str],
    debug: bool = False,
) -> str:
    """Rewrite an existing (already correct) answer in a tone-matched character voice.

    Intended to be used post-reasoning by the router.
    """

    original_answer = (original_answer or "").strip()
    if not original_answer:
        return ""

    blocks = _load_quotes_by_tag(quotes_path)
    tag_lines = [t for (t, qs) in blocks if qs]

    if not blocks or not tag_lines:
        # No quotes available; degrade gracefully
        mode_kicker = _format_mode_kicker("FUN REWRITE", "Good news, everyone!")
        return f"{mode_kicker}\n\n{original_answer}"

    chosen_tag = _choose_tag_for_rewrite(
        session_id=session_id,
        user_text=user_text,
        original_answer=original_answer,
        tag_lines=tag_lines,
        call_model=call_model,
    )

    quotes_for_tag: List[str] = []
    for t, qs in blocks:
        if t == chosen_tag:
            quotes_for_tag = qs
            break

    seed = _pick_kicker(quotes_for_tag, session_id)
    mode_kicker = _format_mode_kicker("FUN REWRITE", seed)

    actor_prompt = (
        f"{FUN_REWRITE_ACTOR_PROMPT}\n\n"
        f"QUOTE_KICKER:\n{mode_kicker}\n\n"
        f"SEED_QUOTE:\n{seed}\n\n"
        f"ORIGINAL_ANSWER:\n{original_answer}\n\n"
        "REWRITE:\n"
    )

    styled = call_model(
        role="thinker",
        prompt=actor_prompt,
        max_tokens=360,
        temperature=0.95,
        top_p=0.95,
    )

    styled = (styled or "").strip()
    if not styled:
        return f"{mode_kicker}\n\n{original_answer}"

    if "REWRITE:" not in styled:
        rewritten = styled
    else:
        rewritten = styled.split("REWRITE:", 1)[1].strip() or original_answer

    if debug:
        return (
            f"{mode_kicker}\n\n"
            "[DEBUG ORIGINAL]\n" + original_answer + "\n\n"
            "[REWRITE]\n" + rewritten
        )

    return f"{mode_kicker}\n\n{rewritten}"
```
