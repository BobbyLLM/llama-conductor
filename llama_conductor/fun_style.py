from __future__ import annotations

import random
from typing import Any, Dict, List

from .config import cfg_get
from .interaction_profile import effective_profile
from .model_calls import call_model_prompt
from .quotes import infer_tone as _infer_tone, pick_quote_for_tone as _pick_quote_for_tone, quotes_by_tag as _quotes_by_tag
from .serious import run_serious

_SNARK_TAGS = {"snark", "sarcastic", "banter", "quips", "one-liners", "deadpan", "dry", "threat", "warning"}
_WARM_TAGS = {"warm", "supportive", "compassionate", "hopeful", "resilient"}
_DIRECT_TAGS = {"resilient", "dry", "deadpan"}


def _unique_quotes_for_tags(qb: Dict[str, List[str]], tags: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        for q in qb.get((tag or "").lower(), []) or []:
            k = (q or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(q)
    return out


def _build_fun_quote_pool(*, state: Any, user_text: str, tone: str) -> List[str]:
    """
    Minimal deterministic quote prefilter for Fun mode.
    Combines inferred tone and profile traits, then falls back safely.
    """
    qb = _quotes_by_tag()
    if not qb:
        return []

    include_tags: List[str] = ["default"]
    t = (tone or "").strip().lower()
    if t:
        include_tags.append(t)

    exclude_snark = False
    if bool(getattr(state, "profile_enabled", False)):
        try:
            prof = effective_profile(state.interaction_profile, user_text)
            if prof.correction_style == "softened":
                include_tags.extend(sorted(_WARM_TAGS))
            elif prof.correction_style == "direct":
                include_tags.extend(sorted(_DIRECT_TAGS))

            if prof.sarcasm_level in ("medium", "high") or prof.snark_tolerance in ("medium", "high"):
                include_tags.extend(sorted(_SNARK_TAGS))
            else:
                exclude_snark = True
        except Exception:
            pass

    pool = _unique_quotes_for_tags(qb, include_tags)
    if not pool:
        pool = _unique_quotes_for_tags(qb, list(qb.keys()))

    if exclude_snark:
        snark_set = {q.strip().lower() for q in _unique_quotes_for_tags(qb, sorted(_SNARK_TAGS))}
        filtered = [q for q in pool if q.strip().lower() not in snark_set]
        if filtered:
            pool = filtered

    return pool


def select_fun_style_seed(*, state: Any, user_text: str, base_text: str) -> Dict[str, Any]:
    """Shared deterministic selector core for both FUN and FR renderers."""
    tone = _infer_tone(user_text, base_text)
    pool = _build_fun_quote_pool(state=state, user_text=user_text, tone=tone)
    seed = random.choice(pool) if pool else (_pick_quote_for_tone(tone) or "")
    return {"tone": tone, "pool": pool, "seed": seed}


def run_fun_rewrite_fallback(
    *,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    vodka: Any,
    facts_block: str,
    state: Any,
    base_override: str = "",
) -> str:
    """Two-pass Fun Rewrite implemented in-router fallback path."""

    base = (base_override or "").strip()
    if not base:
        base = run_serious(
            session_id=session_id,
            user_text=user_text,
            history=history,
            vodka=vodka,
            call_model=call_model_prompt,
            facts_block=facts_block,
            constraints_block="",
            thinker_role="thinker",
            max_tokens=int(cfg_get("serious.max_tokens", 384)),
        ).strip()

    sel = select_fun_style_seed(state=state, user_text=user_text, base_text=base)
    quote = str(sel.get("seed") or "")

    rewrite_prompt = (
        "You are rewriting an answer in a pop-culture character voice.\n"
        "You are given a SEED_QUOTE which anchors tone/voice.\n\n"
        "Rules:\n"
        "- Style may bend grammar, tone, and voice, but never semantics.\n"
        "- Attitudinal worldview may be emulated, but epistemic claims may not be altered.\n"
        "- Do NOT add new facts. Do NOT remove key facts.\n"
        "- Output ONLY the rewritten answer (no preamble, no analysis).\n\n"
        f"SEED_QUOTE: {quote}\n\n"
        f"ORIGINAL_ANSWER:\n{base}\n\n"
        "REWRITE:"
    )

    rewritten = call_model_prompt(
        role="thinker",
        prompt=rewrite_prompt,
        max_tokens=420,
        temperature=0.85,
        top_p=0.95,
    ).strip()

    if not rewritten:
        rewritten = base

    qline = f'"{quote}"' if quote else '""'
    return f"[FUN REWRITE] {qline}\n\n{rewritten}"
