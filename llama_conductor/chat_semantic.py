from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from .config import cfg_get
from .consistency_verifier import should_verify_response, verify_response_consistency


def maybe_apply_consistency_verifier(
    *,
    user_text: str,
    draft_text: str,
    query_family: str,
    lock_active: bool,
    state_solver_used: bool,
    prior_user_text: str,
    call_model_messages_fn: Callable[..., str],
) -> str:
    draft = str(draft_text or "").strip()
    if not draft:
        return draft
    if lock_active or state_solver_used:
        return draft
    if not bool(cfg_get("state_solver.consistency_verifier.enabled", True)):
        return draft
    if not should_verify_response(user_text=user_text, query_family=query_family):
        return draft
    role = str(cfg_get("state_solver.consistency_verifier.role", "thinker") or "thinker").strip() or "thinker"
    max_tokens = int(cfg_get("state_solver.consistency_verifier.max_tokens", 220))
    temperature = float(cfg_get("state_solver.consistency_verifier.temperature", 0.0))
    top_p = float(cfg_get("state_solver.consistency_verifier.top_p", 0.1))
    try:
        verified = verify_response_consistency(
            user_text=user_text,
            draft_text=draft,
            role=role,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            call_model_messages=call_model_messages_fn,
            prior_user_text=prior_user_text,
        )
        return str(verified or draft).strip()
    except Exception:
        return draft


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def semantic_pick_clarifier_option(
    *,
    user_text: str,
    options: List[str],
    call_model_prompt_fn: Callable[..., str],
) -> str:
    """Map ambiguous user wording to one of the existing options (or empty if unclear)."""
    opts = [str(x).strip() for x in (options or []) if str(x).strip()]
    if not opts:
        return ""
    low_user = str(user_text or "").strip().lower()
    if re.search(r"\b(difference|same|equivalent|what difference|aren't|arent)\b", low_user):
        return ""
    role = str(cfg_get("state_solver.semantic_choice.role", "thinker") or "thinker").strip() or "thinker"
    max_tokens = int(cfg_get("state_solver.semantic_choice.max_tokens", 120))
    temperature = float(cfg_get("state_solver.semantic_choice.temperature", 0.0))
    top_p = float(cfg_get("state_solver.semantic_choice.top_p", 0.1))

    opts_block = "\n".join(f"{i+1}) {o}" for i, o in enumerate(opts))
    prompt = (
        "You map a user clarification reply to one option.\n"
        "Return JSON only: {\"pick\":\"<exact option or empty>\",\"confidence\":\"high|medium|low\"}\n"
        "Rules:\n"
        "- pick must be EXACTLY one option string from OPTIONS, or empty string if unclear.\n"
        "- do not invent options.\n"
        "- if user asks for explanation/difference rather than choosing, return empty pick.\n\n"
        f"OPTIONS:\n{opts_block}\n\n"
        f"USER_REPLY:\n{str(user_text or '').strip()}\n"
    )

    raw = call_model_prompt_fn(
        role=role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    obj = _parse_json_object(raw)
    if not obj:
        return ""
    pick = str(obj.get("pick", "") or "").strip()
    if not pick:
        return ""
    if pick in opts:
        return pick

    low = pick.lower()
    if low in {"1", "option 1", "first", "the first one", "first one"} and len(opts) >= 1:
        return opts[0]
    if low in {"2", "option 2", "second", "the second one", "second one"} and len(opts) >= 2:
        return opts[1]
    if low in {"3", "option 3", "third", "the third one", "third one"} and len(opts) >= 3:
        return opts[2]

    best, best_score = "", 0.0
    for o in opts:
        s = SequenceMatcher(None, low, o.lower()).ratio()
        if s > best_score:
            best, best_score = o, s
    return best if best_score >= 0.88 else ""


def semantic_refine_constraint_choice(
    *,
    user_text: str,
    base_answer: str,
    frame: Dict[str, Any],
    reason: str,
    call_model_prompt_fn: Callable[..., str],
) -> str:
    """Optional narrow LLM rewrite for ambiguous non-vehicle assets."""
    if not bool(cfg_get("state_solver.semantic_refine.enabled", True)):
        return ""
    asset = str((frame or {}).get("asset") or "").strip().lower()
    if not asset:
        return ""
    animal_tokens = {
        "dog", "dogs", "cat", "cats", "horse", "horses", "cow", "cows",
        "sheep", "goat", "goats", "pig", "pigs", "puppy", "puppies",
        "kitten", "kittens", "pet", "pets",
    }
    if any(tok in asset.split() for tok in animal_tokens):
        return ""
    if asset in {"car", "truck", "van", "bus", "train", "tram", "boat", "ship", "motorcycle", "bike", "scooter"}:
        return ""
    if reason not in {
        "clarification_choice_operate",
        "clarification_choice_tow_transport",
        "clarification_choice_yourself_first",
        "clarification_choice_yourself_drive",
        "clarification_choice_walk",
    }:
        return ""

    role = str(cfg_get("state_solver.semantic_refine.role", "thinker") or "thinker").strip() or "thinker"
    max_tokens = int(cfg_get("state_solver.semantic_refine.max_tokens", 140))
    temperature = float(cfg_get("state_solver.semantic_refine.temperature", 0.0))
    top_p = float(cfg_get("state_solver.semantic_refine.top_p", 0.1))
    prompt = (
        "Rewrite the deterministic answer for clarity only.\n"
        "Do not change the selected action or decision.\n"
        "Keep to 1-2 short sentences.\n"
        "No new options, no extra facts, no policy words.\n"
        "If helpful, explain 'move under own power' in plain language.\n\n"
        f"asset: {asset}\n"
        f"user_reply: {str(user_text or '').strip()}\n"
        f"deterministic_answer: {str(base_answer or '').strip()}\n"
    )
    out = call_model_prompt_fn(
        role=role,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ).strip()
    if not out or out.startswith("[model "):
        return ""
    return out
