from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import hashlib
import json
import re
from difflib import SequenceMatcher

from .config import cfg_get


def _maybe_append_fun_train_kicker(
    *,
    text: str,
    user_text: str,
    session_id: str,
    seed_quote: str,
) -> str:
    """Add a tiny train-themed kicker for deterministic FUN transport answers.

    Design constraints:
    - no extra model call
    - scoped narrowly (train prompts only)
    - never mutate clarifier prompts
    """
    t = (text or "").strip()
    q = (user_text or "").lower()
    if not t:
        return t
    if "quick check: did you mean" in t.lower():
        return t
    if "train" not in q and "train" not in t.lower():
        return t
    if "destination" not in t.lower():
        return t
    # Avoid double-appending.
    if t.rstrip().endswith(("Choo!", "All aboard!", "Full steam!")):
        return t

    kickers = ("Choo!", "All aboard!", "Full steam!")
    key = f"{session_id}|{q}|{seed_quote}|{t}"
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(kickers)
    return t.rstrip() + " " + kickers[idx]


_INLINE_FOOTER_RE = re.compile(r"^\s*(confidence:|source:|sources:|profile:)\s*", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_FR_DISALLOWED_META_RE = re.compile(
    r"\b("
    r"i can't see|i cannot see|can't access|do not have access|"
    r"as an ai|source:\s*me|confidence:\s*high\.\s*source:\s*me|"
    r"original prompt is unavailable|offline history"
    r")\b",
    re.IGNORECASE,
)
_EXPLAIN_LOGIC_RE = re.compile(
    r"\b(flesh out|implicit assumptions|underlying logic|why you said|explain (the )?logic|assumptions)\b",
    re.IGNORECASE,
)


def _strip_mode_and_footer_lines(text: str) -> str:
    out: List[str] = []
    for ln in str(text or "").splitlines():
        s = (ln or "").strip()
        if not s:
            out.append("")
            continue
        if s.startswith("[FUN]") or s.startswith("[FUN REWRITE]"):
            continue
        if _INLINE_FOOTER_RE.match(s):
            continue
        out.append(ln)
    # squeeze blanks
    collapsed: List[str] = []
    blank = 0
    for ln in out:
        if not (ln or "").strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        collapsed.append(ln)
    return "\n".join(collapsed).strip()


def _transport_action_signature(text: str) -> str:
    t = str(text or "").lower()
    if "under its own power" in t or "move it under its own power" in t or "operate the train" in t:
        return "operate"
    if any(k in t for k in ("tow", "haul", "transport it", "transport the")):
        return "tow_transport"
    if any(k in t for k in ("get yourself there first", "walking is practical", "walk there first")):
        return "yourself_first"
    if re.search(r"\bdrive\.", t):
        return "drive"
    if re.search(r"\bwalk\.", t):
        return "walk"
    return ""


def _verify_rewrite_semantics_llm(
    *,
    user_text: str,
    base: str,
    rewritten: str,
    call_model_prompt: Callable[..., str],
) -> bool:
    role = str(cfg_get("state_solver.fr_guard.role", "thinker") or "thinker").strip() or "thinker"
    max_tokens = int(cfg_get("state_solver.fr_guard.max_tokens", 96))
    temperature = float(cfg_get("state_solver.fr_guard.temperature", 0.0))
    top_p = float(cfg_get("state_solver.fr_guard.top_p", 0.1))
    prompt = (
        "You are a strict semantic consistency checker.\n"
        "Given BASE_ANSWER and REWRITE_ANSWER, determine if REWRITE preserves BASE semantics.\n"
        "Rules:\n"
        "- REWRITE may change tone/style only.\n"
        "- It must not change decisions, numbers, constraints, or factual claims.\n"
        "- Output JSON only: {\"preserved\": true|false}\n\n"
        f"USER_QUERY:\n{str(user_text or '').strip()}\n\n"
        f"BASE_ANSWER:\n{base}\n\n"
        f"REWRITE_ANSWER:\n{rewritten}\n"
    )
    raw = str(
        call_model_prompt(
            role=role,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        or ""
    ).strip()
    if not raw:
        return False
    try:
        obj = json.loads(raw)
        return bool(obj.get("preserved", False))
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return False
        try:
            obj = json.loads(m.group(0))
            return bool(obj.get("preserved", False))
        except Exception:
            return False


def _guard_fun_rewrite(
    *,
    user_text: str,
    base_text: str,
    rewritten_text: str,
    call_model_prompt: Callable[..., str],
) -> str:
    """Hybrid FR guard for deterministic/state-solver turns.

    Phase 1: deterministic invariants (fast, zero extra call).
    Phase 2: optional semantic verifier only if deterministic check is inconclusive.
    """
    base = _strip_mode_and_footer_lines(base_text)
    rew = _strip_mode_and_footer_lines(rewritten_text)
    if not base or not rew:
        return base_text
    if _FR_DISALLOWED_META_RE.search(rew):
        return base_text

    # Deterministic numeric invariant: rewrite cannot introduce new numeric claims.
    bnums = set(_NUM_RE.findall(base))
    rnums = set(_NUM_RE.findall(rew))
    qnums = set(_NUM_RE.findall(str(user_text or "")))
    if bnums and (rnums - bnums):
        return base_text
    if (not bnums) and (rnums - qnums):
        return base_text

    # Deterministic action invariant for transport/state-decision branch.
    bsig = _transport_action_signature(base)
    rsig = _transport_action_signature(rew)
    if bsig and rsig and (bsig != rsig):
        return base_text

    # If deterministic checks are conclusive and close enough, accept quickly.
    sim = SequenceMatcher(None, base.lower(), rew.lower()).ratio()
    deterministic_conclusive = bool(bnums or bsig)
    if deterministic_conclusive and sim >= 0.45:
        return rewritten_text

    # Hybrid phase: semantic verifier only for ambiguous cases.
    if not bool(cfg_get("state_solver.fr_guard.semantic_verify", True)):
        return rewritten_text if sim >= 0.45 else base_text
    ok = _verify_rewrite_semantics_llm(
        user_text=user_text,
        base=base,
        rewritten=rew,
        call_model_prompt=call_model_prompt,
    )
    return rewritten_text if ok else base_text


def _deterministic_constraint_explanation(*, state: Any, user_text: str) -> str:
    """Build a grounded explanation for follow-up 'why/assumptions' turns in constraint lane."""
    if not _EXPLAIN_LOGIC_RE.search(str(user_text or "")):
        return ""
    if str(getattr(state, "deterministic_last_family", "") or "") != "constraint_decision":
        # Fallback: infer prior constraint-decision context from previous turn snapshots.
        prev_user = str(getattr(state, "last_user_text", "") or "").lower()
        prev_asst = str(getattr(state, "last_assistant_text", "") or "").lower()
        looks_like_prior_constraint = (
            ("should i" in prev_user)
            and ("walk" in prev_user)
            and ("drive" in prev_user)
            and any(k in prev_user for k in ("wash", "service", "repair", "car", "train", "truck", "bike", "motorbike", "motorcycle"))
            and ("destination" in prev_asst or "hard precondition" in prev_asst)
        )
        if not looks_like_prior_constraint:
            return ""
        asset = "asset"
        for cand in ("car", "train", "truck", "van", "bus", "motorcycle", "motorbike", "bike", "scooter", "boat", "ship", "tram"):
            if cand in prev_user:
                asset = "motorcycle" if cand == "motorbike" else cand
                break
        noun = "the asset" if asset == "asset" else f"the {asset}"
        return (
            f"The key assumption is that the task requires {noun} to be physically at the destination. "
            f"Walking only moves you, not {noun}, so it does not satisfy the hard precondition.\n\n"
            "Underlying logic: choose the action that satisfies the goal state directly. "
            "If constraints change (for example fuel, access, or towing availability), the answer can change."
        )
    frame = dict(getattr(state, "deterministic_last_frame", {}) or {})
    if str(frame.get("kind") or "") != "option_feasibility":
        return ""
    selected = str(frame.get("selected_action") or "").strip().lower()
    asset = str(frame.get("asset") or "asset").strip().lower() or "asset"
    noun = "the asset" if asset == "asset" else f"the {asset}"
    if selected in {"operate", "drive"}:
        return (
            f"The key assumption is that the task requires {noun} to be physically at the destination. "
            f"Walking only moves you, not {noun}, so it does not satisfy the hard precondition.\n\n"
            "Underlying logic: choose the action that satisfies the goal state directly. "
            "If constraints change (for example fuel, access, or towing availability), the answer can change."
        )
    if selected == "tow_transport":
        return (
            f"The key assumption is still goal-state based: {noun} must end up at the destination. "
            "Tow/transport is the chosen path because it satisfies that movement requirement directly.\n\n"
            "Underlying logic: action validity is determined by whether it moves the asset, "
            "not whether it merely moves the person."
        )
    if selected in {"yourself_first_walk", "yourself_first_drive", "walk"}:
        return (
            "The key assumption is that this branch is only about getting yourself to the location first. "
            "That can be practical, but it does not move the asset by itself.\n\n"
            "Underlying logic: separate person-movement from asset-movement. "
            "If the end goal is asset-at-destination, a second move/tow step is still required."
        )
    return ""


def _is_constraint_like_query(text: str) -> bool:
    t = str(text or "").lower()
    if not t:
        return False
    if not (("should i" in t) or ("do i" in t)):
        return False
    if not (("walk" in t) and ("drive" in t)):
        return False
    return any(k in t for k in ("wash", "service", "repair", "fuel", "charge", "park", "move", "tow", "transport", "car", "train", "truck", "van", "bus", "motorbike", "motorcycle", "bike", "scooter", "boat", "ship", "tram"))


def _looks_like_constraint_explain_followup(*, state: Any, user_text: str) -> bool:
    if not _EXPLAIN_LOGIC_RE.search(str(user_text or "")):
        return False
    prev_user = str(getattr(state, "last_user_text", "") or "")
    prev_asst = str(getattr(state, "last_assistant_text", "") or "")
    return bool(
        _is_constraint_like_query(prev_user)
        and ("destination" in prev_asst.lower() or "hard precondition" in prev_asst.lower() or "source: contextual" in prev_asst.lower())
    )


def _history_has_constraint_like_query(history: List[dict]) -> bool:
    for m in reversed(list(history or [])):
        if str(m.get("role", "")).lower() != "user":
            continue
        c = str(m.get("content", "") or "")
        if _is_constraint_like_query(c):
            return True
    return False


def _history_has_constraint_answer_signal(history: List[dict]) -> bool:
    """Detect recent assistant outputs that look like constraint-lane grounded answers."""
    if not history:
        return False
    for m in reversed(list(history or [])[-8:]):
        if str(m.get("role", "")).lower() != "assistant":
            continue
        c = str(m.get("content", "") or "").lower()
        if (
            ("destination" in c and "hard precondition" in c)
            or ("that gets" in c and "to the destination" in c)
            or ("walk there first" in c and "does not move" in c)
        ):
            return True
    return False


async def maybe_handle_fun_fr_raw(
    *,
    fun_mode: str,
    state: Any,
    user_text: str,
    session_id: str,
    history_text_only: List[dict],
    facts_block: str,
    constraints_block: str,
    scratchpad_grounded: bool,
    scratchpad_quotes: List[str],
    lock_active: bool,
    stream: bool,
    sensitive_override_once: bool,
    state_solver_answer: str,
    run_sync: Callable[..., Any],
    run_fun: Optional[Callable[..., str]],
    run_raw: Optional[Callable[..., str]],
    run_serious: Callable[..., str],
    run_fun_rewrite_fallback: Callable[..., str],
    call_model_prompt: Callable[..., str],
    no_op_vodka_cls: Any,
    select_fun_style_seed: Callable[..., Dict[str, Any]],
    serious_max_tokens_for_query: Callable[[str], int],
    is_argumentative_prompt: Callable[[str], bool],
    is_argumentatively_complete: Callable[[str], bool],
    fallback_with_mode_header: Callable[[str, str], str],
    finalize_chat_response: Callable[..., Any],
) -> Optional[Any]:
    if fun_mode == "fun":
        # Preserve deterministic solver output exactly when present.
        # We only decorate with a FUN header/quote; we do not run a rewrite pass.
        if (state_solver_answer or "").strip():
            base = (state_solver_answer or "").strip()
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base)
            quote = str(sel.get("seed") or "")
            base = _maybe_append_fun_train_kicker(
                text=base,
                user_text=user_text,
                session_id=session_id,
                seed_quote=quote,
            )
            text = f'[FUN] "{quote}"\n\n{base}' if quote else f"[FUN]\n\n{base}"
            return finalize_chat_response(
                text=text,
                user_text=user_text,
                state=state,
                lock_active=lock_active,
                scratchpad_grounded=scratchpad_grounded,
                scratchpad_quotes=scratchpad_quotes,
                has_facts_block=bool((facts_block or "").strip()),
                stream=stream,
                mode="fun",
                sensitive_override_once=sensitive_override_once,
                deterministic_state_solver=True,
            )

        if run_fun is None:
            base = (state_solver_answer or "").strip()
            if not base:
                base = (await run_sync(
                    run_serious,
                    session_id=session_id,
                    user_text=user_text,
                    history=history_text_only,
                    vodka=no_op_vodka_cls(),
                    call_model=call_model_prompt,
                    facts_block=facts_block,
                    constraints_block=constraints_block,
                    thinker_role="thinker",
                    max_tokens=serious_max_tokens_for_query(user_text),
                )).strip()
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base)
            quote = str(sel.get("seed") or "")
            text = f'[FUN] "{quote}"\n\n{base}' if quote else f"[FUN]\n\n{base}"
        else:
            base_preview = (state_solver_answer or "").strip()
            if not base_preview:
                base_preview = (await run_sync(
                    run_serious,
                    session_id=session_id,
                    user_text=user_text,
                    history=history_text_only,
                    vodka=no_op_vodka_cls(),
                    call_model=call_model_prompt,
                    facts_block=facts_block,
                    constraints_block=constraints_block,
                    thinker_role="thinker",
                    max_tokens=serious_max_tokens_for_query(user_text),
                )).strip()
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base_preview)
            pool = list(sel.get("pool") or [])
            seed = str(sel.get("seed") or "")

            styled = (await run_sync(
                run_fun,
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                facts_block=facts_block,
                quote_pool=pool,
                seed_override=seed,
                base_answer_override=base_preview,
                vodka=no_op_vodka_cls(),
                call_model=call_model_prompt,
                thinker_role="thinker",
            )).strip()
            lines = styled.splitlines()
            if lines:
                q = lines[0].strip()
                if q and not (q.startswith('"') and q.endswith('"')):
                    q = '"' + q.strip('"') + '"'
                lines[0] = f"[FUN] {q}" if q else "[FUN]"
                text = "\n".join(lines)
            else:
                text = "[FUN]"

            if is_argumentative_prompt(user_text) and not is_argumentatively_complete(text):
                text = fallback_with_mode_header(text, base_preview)

        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="fun",
            sensitive_override_once=sensitive_override_once,
            deterministic_state_solver=bool((state_solver_answer or "").strip()),
        )

    if fun_mode == "fun_rewrite":
        base_preview = (state_solver_answer or "").strip()
        if not base_preview:
            base_preview = _deterministic_constraint_explanation(state=state, user_text=user_text).strip()
        if not base_preview:
            base_preview = (await run_sync(
                run_serious,
                session_id=session_id,
                user_text=user_text,
                history=history_text_only,
                vodka=no_op_vodka_cls(),
                call_model=call_model_prompt,
                facts_block=facts_block,
                constraints_block=constraints_block,
                thinker_role="thinker",
                max_tokens=serious_max_tokens_for_query(user_text),
            )).strip()
        constraint_like = _is_constraint_like_query(user_text)
        constraint_explain_followup = (
            _looks_like_constraint_explain_followup(state=state, user_text=user_text)
            or (_EXPLAIN_LOGIC_RE.search(str(user_text or "")) and _history_has_constraint_like_query(history_text_only))
        )
        if _EXPLAIN_LOGIC_RE.search(str(user_text or "")) and (
            ("source: contextual" in str(getattr(state, "last_assistant_text", "") or "").lower())
            or _history_has_constraint_answer_signal(history_text_only)
        ):
            constraint_explain_followup = True
        fr_constraint_context = bool(constraint_like or constraint_explain_followup)
        if fr_constraint_context:
            safe_base = str(base_preview or "").strip()
            if safe_base and ("source:" not in safe_base.lower()):
                safe_base = safe_base.rstrip() + "\nSource: Contextual"
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=safe_base or base_preview)
            quote = str(sel.get("seed") or "")
            qline = f'"{quote}"' if quote else '""'
            text = f"[FUN REWRITE] {qline}\n\n{safe_base or base_preview}"
            return finalize_chat_response(
                text=text,
                user_text=user_text,
                state=state,
                lock_active=lock_active,
                scratchpad_grounded=scratchpad_grounded,
                scratchpad_quotes=scratchpad_quotes,
                has_facts_block=bool((facts_block or "").strip()),
                stream=stream,
                mode="fun_rewrite",
                sensitive_override_once=sensitive_override_once,
                deterministic_state_solver=bool((state_solver_answer or "").strip()) or fr_constraint_context,
            )
        text = (await run_sync(
            run_fun_rewrite_fallback,
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=no_op_vodka_cls(),
            facts_block=facts_block,
            state=state,
            base_override=base_preview,
        )).strip()
        # Hybrid FR guard: style rewrite must preserve base semantics.
        # Deterministic checks first, semantic verifier only when ambiguous.
        text = _guard_fun_rewrite(
            user_text=user_text,
            base_text=base_preview,
            rewritten_text=text,
            call_model_prompt=call_model_prompt,
        )
        # Preserve FR header affordance on fallback.
        if text.strip() == base_preview.strip():
            sel = await run_sync(select_fun_style_seed, state=state, user_text=user_text, base_text=base_preview)
            quote = str(sel.get("seed") or "")
            qline = f'"{quote}"' if quote else '""'
            text = f"[FUN REWRITE] {qline}\n\n{base_preview}"
        if is_argumentative_prompt(user_text) and not is_argumentatively_complete(text):
            text = fallback_with_mode_header(text, base_preview)

        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="fun_rewrite",
            sensitive_override_once=sensitive_override_once,
            deterministic_state_solver=bool((state_solver_answer or "").strip()) or fr_constraint_context,
        )

    if state.raw_sticky and run_raw:
        text = (await run_sync(
            run_raw,
            session_id=session_id,
            user_text=user_text,
            history=history_text_only,
            vodka=no_op_vodka_cls(),
            call_model=call_model_prompt,
            facts_block=facts_block,
            constraints_block=constraints_block,
            thinker_role="thinker",
        )).strip()
        return finalize_chat_response(
            text=text,
            user_text=user_text,
            state=state,
            lock_active=lock_active,
            scratchpad_grounded=scratchpad_grounded,
            scratchpad_quotes=scratchpad_quotes,
            has_facts_block=bool((facts_block or "").strip()),
            stream=stream,
            mode="raw",
            sensitive_override_once=sensitive_override_once,
        )

    return None
