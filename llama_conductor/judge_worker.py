"""Deterministic pairwise ranking worker for >>judge."""

from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple


USAGE_TEXT = (
    "[judge] usage: >>judge [criterion] : item1, item2, item3 [--verbose]\n"
    "[judge] constraints: 2-4 items, non-empty criterion, comma-separated item list"
)


@dataclass
class JudgeComparison:
    criterion: str
    item_a: str
    item_b: str
    order: str
    verdict: str
    reasoning: str
    raw_output: str = ""


@dataclass
class JudgeRun:
    ok: bool
    criterion: str = ""
    items: List[str] = field(default_factory=list)
    ranked: List[Tuple[str, float]] = field(default_factory=list)
    comparisons: List[JudgeComparison] = field(default_factory=list)
    comparison_count: int = 0
    confidence: str = "low"
    error: str = ""
    verbose: bool = False
    audit_jsonl_path: str = ""
    evidence_source: str = "none"
    evidence_locked_indices: List[int] = field(default_factory=list)
    evidence_chars: int = 0
    # Diagnostic split: raw locked scratch chars vs effective judge evidence chars.
    scratch_locked_raw_chars: int = 0
    judge_evidence_chars: int = 0
    bias_detected: bool = False
    bias_repairs: int = 0


def parse_judge_payload(payload: str, *, min_items: int = 2, max_items: int = 4) -> Tuple[str, List[str], bool, str]:
    text = str(payload or "").strip()
    if not text:
        return "", [], False, USAGE_TEXT

    verbose = False
    parts: List[str] = []
    for tok in text.split():
        if tok == "--verbose":
            verbose = True
            continue
        if tok.startswith("--"):
            return "", [], False, f"[judge] unknown flag: {tok}\n{USAGE_TEXT}"
        parts.append(tok)

    normalized = " ".join(parts).strip()
    if ":" not in normalized:
        return "", [], verbose, f"[judge] missing ':' delimiter between criterion and items\n{USAGE_TEXT}"

    criterion, item_blob = normalized.split(":", 1)
    criterion = criterion.strip()
    if not criterion:
        return "", [], verbose, f"[judge] criterion cannot be empty\n{USAGE_TEXT}"

    items = [s.strip() for s in item_blob.split(",") if s.strip()]
    if not (min_items <= len(items) <= max_items):
        return "", [], verbose, f"[judge] invalid item count: {len(items)} (expected {min_items}-{max_items})\n{USAGE_TEXT}"

    seen: set[str] = set()
    for item in items:
        k = item.lower()
        if k in seen:
            return "", [], verbose, f"[judge] duplicate item detected: '{item}'"
        seen.add(k)

    return criterion, items, verbose, ""


def _judge_prompt(
    criterion: str,
    item_a: str,
    item_b: str,
    *,
    retry: bool = False,
    repair: bool = False,
    evidence_block: str = "",
) -> str:
    evidence = str(evidence_block or "").strip()
    evidence_text = ""
    if evidence:
        evidence_text = (
            "\n"
            "Evidence (authoritative for this comparison):\n"
            f"{evidence}\n"
            "Use this evidence directly. If evidence is insufficient to separate A/B, return TIE.\n"
        )
    repair_text = ""
    if repair:
        repair_text = (
            "Judge the underlying items, not the item order.\n"
            "Return exactly one line in this format:\n"
            "WINNER: <exact item text from Candidate 1 or Candidate 2, or TIE>\n"
            "Renaming candidate order must not change which underlying item wins.\n"
        )
    if retry:
        if repair:
            return (
                "You are a strict comparator.\n"
                "Return ONLY one line in this exact format:\n"
                "WINNER: <exact item text from Candidate 1 or Candidate 2, or TIE>\n\n"
                f"Criterion: {criterion}\n"
                f"Candidate 1: {item_a}\n"
                f"Candidate 2: {item_b}\n"
                f"{repair_text}"
                f"{evidence_text}"
            )
        return (
            "You are a strict comparator.\n"
            "Return ONLY one line in this exact format:\n"
            "VERDICT: A|B|TIE\n\n"
            f"Criterion: {criterion}\n"
            f"Option A: {item_a}\n"
            f"Option B: {item_b}\n"
            f"{repair_text}"
            f"{evidence_text}"
        )
    return (
        "You are a strict comparator.\n"
        "Choose which option better satisfies the criterion.\n"
        "Respond with exactly two lines:\n"
        "REASON: <1 short sentence, max 24 words>\n"
        "VERDICT: A|B|TIE\n"
        "No extra lines.\n\n"
        f"Criterion: {criterion}\n"
        f"Option A: {item_a}\n"
        f"Option B: {item_b}\n"
        f"{repair_text}"
        f"{evidence_text}"
    )


_VERDICT_RE = re.compile(r"\b(A|B|TIE)\b", flags=re.IGNORECASE)
_VERDICT_LINE_RE = re.compile(r"VERDICT\s*:\s*(A|B|TIE)\b", flags=re.IGNORECASE)
_REASON_LINE_RE = re.compile(r"REASON\s*:\s*(.+)$", flags=re.IGNORECASE)


def _parse_verdict(raw: str) -> Optional[str]:
    text = str(raw or "")
    m = _VERDICT_LINE_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


_WINNER_LINE_RE = re.compile(r"WINNER\s*:\s*(.+)$", flags=re.IGNORECASE)


def _parse_winner_identity(raw: str, candidates: List[str]) -> Optional[str]:
    """Return the exact candidate text chosen by a repair response.

    The repair contract prefers an exact item name so the mapping back to the
    underlying item is not mediated by A/B slots. Fallback to legacy A/B tokens
    is kept only for resilience.
    """
    text = str(raw or "").strip()
    if not text:
        return None

    m = _WINNER_LINE_RE.search(text)
    if m:
        declared = m.group(1).strip()
        if declared.upper() == "TIE":
            return "TIE"
        declared_norm = _normalize_text(declared)
        declared_tokens = set(_identity_tokens(declared))
        best_candidate: Optional[str] = None
        best_score = 0
        for candidate in candidates:
            candidate_norm = _normalize_text(candidate)
            if candidate_norm == declared_norm:
                return candidate
            candidate_tokens = set(_identity_tokens(candidate))
            if not declared_tokens or not candidate_tokens:
                continue
            overlap = len(declared_tokens & candidate_tokens)
            if overlap <= 0:
                continue
            containment_bonus = 2 if declared_tokens <= candidate_tokens or candidate_tokens <= declared_tokens else 0
            candidate_score = overlap * 10 + containment_bonus
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate
            elif candidate_score == best_score:
                best_candidate = None
        if best_candidate is not None and best_score >= 10:
            return best_candidate

    verdict = _parse_verdict(text)
    if verdict in {"A", "B"} and len(candidates) >= 2:
        # Legacy fallback only. The repair prompt should now return WINNER:<item>.
        idx = 0 if verdict == "A" else 1
        return candidates[idx]
    if verdict == "TIE":
        return "TIE"
    return None


def _parse_reasoning(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    for line in text.splitlines():
        m = _REASON_LINE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    # fallback: first non-empty non-verdict line, trimmed
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if _VERDICT_LINE_RE.search(ln):
            continue
        return ln[:220]
    return ""


def _normalize_text(s: str) -> str:
    t = str(s or "").lower().strip()
    if not t:
        return ""
    trans = str.maketrans({ch: " " for ch in string.punctuation})
    t = t.translate(trans)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _identity_tokens(s: str) -> List[str]:
    text = _normalize_text(s)
    if not text:
        return []
    tokens: List[str] = []
    for tok in text.split():
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"i(\d+)", tok)
        if m:
            tokens.append(m.group(1))
            continue
        tokens.append(tok)
    return tokens


def _detect_reason_preference(reason: str, left: str, right: str) -> Optional[str]:
    """Return 'A'/'B' when reason clearly prefers one side, else None.

    Low-blast heuristic:
    - if reason mentions exactly one option label (normalized), treat that as preferred.
    - if both/neither are present, do not infer preference.
    """
    r = _normalize_text(reason)
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not r or not a or not b:
        return None
    has_a = a in r
    has_b = b in r
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return None


def _score_verdict(verdict: str) -> Tuple[float, float]:
    if verdict == "A":
        return 1.0, 0.0
    if verdict == "B":
        return 0.0, 1.0
    return 0.5, 0.5


def _winner_for_verdict(left: str, right: str, verdict: str) -> Optional[str]:
    verdict_u = str(verdict or "").strip().upper()
    if verdict_u == "A":
        return left
    if verdict_u == "B":
        return right
    return None


def _stable_sort_pair(left: str, right: str) -> Tuple[str, str]:
    left_key = _normalize_text(left)
    right_key = _normalize_text(right)
    if left_key <= right_key:
        return left, right
    return right, left


def _confidence_tier(ranked: List[Tuple[str, float]]) -> str:
    if len(ranked) < 2:
        return "low"
    top = ranked[0][1]
    nxt = ranked[1][1]
    if abs(top - nxt) < 1e-9:
        return "low"
    margin = top - nxt
    if margin >= 1.5:
        return "high"
    if margin >= 0.5:
        return "medium"
    return "low"


def _timestamp_compact_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_audit_jsonl(*, run: JudgeRun, audit_dir: str) -> str:
    os.makedirs(audit_dir, exist_ok=True)
    path = os.path.join(audit_dir, f"judge_audit_{_timestamp_compact_utc()}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for cmp in run.comparisons:
            row: Dict[str, object] = {
                "criterion": cmp.criterion,
                "item_a": cmp.item_a,
                "item_b": cmp.item_b,
                "order": cmp.order,
                "verdict": cmp.verdict,
                "reasoning": cmp.reasoning,
                "raw_output": cmp.raw_output,
                "evidence_source": run.evidence_source,
                "evidence_locked_indices": list(run.evidence_locked_indices),
                "evidence_chars": int(run.evidence_chars),
                "scratch_locked_raw_chars": int(run.scratch_locked_raw_chars),
                "judge_evidence_chars": int(run.judge_evidence_chars),
                "bias_detected": bool(run.bias_detected),
                "bias_repairs": int(run.bias_repairs),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def run_judge(
    *,
    criterion: str,
    items: List[str],
    role: str,
    call_model_prompt_fn: Callable[..., str],
    verbose: bool = False,
    audit_dir: str = os.path.join("total_recall", "judge"),
    evidence_block: str = "",
    evidence_source: str = "none",
    evidence_locked_indices: Optional[List[int]] = None,
    scratch_locked_raw_chars: int = 0,
) -> JudgeRun:
    scores: Dict[str, float] = {item: 0.0 for item in items}
    comparisons: List[JudgeComparison] = []
    normalized_evidence = str(evidence_block or "").strip()
    locked_indices = [int(i) for i in (evidence_locked_indices or []) if int(i) > 0]
    bias_detected = False
    bias_repairs = 0

    for idx_a, idx_b in combinations(range(len(items)), 2):
        item_a = items[idx_a]
        item_b = items[idx_b]
        pair_records: List[Tuple[str, str, str, str, str]] = []
        # Anti-positional-bias pass: evaluate each pair in both orders.
        for order, left, right in (("ab", item_a, item_b), ("ba", item_b, item_a)):
            raw = call_model_prompt_fn(
                role=role,
                prompt=_judge_prompt(
                    criterion,
                    left,
                    right,
                    retry=False,
                    evidence_block=normalized_evidence,
                ),
                max_tokens=48,
                temperature=0.0,
                top_p=1.0,
            )
            verdict = _parse_verdict(raw)
            if verdict is None:
                # Single strict retry path (middle-ground latency/correctness tradeoff).
                retry_raw = call_model_prompt_fn(
                    role=role,
                    prompt=_judge_prompt(
                        criterion,
                        left,
                        right,
                        retry=True,
                        evidence_block=normalized_evidence,
                    ),
                    max_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                )
                verdict = _parse_verdict(retry_raw)
                if verdict is None:
                    return JudgeRun(
                        ok=False,
                        criterion=criterion,
                        items=list(items),
                        error=(
                            "[judge] pairwise verdict parse failure.\n"
                            f"[judge] expected A|B|TIE, got: {str(retry_raw or raw or '').strip()[:160]}"
                        ),
                        verbose=verbose,
                    )
                raw = retry_raw

            reasoning = _parse_reasoning(raw)
            inferred = _detect_reason_preference(reasoning, left, right)
            if inferred is not None and inferred != verdict:
                # One strict retry when reason text and verdict conflict.
                retry_raw = call_model_prompt_fn(
                    role=role,
                    prompt=_judge_prompt(
                        criterion,
                        left,
                        right,
                        retry=True,
                        evidence_block=normalized_evidence,
                    ),
                    max_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                )
                retry_verdict = _parse_verdict(retry_raw)
                if retry_verdict in {"A", "B", "TIE"}:
                    verdict = str(retry_verdict)
                else:
                    verdict = "TIE"
                retry_reason = _parse_reasoning(retry_raw)
                reasoning = retry_reason or "[verdict-only retry applied]"
                raw = f"{str(raw or '')}\n[retry_verdict_only]\n{str(retry_raw or '')}"

            pair_records.append((order, left, right, verdict, reasoning))
            comparisons.append(
                JudgeComparison(
                    criterion=criterion,
                    item_a=left,
                    item_b=right,
                    order=order,
                    verdict=verdict,
                    reasoning=reasoning,
                    raw_output=str(raw or ""),
                )
            )

        pair_winners = [
            _winner_for_verdict(left, right, verdict)
            for _order, left, right, verdict, _reasoning in pair_records
        ]
        pair_winners = [w for w in pair_winners if w is not None]
        if len(pair_winners) >= 2 and len(set(pair_winners)) > 1:
            bias_detected = True
            canonical_left, canonical_right = _stable_sort_pair(item_a, item_b)
            repair_raw = call_model_prompt_fn(
                role=role,
                prompt=_judge_prompt(
                    criterion,
                    canonical_left,
                    canonical_right,
                    retry=True,
                    repair=True,
                    evidence_block=normalized_evidence,
                ),
                max_tokens=24,
                temperature=0.0,
                top_p=1.0,
            )
            repair_winner = _parse_winner_identity(repair_raw, [canonical_left, canonical_right])
            if repair_winner is None:
                repair_winner = "TIE"
            repair_reasoning = _parse_reasoning(repair_raw) or "[bias repair retry applied]"
            comparisons.append(
                JudgeComparison(
                    criterion=criterion,
                    item_a=canonical_left,
                    item_b=canonical_right,
                    order="repair",
                    verdict=repair_winner,
                    reasoning=repair_reasoning,
                    raw_output=str(repair_raw or ""),
                )
            )
            bias_repairs += 1
            if repair_winner == "TIE":
                s_left, s_right = _score_verdict("TIE")
                scores[canonical_left] += s_left
                scores[canonical_right] += s_right
            elif repair_winner == canonical_left:
                scores[canonical_left] += 1.0
            elif repair_winner == canonical_right:
                scores[canonical_right] += 1.0
        else:
            for _order, left, right, verdict, _reasoning in pair_records:
                s_left, s_right = _score_verdict(verdict)
                scores[left] += s_left
                scores[right] += s_right

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    run = JudgeRun(
        ok=True,
        criterion=criterion,
        items=list(items),
        ranked=ranked,
        comparisons=comparisons,
        comparison_count=len(comparisons),
        confidence=_confidence_tier(ranked),
        verbose=verbose,
        evidence_source=(str(evidence_source or "").strip() or "none"),
        evidence_locked_indices=locked_indices,
        evidence_chars=len(normalized_evidence),
        scratch_locked_raw_chars=max(0, int(scratch_locked_raw_chars or 0)),
        judge_evidence_chars=len(normalized_evidence),
        bias_detected=bias_detected,
        bias_repairs=bias_repairs,
    )

    if verbose:
        run.audit_jsonl_path = _write_audit_jsonl(run=run, audit_dir=audit_dir)
    return run


def format_judge_run(run: JudgeRun) -> str:
    if not run.ok:
        return run.error or "[judge] failed"

    lines: List[str] = []
    if str(getattr(run, "evidence_source", "") or "").strip().lower() in {"", "none"}:
        lines.extend(
            [
                "[judge] No evidence loaded. Running ungrounded.",
                "[judge] Confidence reflects model priors, not VERIFIED FACTS.",
                "[judge] For better outcomes, use >>scratch pathway.",
                "",
            ]
        )

    lines.extend(["[judge] ranking", f"criterion: {run.criterion}", ""])
    for i, (item, score) in enumerate(run.ranked, 1):
        lines.append(f"{i}. {item} (score={score:.2f})")
    if run.ranked:
        top_score = float(run.ranked[0][1])
        winners = [item for item, score in run.ranked if abs(float(score) - top_score) < 1e-9]
        if len(winners) == 1:
            lines.append("")
            lines.append(f"Winner: {winners[0]}")
        else:
            lines.append("")
            lines.append(f"Winner: TIE ({', '.join(winners)})")
    lines.extend(
        [
            "",
            f"comparisons: {run.comparison_count}",
            f"Judge confidence: {run.confidence}",
        ]
    )
    if run.bias_detected:
        lines.append("[judge] warning: positional bias detected; repaired via canonical retry.")
    if run.audit_jsonl_path:
        lines.append(f"audit_jsonl: {run.audit_jsonl_path}")
    return "\n".join(lines)
