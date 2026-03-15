"""Deterministic pairwise ranking worker for >>judge."""

from __future__ import annotations

import json
import os
import re
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


def _judge_prompt(criterion: str, item_a: str, item_b: str, *, retry: bool = False) -> str:
    if retry:
        return (
            "You are a strict comparator.\n"
            "Return ONLY one line in this exact format:\n"
            "VERDICT: A|B|TIE\n\n"
            f"Criterion: {criterion}\n"
            f"Option A: {item_a}\n"
            f"Option B: {item_b}\n"
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


def _score_verdict(verdict: str) -> Tuple[float, float]:
    if verdict == "A":
        return 1.0, 0.0
    if verdict == "B":
        return 0.0, 1.0
    return 0.5, 0.5


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
) -> JudgeRun:
    scores: Dict[str, float] = {item: 0.0 for item in items}
    comparisons: List[JudgeComparison] = []

    for idx_a, idx_b in combinations(range(len(items)), 2):
        item_a = items[idx_a]
        item_b = items[idx_b]
        # Anti-positional-bias pass: evaluate each pair in both orders.
        for order, left, right in (("ab", item_a, item_b), ("ba", item_b, item_a)):
            raw = call_model_prompt_fn(
                role=role,
                prompt=_judge_prompt(criterion, left, right, retry=False),
                max_tokens=48,
                temperature=0.0,
                top_p=1.0,
            )
            verdict = _parse_verdict(raw)
            if verdict is None:
                # Single strict retry path (middle-ground latency/correctness tradeoff).
                retry_raw = call_model_prompt_fn(
                    role=role,
                    prompt=_judge_prompt(criterion, left, right, retry=True),
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

            s_left, s_right = _score_verdict(verdict)
            scores[left] += s_left
            scores[right] += s_right
            reasoning = _parse_reasoning(raw)
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
    )

    if verbose:
        run.audit_jsonl_path = _write_audit_jsonl(run=run, audit_dir=audit_dir)
    return run


def format_judge_run(run: JudgeRun) -> str:
    if not run.ok:
        return run.error or "[judge] failed"

    lines: List[str] = ["[judge] ranking", f"criterion: {run.criterion}", ""]
    for i, (item, score) in enumerate(run.ranked, 1):
        lines.append(f"{i}. {item} (score={score:.2f})")
    lines.extend(
        [
            "",
            f"comparisons: {run.comparison_count}",
            f"Judge confidence: {run.confidence}",
        ]
    )
    if run.audit_jsonl_path:
        lines.append(f"audit_jsonl: {run.audit_jsonl_path}")
    return "\n".join(lines)
