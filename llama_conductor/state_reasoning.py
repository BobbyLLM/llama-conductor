"""Deterministic state-transition and constraint-decision helpers.

Purpose:
- classify machine-checkable state-transition prompts
- solve common bounded-capacity transfer/addition and balance-update problems deterministically
- solve high-precision feasibility decisions via hard preconditions
- fail loud on partial parse for high-signal state questions
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Optional


_NUM_RE = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)")
_STATE_INTENT_RE = re.compile(
    r"\b("
    r"add|pour|transfer|overflow|capacity|cup|container|bucket|bottle|already has|over capacity|"
    r"inventory|stock|queue|seat|seats|slot|slots|round|rounds|batch|batches|schedule|scheduled|"
    r"passenger|passengers|account|balance|budget|wallet|warehouse|clinic|room|rooms|worker|workers|"
    r"board|leave|arrive|ship|receive|deposit|withdraw|spend|remaining|left"
    r")\b",
    re.IGNORECASE,
)
_STATE_VERB_HINT_RE = re.compile(
    r"\b(add|added|adding|pour|transfer|receive|received|receives|board|boarded|boards|"
    r"arrive|arrived|arrives|leave|left|leaves|remove|removed|spend|spent|deposit|deposited|"
    r"withdraw|withdrew|ship|shipped|ships|restock|restocked|remaining|remain|remains|"
    r"process|processed|processes|handle|handled|handles|complete|completed|completes|finish|finished)\b",
    re.IGNORECASE,
)
_ASK_OVERFLOW_RE = re.compile(r"\b(over capacity|overflow|spill(?:ed|s|ing)?)\b", re.IGNORECASE)

_FIRST_ADD_RE = re.compile(
    r"""
    (?P<cap>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)\s*cup
    [^.!?\n]{0,120}?
    with\s*(?P<start>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)
    [^.!?\n]{0,160}?
    (?:add|added|adding|pour(?:\s+in)?|put(?:\s+in)?)\s*(?P<delta>\d+(?:\.\d+)?)\s*(?:more\s*)?(?:oz|ounce|ounces|fluid\s*ounces?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FIRST_POUR_RE = re.compile(
    r"""
    pour\s*(?P<input>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)\s*(?:of\s+[a-z]+\s*)?
    into\s*(?:a\s*)?(?P<cap1>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)\s*cup
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SECOND_POUR_EVERYTHING_RE = re.compile(
    r"""
    then\s+pour\s+everything[^\n]{0,200}?
    into\s*(?:a\s*)?(?P<cap2>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)\s*cup
    (?:[^\n]{0,120}?already\s+has\s*(?P<start2>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ASK_FINAL_TARGET_RE = re.compile(
    r"\bhow much\b[^.!?\n]{0,120}\bin\b[^.!?\n]{0,80}\b(?:cup|container|bucket|bottle)\b",
    re.IGNORECASE,
)
_DIRECT_TOTAL_CAP_RE = re.compile(
    r"""
    (?P<total>\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|fluid\s*ounces?)
    [^.!?\n]{0,120}?
    in(?:\s+a)?\s+(?P<cap>\d+(?:\.\d+)?)(?:\s*-\s*|\s+)?(?:oz|ounce|ounces|fluid\s*ounces?)\s*cup
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BALANCE_START_STRONG_RE = re.compile(
    r"""
    (?:starts?\s+with|starting\s+with|currently\s+has|already\s+has)\s*
    (?P<start>\d+(?:\.\d+)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BALANCE_START_WEAK_RE = re.compile(
    r"""
    (?:has|with)\s*(?P<start>\d+(?:\.\d+)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BALANCE_CAP_RE = re.compile(
    r"""
    (?P<cap>\d+(?:\.\d+)?)\s*
    (?:seat|seats|slot|slots|capacity|max(?:imum)?|limit|spaces?|tickets?)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ASK_REMAINING_RE = re.compile(r"\b(how much|how many|remaining|left|now)\b", re.IGNORECASE)
_SCHED_TASKS_RE = re.compile(
    r"\b(?P<n>\d+(?:\.\d+)?)\s*(tasks?|jobs?|patients?|orders?|tickets?|items?|units?)\b",
    re.IGNORECASE,
)
_SCHED_RESOURCE_RE = re.compile(
    r"\b(?P<n>\d+(?:\.\d+)?)\s*(rooms?|lanes?|servers?|workers?|stations?|desks?)\b",
    re.IGNORECASE,
)
_SCHED_PER_RESOURCE_RE = re.compile(
    r"\b(?P<n>\d+(?:\.\d+)?)\s*(tasks?|jobs?|patients?|orders?|tickets?|items?|units?)\s*per\s*"
    r"(room|lane|server|worker|station|desk|slot|round|batch)\b",
    re.IGNORECASE,
)
_SCHED_ASK_RE = re.compile(r"\b(how many|slots?|rounds?|batches?|waves?)\b", re.IGNORECASE)

_POS_VERBS = (
    "add", "added", "adding",
    "receive", "received", "receives",
    "gain", "gained", "gains",
    "deposit", "deposited", "deposits",
    "board", "boarded", "boards",
    "enter", "entered", "enters",
    "arrive", "arrived", "arrives",
    "restock", "restocked", "restocks",
)
_NEG_VERBS = (
    "remove", "removed", "removes",
    "spend", "spent", "spends",
    "withdraw", "withdrew", "withdrawn", "withdraws",
    "leave", "left", "leaves",
    "exit", "exited", "exits",
    "sell", "sold", "sells",
    "ship", "shipped", "ships",
    "use", "used", "uses",
    "offload", "offloaded", "offloads",
)
_POS_VERB_RE = "|".join(re.escape(v) for v in _POS_VERBS)
_NEG_VERB_RE = "|".join(re.escape(v) for v in _NEG_VERBS)
_POS_VERB_NUM_RE = re.compile(
    rf"\b(?P<verb>{_POS_VERB_RE})\b\s*(?:an?\s*)?(?P<num>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NEG_VERB_NUM_RE = re.compile(
    rf"\b(?P<verb>{_NEG_VERB_RE})\b\s*(?:an?\s*)?(?P<num>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_POS_NUM_VERB_RE = re.compile(
    rf"(?P<num>\d+(?:\.\d+)?)\s*(?:people|persons|passengers|units?|items?|tickets?|oz|ounces?|dollars?|bucks|usd|\$)?\s*(?P<verb>{_POS_VERB_RE})\b",
    re.IGNORECASE,
)
_NEG_NUM_VERB_RE = re.compile(
    rf"(?P<num>\d+(?:\.\d+)?)\s*(?:people|persons|passengers|units?|items?|tickets?|oz|ounces?|dollars?|bucks|usd|\$)?\s*(?P<verb>{_NEG_VERB_RE})\b",
    re.IGNORECASE,
)

_DECISION_INTENT_RE = re.compile(r"\b(should i|should we|do i|do we)\b", re.IGNORECASE)
_DECISION_OR_RE = re.compile(r"\bor\b", re.IGNORECASE)
_OPT_WALK_RE = re.compile(r"\bwalk(?:ing)?\b", re.IGNORECASE)
_OPT_DRIVE_RE = re.compile(r"\bdrive|driving\b", re.IGNORECASE)
_ASSET_PATTERN = r"(?:car|truck|van|bus|train|tram|boat|ship|motorcycle|motorbike|bike|scooter)"
_GENERIC_ASSET_ACTION_RE = re.compile(
    r"\b(?:wash|service|repair|fuel|charge|park|move|tow|transport|clean)\s+(?:my|the)\s+(?P<asset>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3})\b",
    re.IGNORECASE,
)
_ASSET_TASK_RE = re.compile(
    r"\b("
    rf"wash\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|{_ASSET_PATTERN}\s+wash"
    rf"|take\s+(?:my|the)\s+{_ASSET_PATTERN}\s+to"
    rf"|service\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|repair\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|fuel\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|charge\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|park\s+(?:my|the)\s+{_ASSET_PATTERN}"
    rf"|(?:my|the)\s+{_ASSET_PATTERN}\s+.*\b(mechanic|garage|dealership|service\s+center|wash|gas\s+station|petrol\s+station|charging\s+station|depot)\b"
    r")\b",
    re.IGNORECASE,
)
_FOLLOWUP_QUERY_RE = re.compile(
    r"\b(then what|now what|what now|are you sure|still|so what|what should i do)\b",
    re.IGNORECASE,
)
_CONFUSION_RE = re.compile(
    r"\b(what\??|wtf|no idea|don't understand|dont understand|confused|huh)\b",
    re.IGNORECASE,
)
_CHOICE_COMPARISON_RE = re.compile(
    r"\b(difference|same|equivalent|what difference|aren't|arent)\b",
    re.IGNORECASE,
)
_CONSTRAINT_FUEL_FALSE_RE = re.compile(r"\b(out of fuel|no fuel|without fuel|fuel is empty|tank is empty)\b", re.IGNORECASE)
_CONSTRAINT_FUEL_TRUE_RE = re.compile(r"\b(have fuel|with fuel|fuel available|tank has fuel|refueled|fuelled)\b", re.IGNORECASE)
_DISTANCE_RE = re.compile(
    r"\b(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometer|kilometers|kilometre|kilometres|m|meter|meters|metre|metres)\b",
    re.IGNORECASE,
)
_EXPLICIT_REENGAGE_RE = re.compile(
    r"\b(back to|same decision|that decision|re-?engage|option\s*[123])\b",
    re.IGNORECASE,
)
_CORRECTION_INTENT_RE = re.compile(
    r"\b(i mean|i meant|no i meant|sorry i meant|rather than|not)\b",
    re.IGNORECASE,
)
_DELEGATE_DECISION_RE = re.compile(
    r"\b(you tell me|you decide|you choose|your call|pick for me|choose for me|idk|i dont know|i don't know|not sure)\b",
    re.IGNORECASE,
)
_CLARIFIER_EXPLAIN_RE = re.compile(
    r"\b(what do you mean|explain|which one|what are the options|options|huh|confused)\b",
    re.IGNORECASE,
)


@dataclass
class StateReasoningResult:
    handled: bool = False
    fail_loud: bool = False
    family: str = "other"
    answer: str = ""
    reason: str = ""
    frame: Optional[Dict[str, Any]] = None


def classify_query_family(query: str) -> str:
    t = " ".join(str(query or "").split())
    if not t:
        return "other"
    if (
        _STATE_INTENT_RE.search(t)
        and len(_NUM_RE.findall(t)) >= 2
        and (_STATE_VERB_HINT_RE.search(t) or _ASK_OVERFLOW_RE.search(t) or _ASK_REMAINING_RE.search(t))
    ):
        return "state_transition"
    if _is_constraint_decision_candidate(t):
        return "constraint_decision"
    return "other"


def solve_state_transition_query(query: str) -> StateReasoningResult:
    t = " ".join(str(query or "").split())
    if not t:
        return StateReasoningResult()

    fam = classify_query_family(t)
    if fam == "constraint_decision":
        return _solve_constraint_decision_query(t)
    if fam != "state_transition":
        return StateReasoningResult(family=fam)

    # Case A: single-container initial + add
    m_add = _FIRST_ADD_RE.search(t)
    if m_add:
        cap = _as_float(m_add.group("cap"))
        start = _as_float(m_add.group("start"))
        delta = _as_float(m_add.group("delta"))
        if cap is None or start is None or delta is None:
            return _fail_loud("Missing values for capacity/addition state update.")
        total = start + delta
        contained = min(cap, total)
        overflow = max(0.0, total - cap)
        if _ASK_OVERFLOW_RE.search(t):
            ans = (
                f"The cup is over capacity by {_fmt(overflow)} oz."
                f"\nSource: Contextual"
            )
        else:
            ans = (
                f"Total directed volume is {_fmt(total)} oz. "
                f"The cup capacity is {_fmt(cap)} oz, so the cup contains {_fmt(contained)} oz"
                f" and {_fmt(overflow)} oz overflows."
                f"\nSource: Contextual"
            )
        return StateReasoningResult(handled=True, family=fam, answer=ans, reason="capacity_add")

    # Case A2: direct stated total vs cup capacity.
    m_direct = _DIRECT_TOTAL_CAP_RE.search(t)
    if m_direct:
        total = _as_float(m_direct.group("total"))
        cap = _as_float(m_direct.group("cap"))
        if total is None or cap is None:
            return _fail_loud("Missing direct total/capacity values.")
        contained = min(cap, max(0.0, total))
        overflow = max(0.0, total - cap)
        if _ASK_OVERFLOW_RE.search(t):
            ans = f"The cup is over capacity by {_fmt(overflow)} oz.\nSource: Contextual"
        else:
            ans = (
                f"Stated volume is {_fmt(total)} oz and cup capacity is {_fmt(cap)} oz. "
                f"The cup can physically contain {_fmt(contained)} oz; {_fmt(overflow)} oz is overflow."
                f"\nSource: Contextual"
            )
        return StateReasoningResult(handled=True, family=fam, answer=ans, reason="direct_total_capacity")

    # Case B: pour-then-pour-everything chain
    m1 = _FIRST_POUR_RE.search(t)
    m2 = _SECOND_POUR_EVERYTHING_RE.search(t)
    if m1 and m2:
        inp = _as_float(m1.group("input"))
        cap1 = _as_float(m1.group("cap1"))
        cap2 = _as_float(m2.group("cap2"))
        start2 = _as_float(m2.group("start2")) if m2.group("start2") is not None else 0.0
        if inp is None or cap1 is None or cap2 is None or start2 is None:
            return _fail_loud("Missing values for transfer-chain state update.")

        first_contained = min(cap1, inp)
        first_overflow = max(0.0, inp - cap1)

        transfer = first_contained
        second_total = start2 + transfer
        second_contained = min(cap2, second_total)
        second_overflow = max(0.0, second_total - cap2)

        if _ASK_OVERFLOW_RE.search(t) and not _ASK_FINAL_TARGET_RE.search(t):
            ans = (
                f"Step-2 overflow is {_fmt(second_overflow)} oz "
                f"(total overflow across both steps is {_fmt(first_overflow + second_overflow)} oz)."
                f"\nSource: Contextual"
            )
        else:
            ans = (
                f"Step 1: {_fmt(inp)} oz into {_fmt(cap1)} oz cup -> {_fmt(first_contained)} oz in cup, "
                f"{_fmt(first_overflow)} oz overflow. "
                f"Step 2: pour {_fmt(transfer)} oz into {_fmt(cap2)} oz cup"
                f"{_prefill_suffix(start2)} -> {_fmt(second_contained)} oz in target cup, "
                f"{_fmt(second_overflow)} oz overflow at step 2."
                f"\nSource: Contextual"
            )
        return StateReasoningResult(handled=True, family=fam, answer=ans, reason="capacity_transfer_chain")

    # Case C: generic bounded/unbounded balance updates (inventory/queue/account-like).
    generic = _solve_generic_balance_update(t)
    if generic.handled:
        return generic

    # Case D: generic scheduling/resource-allocation (capacity per round/slot).
    sched = _solve_generic_schedule_allocation(t)
    if sched.handled:
        return sched

    # Strong state intent + numeric payload, but not parseable enough -> fail loud.
    return _fail_loud(
        "State-transition query detected but required variables were ambiguous. "
        "Provide explicit capacities (if any), starting amount, and each add/remove transfer."
    )


def _fail_loud(msg: str) -> StateReasoningResult:
    return StateReasoningResult(
        handled=True,
        fail_loud=True,
        family="state_transition",
        answer=f"Cannot verify this state update deterministically: {msg}\nSource: Contextual",
        reason="fail_loud_partial_parse",
    )


def _as_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _distance_km(text: str) -> Optional[float]:
    m = _DISTANCE_RE.search(text or "")
    if not m:
        return None
    val = _as_float(m.group("val"))
    if val is None:
        return None
    unit = (m.group("unit") or "").lower()
    if unit.startswith("km") or unit.startswith("kilo"):
        return val
    return val / 1000.0


def _yourself_first_distance_clause(km: float) -> str:
    if km < 1.5:
        return "walking is practical; driving is faster and lower-effort if you prefer"
    if km < 5.0:
        return "walking is doable but moderate effort; driving is the lower-effort option"
    return "walking is possible but high-effort; driving is the practical low-effort option"


def _is_explicit_reengage(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if _EXPLICIT_REENGAGE_RE.search(q):
        return True
    toks = _normalize_text_tokens(q)
    if toks & {"1", "2", "3"}:
        return True
    if {"walk", "drive"} <= toks:
        return True
    if {"move", "tow"} & toks:
        return True
    return False


def _is_correction_intent(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return bool(_CORRECTION_INTENT_RE.search(q))


def _fmt(v: float) -> str:
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _prefill_suffix(start2: float) -> str:
    if start2 <= 0:
        return ""
    return f" (already holding {_fmt(start2)} oz)"


def _extract_signed_ops(text: str) -> list[tuple[int, int, float, str]]:
    rows: list[tuple[int, int, float, str]] = []
    for m in _POS_VERB_NUM_RE.finditer(text):
        n = _as_float(m.group("num"))
        if n is not None:
            rows.append((m.start(), +1, n, m.group("verb").lower()))
    for m in _NEG_VERB_NUM_RE.finditer(text):
        n = _as_float(m.group("num"))
        if n is not None:
            rows.append((m.start(), -1, n, m.group("verb").lower()))
    for m in _POS_NUM_VERB_RE.finditer(text):
        n = _as_float(m.group("num"))
        if n is not None:
            rows.append((m.start(), +1, n, m.group("verb").lower()))
    for m in _NEG_NUM_VERB_RE.finditer(text):
        n = _as_float(m.group("num"))
        if n is not None:
            rows.append((m.start(), -1, n, m.group("verb").lower()))
    rows.sort(key=lambda r: r[0])
    dedup: list[tuple[int, int, float, str]] = []
    last_pos = -1
    for row in rows:
        if row[0] == last_pos:
            continue
        dedup.append(row)
        last_pos = row[0]
    return dedup


def _solve_generic_balance_update(text: str) -> StateReasoningResult:
    m_start = _BALANCE_START_STRONG_RE.search(text)
    if not m_start:
        m_start = _BALANCE_START_WEAK_RE.search(text)
    if not m_start:
        return StateReasoningResult()
    start = _as_float(m_start.group("start"))
    if start is None:
        return StateReasoningResult()
    ops = _extract_signed_ops(text)
    if not ops:
        return StateReasoningResult()

    cap = None
    m_cap = _BALANCE_CAP_RE.search(text)
    if m_cap:
        cap = _as_float(m_cap.group("cap"))

    delta_total = 0.0
    pieces: list[str] = []
    for _, sign, n, verb in ops:
        delta_total += (sign * n)
        op_char = "+" if sign > 0 else "-"
        pieces.append(f"{op_char}{_fmt(n)} ({verb})")

    raw_final = start + delta_total
    if cap is None:
        if _ASK_OVERFLOW_RE.search(text):
            return _fail_loud("Overflow requested but no capacity value was provided.")
        ans = (
            f"Start {_fmt(start)}; operations {', '.join(pieces)} -> final {_fmt(raw_final)}."
            f"\nSource: Contextual"
        )
        return StateReasoningResult(handled=True, family="state_transition", answer=ans, reason="generic_balance_unbounded")

    bounded_final = min(cap, max(0.0, raw_final))
    overflow = max(0.0, raw_final - cap)
    underflow = max(0.0, -raw_final)
    if _ASK_OVERFLOW_RE.search(text):
        ans = f"Over-capacity amount is {_fmt(overflow)}.\nSource: Contextual"
    else:
        parts = [
            f"Start {_fmt(start)}; operations {', '.join(pieces)} -> raw {_fmt(raw_final)}.",
            f"With capacity {_fmt(cap)}, contained amount is {_fmt(bounded_final)}.",
        ]
        if overflow > 0:
            parts.append(f"Overflow is {_fmt(overflow)}.")
        if underflow > 0:
            parts.append(f"Requested removals exceeded available by {_fmt(underflow)} before bounding.")
        ans = " ".join(parts) + "\nSource: Contextual"
    return StateReasoningResult(handled=True, family="state_transition", answer=ans, reason="generic_balance_bounded")


def _solve_generic_schedule_allocation(text: str) -> StateReasoningResult:
    if not _SCHED_ASK_RE.search(text):
        return StateReasoningResult()
    m_tasks = _SCHED_TASKS_RE.search(text)
    if not m_tasks:
        return StateReasoningResult()
    tasks = _as_float(m_tasks.group("n"))
    if tasks is None or tasks <= 0:
        return StateReasoningResult()

    resources = 1.0
    m_res = _SCHED_RESOURCE_RE.search(text)
    if m_res:
        rv = _as_float(m_res.group("n"))
        if rv is not None and rv > 0:
            resources = rv

    per_resource = 1.0
    m_per = _SCHED_PER_RESOURCE_RE.search(text)
    if m_per:
        pv = _as_float(m_per.group("n"))
        if pv is not None and pv > 0:
            per_resource = pv

    capacity_per_round = resources * per_resource
    if capacity_per_round <= 0:
        return _fail_loud("Scheduling capacity per round could not be determined.")

    rounds = int(math.ceil(tasks / capacity_per_round))
    remainder = float(tasks - ((rounds - 1) * capacity_per_round)) if rounds > 1 else float(tasks)
    ans = (
        f"Total workload {_fmt(tasks)} with capacity {_fmt(capacity_per_round)} per round "
        f"(resources={_fmt(resources)}, per-resource={_fmt(per_resource)}) requires {rounds} rounds. "
        f"Final round load is {_fmt(remainder)}."
        f"\nSource: Contextual"
    )
    return StateReasoningResult(handled=True, family="state_transition", answer=ans, reason="generic_schedule_allocation")


def _is_constraint_decision_candidate(text: str) -> bool:
    t = " ".join((text or "").split())
    if not t:
        return False
    if not _DECISION_INTENT_RE.search(t):
        return False
    if not _DECISION_OR_RE.search(t):
        return False
    if not (_OPT_WALK_RE.search(t) and _OPT_DRIVE_RE.search(t)):
        return False
    # High precision trigger: must imply a task requiring an asset physically at destination.
    # Prefer explicit known-asset matcher, then fallback to structural generic action-on-asset matcher.
    if not (_ASSET_TASK_RE.search(t) or _GENERIC_ASSET_ACTION_RE.search(t)):
        return False
    return True


def _clean_asset_phrase(raw: str) -> str:
    toks = [x for x in re.split(r"\s+", (raw or "").strip().lower()) if x]
    cut_words = {"and", "or", "but"}
    keep: list[str] = []
    for tok in toks:
        if tok in cut_words:
            break
        if tok in {"my", "the", "a", "an"}:
            continue
        keep.append(tok)
        if len(keep) >= 4:
            break
    toks = keep
    while toks and toks[-1] in {"to", "for", "at", "in", "on", "from", "with", "wash", "station"}:
        toks.pop()
    if not toks:
        return "asset"
    return " ".join(toks[:4])


def _extract_asset_candidate(text: str) -> Optional[str]:
    t = " ".join((text or "").split())
    if not t:
        return None
    m = _GENERIC_ASSET_ACTION_RE.search(t)
    if m and m.group("asset"):
        v = _clean_asset_phrase(m.group("asset"))
        if v and v != "asset":
            return v
    return None


def _infer_asset_label(text: str) -> str:
    t = " ".join((text or "").split()).lower()
    generic = _extract_asset_candidate(t)
    if generic:
        return _canonical_asset(generic)
    for asset in ("car", "train", "bus", "truck", "van", "tram", "boat", "ship", "motorcycle", "motorbike", "bike", "scooter"):
        if re.search(rf"\b{re.escape(asset)}\b", t):
            return _canonical_asset(asset)
    return "asset"


def _canonical_asset(asset: str) -> str:
    a = (asset or "").strip().lower()
    aliases = {
        "motorbike": "motorcycle",
        "motor cycle": "motorcycle",
        "motor-bike": "motorcycle",
    }
    return aliases.get(a, a)


def _infer_destination_label(text: str) -> str:
    t = " ".join((text or "").split()).lower()
    for dest in ("car wash", "train wash", "wash", "mechanic", "garage", "dealership", "service center", "depot"):
        if dest in t:
            return dest.replace(" ", "_")
    return "destination"


def _is_drive_option_compatible(asset: str) -> bool:
    a = _canonical_asset(asset)
    # Keep this deliberately small and high-precision.
    return a in {"car", "truck", "van", "bus", "motorcycle", "bike", "scooter"}


def _asset_ref(asset: str) -> str:
    a = _canonical_asset(asset)
    if a in {"car", "truck", "van", "bus", "train", "tram", "boat", "ship", "motorcycle", "bike", "scooter"}:
        return a
    if _is_likely_animal(a):
        return a
    return "it"


def _asset_noun(ref: str) -> str:
    r = (ref or "").strip().lower()
    return "it" if (not r or r == "it") else f"the {r}"


def _operate_phrase(ref: str) -> str:
    r = (ref or "").strip().lower()
    return "operate it" if (not r or r == "it") else f"operate the {r}"


def _is_likely_animal(asset: str) -> bool:
    toks = _normalize_text_tokens(asset)
    animals = {
        "dog", "dogs", "cat", "cats", "horse", "horses", "cow", "cows",
        "sheep", "goat", "goats", "pig", "pigs", "puppy", "puppies",
        "kitten", "kittens", "pet", "pets",
    }
    return bool(toks & animals)


def _infer_clarify_options(asset: str) -> list[str]:
    a = _asset_ref(asset)
    if a == "train":
        return ["move the train to the destination", "tow/haul/transport it", "get yourself there first"]
    if a in {"boat", "ship"}:
        return [f"move the {a} to the destination", "tow/haul/transport it", "get yourself there first"]
    if a in {"tram"}:
        return ["move the tram to the destination", "tow/haul/transport it", "get yourself there first"]
    if _is_likely_animal(a):
        return [f"move the {a} to the destination", f"tow/haul/transport the {a}", "get yourself there first"]
    if a == "it":
        return ["move it to the destination", "tow/haul/transport it", "get yourself there first"]
    return [f"move the {a} to the destination", f"tow/haul/transport the {a}", "get yourself there first"]


def _default_clarifier_choice(frame: Dict[str, Any]) -> str:
    asset = str(frame.get("asset") or "asset").strip() or "asset"
    if _is_likely_animal(asset):
        return "yourself_first_walk"
    return "operate"


def _normalize_text_tokens(text: str) -> set[str]:
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not t:
        return set()
    return {x for x in t.split() if x}


def _match_clarify_choice(query: str, options: list[str]) -> Optional[str]:
    q = (query or "").strip().lower()
    if not q:
        return None
    qt = _normalize_text_tokens(q)
    opts_join = " ".join((options or [])).lower()

    # Stage-aware precedence: if options explicitly include propulsion refinement,
    # map user wording directly to operate/tow branches.
    if "under its own power" in opts_join:
        if {"under", "own", "power"} <= qt or "under its own power" in q:
            return "operate"
    if "external transport" in opts_join:
        if {"external", "transport"} <= qt or "external transport" in q:
            return "tow/transport"

    # High-precision direct cues for current option families.
    if {"walk"} & qt:
        return "yourself_first_walk"
    if {"drive", "driving"} & qt:
        return "yourself_first_drive"
    if {"self"} & qt:
        return "yourself_first"
    if {"myself"} & qt:
        return "yourself_first"
    if {"get", "there"} <= qt:
        return "yourself_first"
    if {"operate", "run", "pilot"} & qt:
        return "operate"
    if {"move", "moving"} & qt:
        return "move"
    if {"tow", "transport", "truck", "haul"} & qt:
        return "tow/transport"
    if {"external"} & qt:
        return "tow/transport"
    if ("first" in qt or "1" in qt) and len(options or []) >= 1:
        return str(options[0]).strip()
    if ("second" in qt or "2" in qt) and len(options or []) >= 2:
        return str(options[1]).strip()
    if ("third" in qt or "3" in qt) and len(options or []) >= 3:
        return str(options[2]).strip()

    # Fallback lexical overlap against provided options.
    best = None
    best_score = 0
    for opt in options or []:
        ot = _normalize_text_tokens(opt)
        if not ot:
            continue
        score = len(qt & ot)
        if score > best_score:
            best_score = score
            best = opt
    if best_score >= 1:
        return best
    return None


def _solve_constraint_decision_query(text: str) -> StateReasoningResult:
    if not _is_constraint_decision_candidate(text):
        return StateReasoningResult(family="constraint_decision")
    asset = _infer_asset_label(text)
    destination = _infer_destination_label(text)
    frame: Dict[str, Any] = {
        "kind": "option_feasibility",
        "goal": "asset_at_destination",
        "asset": asset,
        "destination": destination,
        "options": {
            "walk": {"moves_asset": False, "requires": []},
            "drive": {"moves_asset": True, "requires": ["fuel"]},
        },
        "constraints": {"fuel": "unknown"},
    }
    dist_km = _distance_km(text)
    if dist_km is not None:
        frame["distance_km"] = dist_km
    # Compatibility gate: if the user asks walk/drive but "drive" is not a valid
    # action for this asset in this decision family, ask a deterministic clarifier.
    if not _is_drive_option_compatible(asset):
        opts = _infer_clarify_options(asset)
        answer = (
            f"Quick check: did you mean {opts[0]}, {opts[1]}, or {opts[2]}? "
            "Pick one and I'll solve it cleanly."
            "\nSource: Contextual"
        )
        frame["needs_clarification"] = True
        frame["clarify_options"] = opts
        return StateReasoningResult(
            handled=True,
            fail_loud=True,
            family="constraint_decision",
            answer=answer,
            reason="constraint_option_incompatible_requires_clarification",
            frame=frame,
        )

    ref = _asset_ref(asset)
    noun = _asset_noun(ref)
    answer = (
        f"Drive. The task requires the {ref} to be physically at the destination, "
        f"and walking only moves you, not {noun}. "
        "So the hard precondition is satisfied by driving."
        "\nSource: Contextual"
    )
    return StateReasoningResult(
        handled=True,
        fail_loud=False,
        family="constraint_decision",
        answer=answer,
        reason="constraint_precondition_transport",
        frame=frame,
    )


def is_followup_consistency_query(query: str) -> bool:
    t = " ".join((query or "").split())
    if not t:
        return False
    return bool(_FOLLOWUP_QUERY_RE.search(t) or _CONSTRAINT_FUEL_FALSE_RE.search(t) or _CONSTRAINT_FUEL_TRUE_RE.search(t))


def solve_constraint_followup(*, frame: Dict[str, Any], query: str) -> StateReasoningResult:
    t = " ".join((query or "").split())
    if not t or not isinstance(frame, dict):
        return StateReasoningResult(family="constraint_decision")
    if str(frame.get("kind") or "") != "option_feasibility":
        return StateReasoningResult(family="constraint_decision")
    # Fresh top-level decision questions must be re-solved as new decisions,
    # not treated as sticky follow-ups from prior selected action.
    if _is_constraint_decision_candidate(t):
        return StateReasoningResult(family="constraint_decision")
    # If a deterministic clarifier question is active, resolve direct user choice first.
    needs_clarification = bool(frame.get("needs_clarification", False))
    clarify_options = [str(x).strip() for x in (frame.get("clarify_options") or []) if str(x).strip()]
    if needs_clarification and clarify_options:
        # Comparison-intent questions must be explained, not auto-picked by ordinal words.
        if _CHOICE_COMPARISON_RE.search(t):
            opts = clarify_options[:3]
            if len(opts) == 3:
                asset = str(frame.get("asset") or "asset").strip() or "asset"
                ref = _asset_ref(asset)
                ans = (
                    f"Good question. Option 1 means {('the ' + ref) if ref != 'it' else 'the asset'} moves to the destination under its own power. "
                    "Option 2 means external movement (tow/haul/transport). "
                    "If those are effectively the same in your situation, pick either 1 or 2 and I'll continue. "
                    "If you only mean getting yourself there first, pick option 3."
                    "\nSource: Contextual"
                )
                return StateReasoningResult(
                    handled=True,
                    fail_loud=True,
                    family="constraint_decision",
                    answer=ans,
                    reason="clarification_explain_option_difference",
                    frame=dict(frame),
                )
        choice = _match_clarify_choice(t, clarify_options)
        if (not choice) and _DELEGATE_DECISION_RE.search(t):
            choice = _default_clarifier_choice(frame)
        if choice:
            asset = str(frame.get("asset") or "asset").strip() or "asset"
            ref = _asset_ref(asset)
            noun = _asset_noun(ref)
            c = choice.lower()
            # Two-stage clarification for ambiguous "move <asset>" intents on heavy assets.
            if c == "move" and ref in {"train", "tram", "boat", "ship"}:
                ans = (
                    f"Got it. Just to confirm, do you mean moving the {ref} under its own power, "
                    "or moving it with external transport (tow/haul)?"
                    "\nSource: Contextual"
                )
                out_frame = dict(frame)
                out_frame["needs_clarification"] = True
                out_frame["clarify_options"] = ["under its own power", "external transport (tow/haul)"]
                out_frame["clarify_stage"] = "propulsion_mode"
                return StateReasoningResult(
                    handled=True,
                    fail_loud=True,
                    family="constraint_decision",
                    answer=ans,
                    reason="clarification_move_requires_propulsion_mode",
                    frame=out_frame,
                )

            if "walk" in c:
                ans = (
                    f"If you mean getting yourself there first, walk is the practical choice for this short distance. "
                    f"That gets you there, but {noun} stays where it is."
                    "\nSource: Contextual"
                )
                reason = "clarification_choice_walk"
            elif ("yourself_first" in c) or ("get yourself there first" in c):
                d_km = frame.get("distance_km")
                if isinstance(d_km, (int, float)):
                    d_km_f = float(d_km)
                    clause = _yourself_first_distance_clause(d_km_f)
                    ans = (
                        f"If you mean getting yourself there first, at about {_fmt(d_km_f)} km, {clause}. "
                        f"That gets you there, but {noun} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                    reason = "clarification_choice_yourself_first_distance_aware"
                else:
                    ans = (
                        f"If you mean getting yourself there first, walk is usually the practical choice for this short distance. "
                        f"That gets you there, but {noun} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                    reason = "clarification_choice_yourself_first"
            elif "yourself_first_drive" in c:
                ans = (
                    f"If you mean getting yourself there first, driving is possible, but for this short distance walking is usually the practical choice. "
                    f"Either way, {noun} still needs a separate move/tow step."
                    "\nSource: Contextual"
                )
                reason = "clarification_choice_yourself_drive"
            elif ("tow" in c) or ("transport" in c) or ("truck" in c) or ("haul" in c):
                ans = (
                    f"Then transport it. That gets {noun} to the destination."
                    "\nSource: Contextual"
                )
                reason = "clarification_choice_tow_transport"
            else:
                if _is_likely_animal(asset):
                    subject = "it" if (not asset or asset == "asset") else f"the {asset}"
                    ans = (
                        f"Then let {subject} walk there with you. That gets {subject} to the destination."
                        "\nSource: Contextual"
                    )
                    reason = "clarification_choice_operate_animal"
                else:
                    ans = (
                        f"Then move it under its own power. That gets {noun} to the destination."
                        "\nSource: Contextual"
                    )
                    reason = "clarification_choice_operate"

            out_frame = dict(frame)
            out_frame["needs_clarification"] = False
            out_frame["selected_action"] = (
                "yourself_first_drive"
                if ("yourself_first_drive" in c)
                else "yourself_first_walk"
                if ("yourself_first" in c)
                else "yourself_first_walk"
                if ("walk" in c and "yourself" in c)
                else "yourself_first_drive"
                if ("drive" in c and "yourself" in c)
                else "walk"
                if "walk" in c
                else "tow_transport"
                if ("tow" in c or "transport" in c or "truck" in c or "haul" in c or "external" in c)
                else "operate"
            )
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason=reason,
                frame=out_frame,
            )

        # Clarifier is active but user did not choose one of the options.
        if _CONFUSION_RE.search(t):
            opts = clarify_options[:3]
            if len(opts) == 3:
                ans = (
                    f"No worries. I mean this:\n"
                    f"1) {opts[0]}\n"
                    f"2) {opts[1]}\n"
                    f"3) {opts[2]}\n"
                    "Pick one and I'll continue."
                    "\nSource: Contextual"
                )
                return StateReasoningResult(
                    handled=True,
                    fail_loud=True,
                    family="constraint_decision",
                    answer=ans,
                    reason="clarification_user_confused_rephrase",
                    frame=dict(frame),
                )

        # Generic sane-bucket fallback:
        # if user doesn't pick an option and isn't asking to explain options,
        # treat as implicit delegation to avoid infinite clarification loops.
        if not _CLARIFIER_EXPLAIN_RE.search(t):
            choice = _default_clarifier_choice(frame)
            out_frame = dict(frame)
            out_frame["needs_clarification"] = False
            out_frame["selected_action"] = (
                "operate" if choice == "operate" else "yourself_first_walk"
            )
            asset = str(frame.get("asset") or "asset").strip() or "asset"
            ref = _asset_ref(asset)
            noun = _asset_noun(ref)
            if choice == "operate":
                ans = (
                    f"I'll pick the practical default: move it under its own power. "
                    f"That gets {noun} to the destination."
                    "\nSource: Contextual"
                )
                reason = "clarification_implicit_delegate_default_operate"
            else:
                ans = (
                    f"I'll pick the practical default: get yourself there first (walk). "
                    f"That gets you there, but {noun} still needs a separate move/tow step."
                    "\nSource: Contextual"
                )
                reason = "clarification_implicit_delegate_default_yourself_first"
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason=reason,
                frame=out_frame,
            )
        opts = clarify_options[:3]
        repeat_count = int(frame.get("clarifier_repeat_count", 0) or 0)
        if repeat_count >= 1:
            # Guardrail: avoid repeating the same clarifier endlessly.
            choice = _default_clarifier_choice(frame)
            out_frame = dict(frame)
            out_frame["needs_clarification"] = False
            out_frame["clarifier_repeat_count"] = repeat_count + 1
            out_frame["selected_action"] = "operate" if choice == "operate" else "yourself_first_walk"
            asset = str(frame.get("asset") or "asset").strip() or "asset"
            ref = _asset_ref(asset)
            noun = _asset_noun(ref)
            if choice == "operate":
                ans = (
                    f"Got it. I'll choose the practical default: move it under its own power. "
                    f"That gets {noun} to the destination."
                    "\nSource: Contextual"
                )
                reason = "clarification_repeat_default_operate"
            else:
                ans = (
                    f"Got it. I'll choose the practical default: get yourself there first (walk). "
                    f"That gets you there, but {noun} still needs a separate move/tow step."
                    "\nSource: Contextual"
                )
                reason = "clarification_repeat_default_yourself_first"
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason=reason,
                frame=out_frame,
            )
        if len(opts) == 3:
            ans = (
                f"Quick check: did you mean {opts[0]}, {opts[1]}, or {opts[2]}? "
                "Pick one and I'll solve it cleanly."
                "\nSource: Contextual"
            )
            out_frame = dict(frame)
            out_frame["clarifier_repeat_count"] = repeat_count + 1
            return StateReasoningResult(
                handled=True,
                fail_loud=True,
                family="constraint_decision",
                answer=ans,
                reason="clarification_still_required",
                frame=out_frame,
            )

    # Intent-gate for post-choice conversational follow-ups: keep deterministic guidance
    # without forcing the model path for simple "what if/how/should I..." continuations.
    selected_action = str(frame.get("selected_action") or "").strip().lower()
    if selected_action:
        low = t.lower()
        tok = _normalize_text_tokens(low)
        lane_disengaged = bool(frame.get("decision_lane_disengaged", False))
        # Idempotent delegate handling for post-choice turns (e.g., regenerate "you choose"/"idk").
        if _DELEGATE_DECISION_RE.search(t):
            asset = str(frame.get("asset") or "asset").strip() or "asset"
            ref = _asset_ref(asset)
            noun = _asset_noun(ref)
            frame_dist_km = frame.get("distance_km")
            frame_dist_km = float(frame_dist_km) if isinstance(frame_dist_km, (int, float)) else None
            if selected_action in {"yourself_first_walk", "yourself_first_drive", "walk"}:
                if frame_dist_km is not None:
                    clause = _yourself_first_distance_clause(frame_dist_km)
                    ans = (
                        f"Still the same answer: get yourself there first. At about {_fmt(frame_dist_km)} km, {clause}. "
                        f"{noun.capitalize()} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                else:
                    ans = (
                        f"Still the same answer: get yourself there first. For this short distance, walking is usually the practical choice. "
                        f"{noun.capitalize()} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                reason = "post_choice_delegate_replay_yourself_first"
            elif selected_action == "tow_transport":
                ans = (
                    f"Still the same answer: transport it. That gets {noun} to the destination."
                    "\nSource: Contextual"
                )
                reason = "post_choice_delegate_replay_tow_transport"
            else:
                ans = (
                    f"Still the same answer: move it under its own power. That gets {noun} to the destination."
                    "\nSource: Contextual"
                )
                reason = "post_choice_delegate_replay_operate"
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason=reason,
                frame=dict(frame),
            )

        if lane_disengaged and _is_correction_intent(t) and not _is_explicit_reengage(t):
            out_frame = dict(frame)
            out_frame["decision_lane_disengaged"] = True
            return StateReasoningResult(
                handled=False,
                fail_loud=False,
                family="constraint_decision",
                answer="",
                reason="decision_lane_disengaged_correction_passthrough",
                frame=out_frame,
            )
        if lane_disengaged and not _is_explicit_reengage(t):
            out_frame = dict(frame)
            out_frame["decision_lane_disengaged"] = True
            return StateReasoningResult(
                handled=False,
                fail_loud=False,
                family="constraint_decision",
                answer="",
                reason="decision_lane_disengaged",
                frame=out_frame,
            )
        if lane_disengaged and _is_explicit_reengage(t):
            frame = dict(frame)
            frame["decision_lane_disengaged"] = False
        asset = str(frame.get("asset") or "asset").strip() or "asset"
        ref = _asset_ref(asset)
        noun = _asset_noun(ref)
        frame_dist_km = frame.get("distance_km")
        frame_dist_km = float(frame_dist_km) if isinstance(frame_dist_km, (int, float)) else None

        # Deterministic distance refinement for "get yourself there first" branch.
        if selected_action in {"yourself_first_walk", "yourself_first_drive", "walk"} and not (({"self"} & tok) or ({"myself"} & tok)):
            km = _distance_km(low)
            if km is not None:
                clause = _yourself_first_distance_clause(km)
                ans = (
                    f"At about {_fmt(km)} km, {clause}. "
                    f"Either way, {noun} still needs a separate move/tow step."
                    "\nSource: Contextual"
                )
                return StateReasoningResult(
                    handled=True,
                    fail_loud=False,
                    family="constraint_decision",
                    answer=ans,
                    reason="post_choice_distance_refine_yourself_first",
                    frame=dict(frame),
                )

        # Out-of-scope conversational follow-up: disengage deterministic lane and defer to model.
        if not (
            bool(_DISTANCE_RE.search(low))
            or bool(_FOLLOWUP_QUERY_RE.search(low))
            or bool(_CONSTRAINT_FUEL_FALSE_RE.search(low) or _CONSTRAINT_FUEL_TRUE_RE.search(low))
            or bool(re.search(r"\bshould\s+i\b.*\bwalk\b.*\bdrive\b", low))
            or bool(({"self"} & tok) or ({"myself"} & tok))
        ):
            out_frame = dict(frame)
            out_frame["decision_lane_disengaged"] = True
            return StateReasoningResult(
                handled=False,
                fail_loud=False,
                family="constraint_decision",
                answer="",
                reason="decision_lane_disengage_out_of_scope",
                frame=out_frame,
            )

        if ({"self"} & tok or {"myself"} & tok):
            if selected_action in {"yourself_first_walk", "yourself_first_drive", "walk"}:
                if frame_dist_km is not None and frame_dist_km >= 1.0:
                    ans = (
                        f"Still the same answer: get yourself there first. At about {_fmt(frame_dist_km)} km, driving is usually the practical low-effort choice. "
                        f"{noun.capitalize()} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                else:
                    ans = (
                        f"Still the same answer: get yourself there first. For this short distance, walking is usually the practical choice. "
                        f"{noun.capitalize()} still needs a separate move/tow step."
                        "\nSource: Contextual"
                    )
                return StateReasoningResult(
                    handled=True,
                    fail_loud=False,
                    family="constraint_decision",
                    answer=ans,
                    reason="post_choice_self_myself_normalized",
                    frame=dict(frame),
                )
        if re.search(r"\bshould\s+i\b.*\bwalk\b.*\bdrive\b", low):
            asset = str(frame.get("asset") or "asset").strip() or "asset"
            ref = _asset_ref(asset)
            action_label = {
                "walk": "walk there first",
                "yourself_first_walk": "get yourself there first (walk)",
                "yourself_first_drive": "get yourself there first (drive)",
                "operate": _operate_phrase(ref),
                "tow_transport": "transport it",
            }.get(selected_action, selected_action)
            ans = (
                f"You've already selected an action: {action_label}. "
                "Use that path unless your constraints changed."
                "\nSource: Contextual"
            )
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason="post_choice_restate_action",
                frame=dict(frame),
            )
        if ("what if" in low) and re.search(r"\b(rain|raining|storm|weather|wet|snow|ice)\b", low):
            action_label = {
                "walk": "walk-first",
                "yourself_first_walk": "get-yourself-there-first (walk)",
                "yourself_first_drive": "get-yourself-there-first (drive)",
                "operate": "operate",
                "tow_transport": "tow/transport",
            }.get(selected_action, selected_action)
            ans = (
                f"Weather can change safety/comfort, but it doesn't change the selected action path ({action_label}) by itself. "
                "If constraints changed materially, say what changed and I'll re-evaluate."
                "\nSource: Contextual"
            )
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason="post_choice_weather_gate",
                frame=dict(frame),
            )
        if low.startswith("how do i"):
            ans = (
                "That's a practical how-to question rather than a walk/drive decision step. "
                "If you want, ask for a practical checklist with assumptions and constraints."
                "\nSource: Contextual"
            )
            return StateReasoningResult(
                handled=True,
                fail_loud=False,
                family="constraint_decision",
                answer=ans,
                reason="post_choice_howto_gate",
                frame=dict(frame),
            )

    if not is_followup_consistency_query(t):
        return StateReasoningResult(family="constraint_decision")
    asset = str(frame.get("asset") or "asset").strip() or "asset"
    ref = _asset_ref(asset)
    noun = _asset_noun(ref)

    constraints = dict(frame.get("constraints") or {})
    if _CONSTRAINT_FUEL_FALSE_RE.search(t):
        constraints["fuel"] = False
    elif _CONSTRAINT_FUEL_TRUE_RE.search(t):
        constraints["fuel"] = True

    options = dict(frame.get("options") or {})
    walk = dict(options.get("walk") or {})
    drive = dict(options.get("drive") or {})

    walk_feasible = bool(walk.get("moves_asset", False))
    drive_requires = [str(x).strip().lower() for x in (drive.get("requires") or []) if str(x).strip()]
    drive_feasible = bool(drive.get("moves_asset", False))
    for req in drive_requires:
        v = constraints.get(req, "unknown")
        if v is False:
            drive_feasible = False
        elif v == "unknown":
            drive_feasible = False

    if drive_feasible and not walk_feasible:
        ans = (
            f"Drive. The hard precondition is still that the {ref} must be at the destination, "
            f"and walking does not move {noun}."
            "\nSource: Contextual"
        )
        reason = "followup_drive_still_feasible"
        fail_loud = False
    elif (not drive_feasible) and walk_feasible:
        ans = (
            "Walk is feasible for reaching the destination yourself, but it still does not satisfy "
            f"the original hard precondition of moving {noun} there."
            "\nSource: Contextual"
        )
        reason = "followup_walk_only_person"
        fail_loud = False
    elif (not drive_feasible) and (not walk_feasible):
        ans = (
            "Neither option satisfies the hard precondition. "
            f"Driving is infeasible under current constraints, and walking does not move {noun}. "
            "You need a new feasible action (for example, add fuel or towing) before proceeding."
            "\nSource: Contextual"
        )
        reason = "followup_none_feasible"
        fail_loud = True
    else:
        ans = (
            "Both options appear feasible under current constraints, but only options that move the asset "
            "satisfy the original hard precondition."
            "\nSource: Contextual"
        )
        reason = "followup_multiple_feasible"
        fail_loud = False

    out_frame = dict(frame)
    out_frame["constraints"] = constraints
    return StateReasoningResult(
        handled=True,
        fail_loud=fail_loud,
        family="constraint_decision",
        answer=ans,
        reason=reason,
        frame=out_frame,
    )
