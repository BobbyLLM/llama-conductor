"""State-solver mode/policy guards.

Centralizes policy decisions for when deterministic state-solver auto-routing
is allowed for a turn.
"""

from __future__ import annotations

from typing import Any, Callable


def state_solver_enabled(cfg_get: Callable[[str, Any], Any]) -> bool:
    return bool(cfg_get("state_solver.enabled", True)) and bool(
        cfg_get("state_solver.auto_route", True)
    )


def selector_blocked_for_state_solver(
    *, selector: str, cfg_get: Callable[[str, Any], Any]
) -> bool:
    selector_l = str(selector or "").strip().lower()
    if not selector_l:
        return False
    if selector_l == "ocr" and bool(cfg_get("state_solver.ocr_firewall.enabled", True)):
        return True
    return False


def should_skip_state_solver_early(
    *,
    fun_mode: str,
    selector: str,
    lock_active_now: bool,
    classify_query_family: Any,
    solve_state_transition_query: Any,
    cfg_get: Callable[[str, Any], Any],
) -> bool:
    if fun_mode != "":
        return True
    if classify_query_family is None or solve_state_transition_query is None:
        return True
    if not state_solver_enabled(cfg_get):
        return True
    if lock_active_now:
        return True
    if selector_blocked_for_state_solver(selector=selector, cfg_get=cfg_get):
        return True
    return False
