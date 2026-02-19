# session_state.py
"""Session state management for MoA Router."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SessionState:
    attached_kbs: Set[str] = field(default_factory=set)
    fun_sticky: bool = False
    fun_rewrite_sticky: bool = False
    raw_sticky: bool = False

    rag_last_query: str = ""
    rag_last_hits: int = 0

    vault_last_query: str = ""
    vault_last_hits: int = 0
    locked_summ_file: str = ""
    locked_summ_kb: str = ""
    locked_summ_path: str = ""
    locked_summ_rel_path: str = ""
    locked_last_fact_lines: int = 0
    pending_lock_candidate: str = ""

    # One Vodka per session (reduces noise + preserves stateful debug counters)
    vodka: Optional[any] = None  # VodkaFilter

    # Trust mode: pending recommendations for A/B/C response
    pending_trust_query: str = ""
    pending_trust_recommendations: List[Dict[str, str]] = field(default_factory=list)
    
    # Auto-execute query after >>attach all from trust
    auto_query_after_attach: str = ""
    auto_detach_after_response: bool = False
    
    # Cliniko sidecar: staged clinical note pipeline
    cliniko_compacted: Optional[str] = None
    cliniko_compaction_stats: Dict[str, int] = field(default_factory=dict)
    cliniko_last_draft: Optional[str] = None
    cliniko_last_scaffold: str = ""
    cliniko_last_raw: str = ""
    cliniko_last_region: str = ""


# Global session storage
_SESSIONS: Dict[str, SessionState] = {}


def get_state(session_id: str) -> SessionState:
    """Get or create session state for given session ID."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
    return _SESSIONS[session_id]
