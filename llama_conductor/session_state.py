# session_state.py
"""Session state management for MoA Router."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .interaction_profile import InteractionProfile, new_profile


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
    pending_sensitive_confirm_query: str = ""

    # One Vodka per session (reduces noise + preserves stateful debug counters)
    vodka: Optional[Any] = None  # VodkaFilter

    # Trust mode: pending recommendations for A/B/C response
    pending_trust_query: str = ""
    pending_trust_recommendations: List[Dict[str, str]] = field(default_factory=list)
    
    # Auto-execute query after >>attach all from trust
    auto_query_after_attach: str = ""
    auto_detach_after_response: bool = False
    
    # Session interaction profile (ephemeral, in-memory only)
    profile_enabled: bool = True
    interaction_profile: InteractionProfile = field(default_factory=new_profile)
    profile_turn_counter: int = 0
    profile_effective_strength: float = 0.0
    profile_output_compliance: float = 0.0
    profile_blocked_nicknames: Set[str] = field(default_factory=set)

    # Serious-mode anti-loop tracking.
    serious_ack_reframe_streak: int = 0
    serious_last_body_signature: str = ""
    serious_repeat_streak: int = 0


# Global session storage
_SESSIONS: Dict[str, SessionState] = {}


def get_state(session_id: str) -> SessionState:
    """Get or create session state for given session ID."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
    return _SESSIONS[session_id]
