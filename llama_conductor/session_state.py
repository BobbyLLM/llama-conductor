# session_state.py
"""Session state management for MoA Router."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .interaction_profile import InteractionProfile, new_profile, profile_set
from .config import cfg_get

KAIOKEN_CLOSED_THREAD_TTL_TURNS = 10


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
    pending_vodka_comment_ctx_id: str = ""
    pending_vodka_comment_text: str = ""

    # One Vodka per session (reduces noise + preserves stateful debug counters)
    vodka: Optional[Any] = None  # VodkaFilter

    # Trust mode: pending recommendations for A/B/C response
    pending_trust_query: str = ""
    pending_trust_recommendations: List[Dict[str, str]] = field(default_factory=list)
    pending_trust_judge_command: str = ""
    
    # Auto-execute query after >>attach all from trust
    auto_query_after_attach: str = ""
    auto_detach_after_response: bool = False
    
    # Cliniko sidecar: staged clinical note pipeline

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

    # Runtime config preset override (empty means use router_config default).
    vodka_preset_override: str = ""

    # Last deterministic reasoning frame (for follow-up consistency checks).
    deterministic_last_family: str = ""
    deterministic_last_reason: str = ""
    deterministic_last_answer: str = ""
    deterministic_last_frame: Dict[str, Any] = field(default_factory=dict)
    deterministic_last_query_norm: str = ""

    # Last conversational turn snapshots (for correction-binding when clients send short history).
    last_user_text: str = ""
    last_assistant_text: str = ""

    # Scratchpad grounding controls (session-ephemeral).
    # Modes: "strict" (facts-only), "assisted" (facts-first synthesis allowed).
    scratchpad_mode: str = "strict"
    # Optional 1-based indices scoped to current scratchpad list order.
    # Empty means unlocked (all eligible records).
    scratchpad_locked_indices: Set[int] = field(default_factory=set)

    # KAIOKEN telemetry state (phase 0 sensor-only).
    kaioken_turn_counter: int = 0
    kaioken_enabled: bool = field(default_factory=lambda: bool(cfg_get("kaioken.enabled", True)))
    kaioken_mode: str = field(default_factory=lambda: str(cfg_get("kaioken.mode", "log_only") or "log_only"))
    # KAIOKEN topic lifecycle (closed/open/active thread hints).
    kaioken_active_topic: str = ""
    kaioken_open_topics: List[str] = field(default_factory=list)
    kaioken_closed_topics: Set[str] = field(default_factory=set)
    kaioken_threads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    kaioken_active_thread_id: str = ""
    kaioken_thread_seq: int = 0
    kaioken_last_topic_switch_turn: int = 0
    kaioken_last_resolution_turn: int = -9999
    # Topics disclosed in personal/distress turns (structural memory, not lexicon).
    kaioken_distress_topics: Set[str] = field(default_factory=set)
    # Guard-local recent assistant bodies for short-window repeat suppression.
    kaioken_recent_assistant_bodies: List[str] = field(default_factory=list)
    # Recent raw user turns for structural narrative-ownership checks.
    kaioken_recent_user_turns: List[str] = field(default_factory=list)

    # Per-turn provenance overrides (set by retrieval lanes, consumed in finalize).
    turn_footer_source_override: str = ""
    turn_footer_confidence_override: str = ""
    turn_source_url_override: str = ""
    turn_retrieval_track: str = ""
    turn_cheatsheet_hit: bool = False
    turn_playful_override: bool = False
    turn_local_knowledge_line: str = ""
    turn_cheatsheets_warning_line: str = ""
    turn_cheatsheets_warning_key: str = ""
    cheatsheets_warning_last_shown_key: str = ""
    turn_quote_source_durability_guard: bool = False
    last_retrieval_miss_intent: str = ""
    last_retrieval_miss_query_signature: str = ""
    last_retrieval_miss_query_text: str = ""
    # Retrieval follow-up evidence context (generic, one-shot by default).
    evidence_context_active: bool = False
    evidence_context_intent_class: str = ""
    evidence_context_source_lane: str = ""
    evidence_context_query_topic: str = ""
    evidence_context_outcome: str = ""  # success|refusal
    evidence_context_ttl_turns: int = 0
    evidence_context_created_turn_id: int = 0
    # One-shot router-side signal: run narrow wiki fallback before model dispatch.
    evidence_context_wiki_fallback_active: bool = False


# Global session storage
_SESSIONS: Dict[str, SessionState] = {}


def _apply_profile_boot_default(state: SessionState) -> None:
    """Apply optional startup profile preset from config (session bootstrap only)."""
    raw = str(cfg_get("profile.default", "") or "").strip().lower()
    if not raw:
        return

    # Canonical config values: direct|casual|turbo (turbo == feral runtime preset).
    if raw == "turbo":
        profile_set(state.interaction_profile, "correction_style", "direct")
        profile_set(state.interaction_profile, "verbosity", "compact")
        profile_set(state.interaction_profile, "snark_tolerance", "high")
        profile_set(state.interaction_profile, "sarcasm_level", "high")
        profile_set(state.interaction_profile, "profanity_ok", "true")
        return
    if raw == "casual":
        profile_set(state.interaction_profile, "correction_style", "direct")
        profile_set(state.interaction_profile, "verbosity", "compact")
        profile_set(state.interaction_profile, "snark_tolerance", "high")
        profile_set(state.interaction_profile, "sarcasm_level", "medium")
        profile_set(state.interaction_profile, "profanity_ok", "false")
        return
    if raw == "direct":
        profile_set(state.interaction_profile, "correction_style", "direct")
        profile_set(state.interaction_profile, "verbosity", "compact")
        profile_set(state.interaction_profile, "snark_tolerance", "medium")
        profile_set(state.interaction_profile, "sarcasm_level", "low")
        profile_set(state.interaction_profile, "profanity_ok", "false")
        return
    # Invalid value: safe fallback to existing baseline behavior (no crash/no mutation).
    return


def get_state(session_id: str) -> SessionState:
    """Get or create session state for given session ID."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState()
        _apply_profile_boot_default(_SESSIONS[session_id])
        # Fresh process/session bootstrap: clear any persisted scratchpad
        # for this session id unless explicitly disabled in config.
        if bool(cfg_get("scratchpad.clear_on_session_init", True)):
            try:
                from .scratchpad_sidecar import clear_scratchpad  # lazy import to avoid cycle
                clear_scratchpad(session_id)
            except Exception:
                pass
    return _SESSIONS[session_id]

