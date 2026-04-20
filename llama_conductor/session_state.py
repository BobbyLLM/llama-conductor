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
    serious_sticky: bool = False

    # Default mode: empty string = serious (implicit legacy default).
    # Set by _apply_profile_boot_default() from config key mode.default.
    # Legal values: "", "raw", "fun", "fun_rewrite".
    # Survives >>flush; stickies are ephemeral overrides on top of this.
    default_mode: str = ""

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
    pending_scratch_online_query: str = ""
    pending_vodka_comment_ctx_id: str = ""
    pending_vodka_comment_text: str = ""
    vodka_trim_log: List[Dict[str, Any]] = field(default_factory=list)
    last_web_results: List[Dict[str, Any]] = field(default_factory=list)
    last_web_results_load_id: str = ""

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
    # Distress turn sequencing: first distress turn should stabilize before probing.
    distress_turn_count: int = 0

    # B2 Stage 1 telemetry — emit only, no behavior changes.
    serious_guard_evaluated: bool = False
    serious_guard_trigger_condition: Optional[str] = None
    serious_guard_failure_type: Optional[str] = None
    serious_guard_offending_clause: Optional[str] = None
    serious_guard_action_taken: str = "NONE"
    serious_hostile_overreach_evaluated: bool = False
    serious_hostile_overreach_result: Optional[bool] = None
    serious_canned_wellness_trigger: Optional[str] = None
    serious_snark_leak_detected: bool = False

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
    # Recent idiom selections for turn-based cooldown in cheatsheet matching.
    # Entries: {"term": <normalized_term>, "turn": <int turn index>}
    recent_idiom_terms: List[Dict[str, Any]] = field(default_factory=list)
    # Recent assistant opener strings (normalized first sentence) for
    # deterministic anti-repeat suppression in banter-style replies.
    recent_assistant_openers: List[Dict[str, Any]] = field(default_factory=list)
    # FUN body repeat guard (staged): rolling body fingerprints + fallback history.
    recent_fun_body_fingerprints: List[str] = field(default_factory=list)
    recent_fun_fallback_bodies: List[str] = field(default_factory=list)
    turn_feral_register_detected: bool = False
    turn_distress_hint_score: int = 0
    turn_empathy_override_applied: bool = False
    # FUN single-pass telemetry (turn-local; emitted via kaioken outcome).
    fun_single_pass_enabled: bool = False
    fun_model_calls_count: int = 0
    fun_body_repeat_detected: bool = False
    fun_body_swap_applied: bool = False
    fun_quote_register_filter_applied: bool = False
    fun_quote_pool_size_after_filter: int = 0
    fr_semantic_drift_detected: bool = False
    fr_semantic_drift_reason: str = "none"
    fr_semantic_drift_entities_missing: List[str] = field(default_factory=list)
    turn_local_knowledge_line: str = ""
    turn_cheatsheets_warning_line: str = ""
    turn_cheatsheets_warning_key: str = ""
    cheatsheets_warning_last_shown_key: str = ""
    # Previous-turn KAIOKEN macro label for transition-based gate hygiene.
    last_turn_kaioken_macro: str = ""
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
    # Post-editorial follow-up lock (short TTL) to prevent meta follow-ups
    # from being misrouted into broad web retrieval.
    post_editorial_lock_active: bool = False
    post_editorial_lock_ttl: int = 0
    post_editorial_object_context: str = ""
    # One-turn handoff: set by retrieval gate on lock follow-up turns and
    # consumed by router during generation context assembly.
    post_editorial_context_inject_pending: bool = False


# Global session storage
_SESSIONS: Dict[str, SessionState] = {}

# Legal values for default_mode. Empty string = serious (implicit legacy default).
_VALID_DEFAULT_MODES = {"", "raw", "fun", "fun_rewrite"}


def _validate_default_mode(mode: str) -> str:
    """Validate and canonicalise a default_mode value.

    Returns the canonical string to store on SessionState.default_mode.
    Raises ValueError on illegal values (fail-loud).
    """
    m = str(mode or "").strip().lower()
    if m == "serious":
        return ""  # canonical: empty = serious (implicit)
    if m in _VALID_DEFAULT_MODES:
        return m
    raise ValueError(
        f"Invalid default_mode '{mode}'. "
        f"Valid values: serious, {', '.join(sorted(_VALID_DEFAULT_MODES - {''}))}"
    )


def _apply_profile_boot_default(state: SessionState) -> None:
    """Apply optional startup profile preset and default mode from config (session bootstrap only)."""
    # --- Profile preset (existing logic, unchanged) ---
    raw = str(cfg_get("profile.default", "") or "").strip().lower()
    if raw:
        # Canonical config values: direct|casual|turbo (turbo == feral runtime preset).
        if raw == "turbo":
            profile_set(state.interaction_profile, "correction_style", "direct")
            profile_set(state.interaction_profile, "verbosity", "compact")
            profile_set(state.interaction_profile, "snark_tolerance", "high")
            profile_set(state.interaction_profile, "sarcasm_level", "high")
            profile_set(state.interaction_profile, "profanity_ok", "true")
        elif raw == "casual":
            profile_set(state.interaction_profile, "correction_style", "direct")
            profile_set(state.interaction_profile, "verbosity", "compact")
            profile_set(state.interaction_profile, "snark_tolerance", "high")
            profile_set(state.interaction_profile, "sarcasm_level", "medium")
            profile_set(state.interaction_profile, "profanity_ok", "false")
        elif raw == "direct":
            profile_set(state.interaction_profile, "correction_style", "direct")
            profile_set(state.interaction_profile, "verbosity", "compact")
            profile_set(state.interaction_profile, "snark_tolerance", "medium")
            profile_set(state.interaction_profile, "sarcasm_level", "low")
            profile_set(state.interaction_profile, "profanity_ok", "false")
        # Invalid value: safe fallback to existing baseline behavior (no crash/no mutation).

    # --- Default mode (new) ---
    dm_raw = str(cfg_get("mode.default", "") or "").strip().lower()
    if dm_raw:
        try:
            state.default_mode = _validate_default_mode(dm_raw)
        except ValueError:
            # Fail-loud: print warning but do not crash boot.
            print(f"[router] WARNING: invalid mode.default '{dm_raw}' in config; ignoring.")


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

