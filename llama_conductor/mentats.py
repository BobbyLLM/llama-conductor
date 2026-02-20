"""Mentats deep-reasoning pipeline (Vault-grounded).

Runs multi-pass draft/critique/final synthesis with strict
source constraints and explicit `Sources: Vault` contract.
"""

from __future__ import annotations
from typing import List, Dict, Any, Callable
import os
import time

# Where to write Mentats debug logs
MENTATS_DEBUG_LOG = os.path.abspath("mentats_debug.log")


def _log_mentats_debug(step_name: str, query: str, text: str, session_id: str = "") -> None:
    """Append a short debug record for Mentats to mentats_debug.log."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(MENTATS_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"==== {ts} | session={session_id or 'N/A'} | step={step_name} ====\n")
            f.write(f"QUERY: {query}\n\n")
            preview = text if len(text) <= 3000 else text[:3000] + "\n...[truncated]...\n"
            f.write(preview)
            f.write("\n\n")
    except Exception:
        # fail-open: debug logging must never break the pipeline
        pass


# ---------------------------------------------------------------------------
# Prompts (v1.0.2: NUCLEAR HARDENING)
# ---------------------------------------------------------------------------

MENTATS_STEP1_PROMPT = """You are MENTATS STEP 1: DRAFT WRITER.

YOUR ONLY JOB: Answer the question using ONLY the provided FACTS_BLOCK.

CARDINAL SINS (these are FIRING offenses):
1. Mentioning ANY entity (product, company, technology, person, place) NOT explicitly in FACTS_BLOCK
2. Using numbers that don't EXACTLY match FACTS_BLOCK (no rounding, no ~, no "approximately")
3. Making comparisons between things when FACTS_BLOCK doesn't explicitly compare them
4. Inferring, guessing, or filling gaps with "reasonable assumptions"
5. Using your training data knowledge instead of FACTS_BLOCK

IF FACTS_BLOCK DOES NOT CONTAIN THE ANSWER:
You MUST use this exact template:

DRAFT_ANSWER:
The provided facts do not contain sufficient information to answer this question.

FACTS_USED:
NONE

CONSTRAINTS_USED:
NONE

GAPS:
- The facts do not mention [specific thing the question asks about]

DO NOT TRY TO ANSWER ANYWAY. DO NOT USE MODEL KNOWLEDGE.

INPUT FORMAT:

QUERY:
[the user's question]

FACTS_BLOCK:
[zero or more fact lines, or exactly: NONE]

CONSTRAINTS_BLOCK:
[zero or more constraint lines, or exactly: NONE]

OUTPUT FORMAT:

DRAFT_ANSWER:
[Max 2 short paragraphs, ≤ 120 words. Direct answer ONLY using FACTS_BLOCK.]
[OR use REFUSAL TEMPLATE above if facts insufficient]

FACTS_USED:
- [Copy EXACTLY, word-for-word, the specific facts from FACTS_BLOCK you relied on]
[If FACTS_BLOCK was NONE or insufficient, write exactly: NONE]

CONSTRAINTS_USED:
- [Copy EXACTLY, word-for-word, the constraints from CONSTRAINTS_BLOCK you followed]
[If CONSTRAINTS_BLOCK was NONE, write exactly: NONE]

GAPS:
- [What crucial information is missing from FACTS_BLOCK]
[If nothing missing, write exactly: NONE]

CRITICAL EXAMPLES OF VIOLATIONS:

BAD (uses entity not in FACTS_BLOCK):
QUERY: How did Amiga compare to Atari ST?
FACTS_BLOCK: [contains Amiga info only]
DRAFT_ANSWER: "Amiga outperformed the Atari ST in graphics..."
← VIOLATION: "Atari ST" not in FACTS_BLOCK

GOOD (refuses):
DRAFT_ANSWER: The provided facts do not contain information about Atari ST.
FACTS_USED: NONE

BAD (rounds number):
FACTS_BLOCK: "production cost of ~$135"
DRAFT_ANSWER: "production cost of $150"
← VIOLATION: Changed $135 to $150

GOOD (exact match):
DRAFT_ANSWER: "production cost of ~$135"

BAD (invents comparison):
FACTS_BLOCK: [C64 specs only]
QUERY: Did C64 succeed?
DRAFT_ANSWER: "C64 succeeded due to its advantages over competitors"
← VIOLATION: No competitors mentioned in FACTS_BLOCK

GOOD (stays in bounds):
DRAFT_ANSWER: "C64 had [specs from FACTS_BLOCK]. Success factors: [only from FACTS_BLOCK]."

REMEMBER: When in doubt, REFUSE. Better to say "I don't know" than hallucinate.
"""

MENTATS_STEP2_PROMPT = """You are MENTATS STEP 2: CRITIC.

YOUR JOB: Hunt for violations in Step 1's draft.

CHECK FOR THESE VIOLATIONS (mark CHECK_OK: NO if ANY found):

1. ENTITY HALLUCINATION
   - Did DRAFT_ANSWER mention ANY product, company, technology, person, or place NOT in FACTS_USED?
   - Examples: "Atari ST", "Starglider", "IBM PC", specific game titles, specific software names
   - If ANY entity appears in DRAFT_ANSWER but NOT in FACTS_USED → CHECK_OK: NO

2. NUMBER MISMATCH
   - Did DRAFT_ANSWER change ANY number from FACTS_USED?
   - "$135" → "$150" is a VIOLATION
   - "~$135" → "$135" is OK (removing ~ is fine)
   - "1982" → "early 1980s" is a VIOLATION

3. INVENTED COMPARISONS
   - Did DRAFT_ANSWER compare things when FACTS_USED doesn't explicitly compare them?
   - "outperformed competitors" when FACTS_USED only lists specs → VIOLATION

4. INFERENCE/GUESSING
   - Did DRAFT_ANSWER draw conclusions not explicitly in FACTS_USED?
   - "led to success" when FACTS_USED only lists features → VIOLATION

5. MODEL KNOWLEDGE
   - Did DRAFT_ANSWER add historical context, background, or details not in FACTS_USED?
   - Any sentence that doesn't directly tie to a specific FACTS_USED item → suspect

INPUT (from Step 1):

DRAFT_ANSWER:
...

FACTS_USED:
...

CONSTRAINTS_USED:
...

GAPS:
...

OUTPUT:

CHECK_OK: [YES or NO]

ISSUES:
- [Each violation found, be specific]
[If none, write: NONE]

VIOLATING_SNIPPET:
[Quote the EXACT text from DRAFT_ANSWER that violates]
[If CHECK_OK: YES, write: NONE]

RULES:
- BE AGGRESSIVE. If in doubt about whether something is in FACTS_USED, mark CHECK_OK: NO
- Look for sneaky violations (paraphrasing entities, vague claims that seem "reasonable")
- If DRAFT_ANSWER used REFUSAL TEMPLATE and FACTS_USED is NONE → that's CORRECT, mark CHECK_OK: YES
- Do NOT mark style issues as violations (only factual overreach matters)
"""

MENTATS_STEP3_PROMPT = """You are MENTATS STEP 3: FINAL ANSWER WRITER.

YOUR JOB: 
- If Step 2 marked CHECK_OK: YES → Clean up Step 1's draft (minor fixes only)
- If Step 2 marked CHECK_OK: NO → REMOVE the violating parts, do NOT replace them

ABSOLUTE RULES:
1. Do NOT add new facts not in Step 1's FACTS_USED
2. Do NOT try to "fix" refusals by using model knowledge
3. Do NOT change numbers
4. Do NOT add entity names not in Step 1's FACTS_USED
5. MUST end with "Sources: Vault" as the last line

IF STEP 1 REFUSED (FACTS_USED: NONE):
Preserve the refusal in FINAL_ANSWER. Do not try to answer anyway.

INPUT:

=== STEP1_BEGIN ===
[Raw Step 1 output]
=== STEP1_END ===

=== STEP2_BEGIN ===
[Raw Step 2 output]
=== STEP2_END ===

OUTPUT:

FINAL_ANSWER:
[Max 3 short paragraphs, ≤ 200 words]
[If Step 1 refused, keep the refusal]
[Remove violations if Step 2 found any, but do NOT add new content]
[Last line MUST be: Sources: Vault]

FACTS_USED:
[Copy Step 1's FACTS_USED exactly, word-for-word]

CONSTRAINTS_USED:
[Copy Step 1's CONSTRAINTS_USED exactly, word-for-word]

NOTES:
[One sentence about how you handled Step 2's feedback]

EXAMPLE (Refusal Preserved):

Step 1 FACTS_USED: NONE
Step 1 DRAFT_ANSWER: "The provided facts do not contain information about Atari ST."
Step 2 CHECK_OK: YES

FINAL_ANSWER:
The provided facts do not contain information about Atari ST.

Sources: Vault

EXAMPLE (Violation Removed):

Step 1 DRAFT_ANSWER: "The Amiga outperformed the Atari ST with better graphics."
Step 1 FACTS_USED: [Amiga graphics specs only]
Step 2 CHECK_OK: NO (mentions Atari ST not in FACTS_USED)

FINAL_ANSWER:
The Amiga featured advanced graphics capabilities. [removed comparison to Atari ST]

Sources: Vault

REMEMBER: Removing violations means DELETE, not REPLACE. Leave gaps visible.
"""


def run_mentats(
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    *,
    vodka: Any,
    call_model: Callable[..., str],
    build_rag_block: Callable[[str, str], str],
    build_constraints_block: Callable[[str], str],
    facts_collection: str = "vault",
    thinker_role: str = "thinker",
    critic_role: str = "critic",
    max_tokens_step1: int = 512,
    max_tokens_step2: int = 256,
    max_tokens_step3: int = 512,
) -> str:
    """Run the full Mentats pipeline for a single question.

    NOTE (v1.0.1 invariants):
    - 'history' is ignored and MUST be empty from the router.
    - 'vodka' is ignored and MUST NOT inject anything.
    Mentats reasons only from Vault-derived FACTS_BLOCK.
    """

    query = (user_text or "").strip()

    # Defensive isolation (router should already enforce this)
    if history:
        _log_mentats_debug("VIOLATION_HISTORY_NONEMPTY", query, "History was non-empty; ignoring.", session_id=session_id)

    # Do NOT call vodka.inlet/outlet here (Mentats must not use Vodka)

    if not query:
        return "PROTOCOL_ERROR: MISSING_QUERY"

    # Build FACTS and CONSTRAINTS blocks
    try:
        facts_block_raw = build_rag_block(query, collection=facts_collection)
    except Exception:
        facts_block_raw = ""

    try:
        constraints_block_raw = build_constraints_block(query)
    except Exception:
        constraints_block_raw = ""

    facts_block = (facts_block_raw or "").strip() or "NONE"
    constraints_block = (constraints_block_raw or "").strip() or "NONE"

    # If no facts, signal protocol error (router should refuse before calling Mentats)
    if facts_block == "NONE":
        _log_mentats_debug("NO_FACTS", query, "No facts were provided to Mentats.", session_id=session_id)
        return "PROTOCOL_ERROR: NO_FACTS"

    # Step 1: Draft from facts
    step1_input = (
        f"{MENTATS_STEP1_PROMPT}\n\n"
        "QUERY:\n"
        f"{query}\n\n"
        "FACTS_BLOCK:\n"
        f"{facts_block}\n\n"
        "CONSTRAINTS_BLOCK:\n"
        f"{constraints_block}\n"
    )

    step1_output = call_model(
        role=thinker_role,
        prompt=step1_input,
        max_tokens=max_tokens_step1,
        temperature=0.2,  # Lower temp for less creativity
        top_p=0.85,       # Lower top_p for less diversity
    )
    _log_mentats_debug("STEP1", query, step1_output, session_id=session_id)

    if (step1_output or "").strip().startswith("PROTOCOL_ERROR"):
        return step1_output.strip()

    # Step 2: Critic hunts for violations
    step2_input = f"{MENTATS_STEP2_PROMPT}\n\n{step1_output}\n"
    step2_output = call_model(
        role=critic_role,
        prompt=step2_input,
        max_tokens=max_tokens_step2,
        temperature=0.1,  # Very low temp for critic
        top_p=0.8,
    )
    _log_mentats_debug("STEP2", query, step2_output, session_id=session_id)

    # Step 3: Final answer (removes violations, doesn't replace)
    step3_input = (
        f"{MENTATS_STEP3_PROMPT}\n\n"
        "=== STEP1_BEGIN ===\n"
        f"{step1_output}\n"
        "=== STEP1_END ===\n\n"
        "=== STEP2_BEGIN ===\n"
        f"{step2_output}\n"
        "=== STEP2_END ===\n"
    )

    final_output = call_model(
        role=thinker_role,
        prompt=step3_input,
        max_tokens=max_tokens_step3,
        temperature=0.2,
        top_p=0.85,
    )
    _log_mentats_debug("STEP3", query, final_output, session_id=session_id)

    return final_output

