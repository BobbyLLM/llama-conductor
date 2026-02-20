# DOCS TRUTH MAP

Last updated: 2026-02-13

Purpose: define which markdown files are active guidance vs archived history, so future work uses one source of truth.

## Active docs (authoritative)

- `README.md` - top-level project orientation.
- `NEW.md` - current feature/behavior deltas.
- `FAQ.md` - operational Q&A.
- `DESIGN.md` - architecture and design intent.
- `WORK-DONE-SO-FAR.md` - execution log and change chronology.
- `FIX-THIS-CLINIKO.md` - current Cliniko issue tracker and outcomes.
- `llama_conductor/command_cheat_sheet.md` - command behavior contract.
- `llama_conductor/CLINIKO-GOLD.md` - expected Cliniko output target.
- `llama_conductor/quotes.md` - curated reference snippets.
- `ChatGPT/CHATGPT-STANDING-ORDERS.md` - retained active operating guardrails.
- `ChatGPT/WHAT I WILL NOT DO.md` - retained active constraints.
- `llama_conductor/test tools/GPT-AUTO-TEST-v3.md` - latest consolidated test audit.
- `llama_conductor/test tools/GPT-AUTO-TEST-v2.md` - prior baseline audit for comparison.
- `llama_conductor/test tools/GOLD-STANDARD-FINAL-OUTPUT.md` - gold final reference.
- `llama_conductor/test tools/GOLD-STANDARD-RAW-note.md` - gold raw-note reference.
- `llama_conductor/test tools/_SMOKE-LBP-AUTO.md` - smoke evidence.
- `llama_conductor/test tools/_SMOKE-LBP-REVIEW.md` - smoke evidence.
- `llama_conductor/test tools/_SMOKE-KNEE-AUTO.md` - smoke evidence.
- `llama_conductor/test tools/_SMOKE-KNEE-REVIEW.md` - smoke evidence.
- `llama_conductor/test tools/_STRESS-CLINIKO-REVIEW-v1.md` - stress-run evidence.
- `SMOKE-SCRATCHPAD.md` - temporary but active working notes.

## Archived docs (non-authoritative)

Canonical archived copies live under `docs/stale/` and are intentionally gitignored.

Archived set:
- `docs/stale/ChatGPT__BEFORE_AFTER.md`
- `docs/stale/ChatGPT__CODEBASE_AUDIT_REPORT-09.02.26.md`
- `docs/stale/ChatGPT__DEPENDENCY_GRAPH_CORRECTED.md`
- `docs/stale/ChatGPT__DEPENDENCY_GRAPH_VERIFIED.md`
- `docs/stale/ChatGPT__FILE_AUDIT_VERIFIED.md`
- `docs/stale/ChatGPT__MODULAR_ARCHITECTURE_BLUEPRINT.md`
- `docs/stale/ChatGPT__NON_CODER_GUIDE.md`
- `docs/stale/ChatGPT__REFACTORING_COMPLETE.md`
- `docs/stale/ChatGPT__new_sidecar_ideas.md`
- `docs/stale/llama_conductor__test tools__CLAUDE-DEPLOYMENT_SUMMARY.md`
- `docs/stale/llama_conductor__test tools__CHATGPT-ASSESS-OF-TEST-OUTPUTS.md`

## Usage rule

If an active doc conflicts with any archived doc, trust the active doc and ignore the archived content.
