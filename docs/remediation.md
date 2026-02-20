# REMEDIATION

Date: 2026-02-20
Scope: agreed changes only (for validation handoff to another repo/version)

## Decision Log (This Conversation)

- Accepted for this remediation pass: items `1`, `3`, `4`, `10`.
- Rejected for this pass: items `2`, `5`, `6`, `7`, `8`, `9`.

## Repo Scope Split

- Public repo validation scope excludes Cliniko routing changes.
- Private repo may retain Cliniko-specific paths/behavior independently.
- Therefore, Cliniko-path remediation is out of scope for this handoff document.

## Agreed Changes

1. Sync docs truth map with current code reality.
- Update `DOCS-TRUTH-MAP.md` to remove stale reference to `llama_conductor/SUMM.md` as an active authoritative doc.
- Ensure active-doc list reflects current post-cleanup state.

2. Centralize and align runtime version reporting.
- Eliminate version drift between package version and router-reported version.
- Implement a single source of truth for version (recommended: `llama_conductor/__about__.py` with `__version__`).
- Wire `/healthz` and any hardcoded router version strings to that source.
- Explicit files to update/verify:
  - `pyproject.toml` (`[project].version`)
  - `llama_conductor/router_fastapi.py` (module header/version text and `/healthz` payload)

3. Add pre-release consistency automation.
- Add a lightweight pre-release check script to validate:
  - version consistency (package metadata vs runtime-reported version),
  - license consistency across metadata/docs/license file,
  - active-doc references do not point to deleted/stale files.
- Script should fail loud with actionable output and be runnable before promotion/public publish.
- Minimum concrete checks:
  - `pyproject.toml` license field matches `LICENSE` and license text in `README.md` / `FAQ.md`.
  - `pyproject.toml` version matches `llama_conductor/router_fastapi.py` runtime-reported version (or centralized `__about__.__version__`).

## Guardrail

- These remediation actions are metadata/docs/version-governance tasks and must not change router command behavior.
- Matrix parity is required after changes (`regressions=0` against approved baseline).

## Explicit Non-Scope (Deferred/Rejected for this pass)

- Cliniko path cleanup in public repo (not applicable to public routing split).
- Broad exception handling refactor in router hot path.
- Semantic test-suite expansion beyond current matrix strategy.
- Type-hygiene-only cleanup (`Optional[any]` style changes).
- Debug print gating/logging refactor.
- `>>doctor` / startup environment validator command.
- Immediate metadata/license change execution in this repo (unless separately approved).

## Implementation Status (2026-02-20)

- Item 1: implemented.
  - `pyproject.toml` license aligned to AGPL family (`AGPL-3.0-or-later`) to match `LICENSE`, `README.md`, and `FAQ.md`.
- Item 3: implemented.
  - `DOCS-TRUTH-MAP.md` active list updated to remove stale `llama_conductor/SUMM.md`.
- Item 4: implemented.
  - Added `llama_conductor/__about__.py` (`__version__` single source).
  - Wired `llama_conductor/router_fastapi.py` `/healthz` to `__version__`.
  - Router header version text updated to match current version.
- Item 10: implemented.
  - Added `tests/pre_release_consistency_check.py` to fail loud on:
    - version mismatch,
    - license mismatch,
    - missing active-doc references from `DOCS-TRUTH-MAP.md`.

## Validation Expectations

- No command-behavior regressions from baseline matrix scope.
- No public/private Cliniko routing scope confusion introduced.
- New checks are deterministic and runnable in CI or local pre-release workflow.

## Handoff Validation Checklist

- `DOCS-TRUTH-MAP.md` no longer lists `llama_conductor/SUMM.md` as active authority.
- Version values are consistent between:
  - `pyproject.toml` (`[project].version`)
  - runtime-reported router version (`/healthz` via `llama_conductor/router_fastapi.py` or centralized `__about__.__version__`).
- License declarations are consistent between:
  - `pyproject.toml`,
  - `LICENSE`,
  - `README.md`,
  - `FAQ.md`.
- Pre-release consistency script exists and fails loud on mismatch.
- Regression matrix rerun shows `regressions=0` vs approved baseline.

## Full Original Checklist Context (1-10) + Decision Rationale

1. Fix license mismatch (high risk, low effort)
- Decision: Accepted.
- Why: canonical license decision made for this pass; metadata/docs alignment is required for release consistency checks.

2. Remove hardcoded Cliniko prompt path (high risk, low effort)
- Decision: Rejected for this pass.
- Why: public-facing repo does not include active `>>cliniko` routing path; this was identified as private-scope behavior and not appropriate for this public handoff scope.

3. Sync truth-map docs (medium risk, low effort)
- Decision: Accepted.
- Why: agreed stale doc mapping should be corrected (`SUMM.md` authority drift) and aligns with current code/changelog reality.

4. Fix runtime version drift (medium risk, low effort)
- Decision: Accepted.
- Why: agreed to address version drift and preferred centralization approach; expected no functional behavior change when implemented cleanly.

5. Reduce broad exception masking in router hot path (medium risk, medium effort)
- Decision: Rejected for this pass.
- Why: not accepted due to unclear immediate value/justification for current goals; considered broader refactor outside this remediation handoff.

6. Strengthen test realism for semantic behavior (medium risk, medium effort)
- Decision: Rejected for this pass.
- Why: not accepted for current scope; user requested justification and chose not to include as part of this handoff batch.

7. Type cleanup and static checks (low risk, low effort)
- Decision: Rejected for this pass.
- Why: classified as non-breaking hygiene; not prioritized for this remediation package.

8. Operational hygiene for debug output (low risk, low effort)
- Decision: Rejected for this pass.
- Why: classified as non-breaking operational cleanup; not prioritized for this remediation package.

9. Config/environment validation command (low risk, medium effort)
- Decision: Rejected for this pass.
- Why: explicitly declined as messy for current scope.

10. Documentation policy automation (low risk, medium effort)
- Decision: Accepted.
- Why: explicitly agreed as useful; included as pre-release consistency automation requirement.
