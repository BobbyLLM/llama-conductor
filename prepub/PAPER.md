# Preprint Draft v5: Reliability of a Local 4B Router-Grounded Stack

Status: pre-publication draft for peer-review preparation. Updated 2026-03-14.

## Abstract

This draft reports bounded reliability testing of a local 4B router-grounded stack at fixed settings (`temperature=0.2`, `top_p=0.9`, `max_tokens=768`, `--ctx 8192`) across raw and routed conditions.
The mechanism is contract-bounded generation enforced by routing policy, explicit grounding constraints, and fail-loud behavior. Benchmark-level format retry/scoring is handled by the external validation harness.

Benchmark runs analyzed in this draft: `8764`.

- Legacy + attribution + post-policy core corpus: `5304`
- Cross-family expansion campaign: `3300`
- Post-patch targeted revalidation: `160`

Separate workflow-stability evidence adds `210` runs (`45 + 165`) and is reported as an out-of-band stream.

Main findings:

1. Early routed Hivemind batteries showed hallucination-flag suppression under grounding (`3.3% -> 0.0%` at 240; `1.4% -> 0.2%` at 1000), with measured contradiction tradeoffs.
2. Pre-policy routed Qwen2507 did not show scratch uplift (`0.2%` no_scratch vs `0.4%` plus_scratch) and incurred `24.9%` format retries.
3. Post-policy routed Qwen2507/Hivemind reruns and missing-lane closures reached floor-level outcomes with zero retries in those slices.
4. Cross-family expansion showed model-policy interaction effects:
   - Granite routed `11/1000` (mostly negative-control lexical/contract family)
   - Phi routed `46/1000` (mostly negative-control lexical/contract family)
   - SmolLM3 routed `78/1000` with strong plus-scratch concentration (`75/500` vs `3/500` no_scratch), dominated by contradiction/reversal lane-quality degradation.
5. A sandbox surgical lane patch followed by targeted rerun (`160` runs) yielded `0/160` flags and `0/160` errors across Granite/Phi/Smol affected lanes.

Interpretation: this evidence supports a bounded system-level control claim (policy/routing can dominate observed hallucination-flag outcomes in tested tasks under this harness), while also showing nontrivial model-policy interaction in specific lanes.

Blinded adjudication note:

- A two-rater blinded protocol exists in project docs.
- Reported results in this draft remain automated-rubric outputs.

## 1. Scope and Protocol

The target claim is constrained truthfulness/faithfulness under grounded tasks, not universal open-domain reasoning.

Common run settings:

- temperature `0.2`
- top_p `0.9`
- max_tokens `768`
- context target `8192`

Operational controls in extension campaigns:

- thermal guard `86C` trigger / `87C` hard / `82C` resume
- 10-minute inter-block cooldown
- sequential execution (single active serving path)

### 1.1 Runtime vs Evaluation Responsibilities

| Component | Runtime system | Evaluation harness |
|---|---|---|
| Lane selection | yes | no |
| Contract/policy enforcement | yes | no |
| Deterministic fail-loud behavior | yes | no |
| Format retry | no (handled by lane logic, not benchmark retry loop) | yes (single bounded retry in battery runner) |
| Rubric scoring | no | yes |
| Hallucination-flag assignment | no | yes |
| Taxonomy aggregation/reporting | no | yes |

## 2. Task Dimensions

The benchmark prompt set covers six categories:

- `reversal`
- `tom`
- `evidence`
- `retraction`
- `contradiction`
- `negative_control`

### 2.1 Task Difficulty Evidence (Representative Examples)

The benchmark is not limited to single-fact lookup. It includes frame inversion, perspective separation, correction handling, contradiction adjudication, and refusal-floor checks.

- `reversal`:
  - prompt pattern: initial claim adjudication, then explicit inversion of key premise.
  - expected behavior: revise answer and state why prior conclusion no longer holds.
- `tom`:
  - prompt pattern: multiple actors with different beliefs/knowledge states.
  - expected behavior: keep role-conditioned beliefs separate without leakage.
- `evidence`:
  - prompt pattern: mixed support strength from provided context.
  - expected behavior: preserve label discipline (`VERIFIED`/`SUPPORTED`/`ASSERTED`) without unsupported upgrades.
- `retraction`:
  - prompt pattern: follow-up correction invalidates earlier assumption.
  - expected behavior: update answer state and retire prior invalid claim.
- `contradiction`:
  - prompt pattern: conflicting statements/sources within bounded context.
  - expected behavior: detect conflict, prioritize source, express uncertainty when unresolved.
- `negative_control`:
  - prompt pattern: insufficient support by design.
  - expected behavior: explicit refusal/insufficient-evidence response, no fabrication.

### 2.2 Worked Example (Artifact-Grounded)

Example category: `negative_control` (from errata-reviewed rows).

- Prompt intent: ask for causal mechanism/threshold not present in provided evidence.
- Expected behavior: explicit insufficient-evidence refusal.
- Observed response family (flagged rows in pre-fix runs): short scratch acknowledgement/refusal-like phrasing without strict refusal contract tokenization.
- Rubric outcome in those rows: `hallucination_flag=1` with `refusal_correctness=0`.
- Taxonomy classification: `lexical/contract` (not auto-promoted to confirmed fabrication without manual adjudication).

Reference: [ERRATA.md](C:/moa-router1.2.1%20TESTING/prepub/ERRATA.md)

## 3. Results

### 3.0 Runtime Router Pipeline (Mechanism Diagram)

```text
Prompt
  ↓
Router (lane selection)
  ↓
Lane contract + grounding policy
  ↓
Model call
  ↓
Post-process / contract/source/footer normalization
  ↓
Deterministic lane fail-loud (when applicable) or finalized response
```

### 3.0.1 Runtime Router Control (Pseudocode)

```python
def run_turn(prompt):
    lane = select_lane(prompt)
    contract = lane_contract(lane)
    response = model_call(prompt, lane=lane, contract=contract)
    response = postprocess(response, lane=lane, contract=contract)
    if deterministic_lane_fail_loud_triggered(response, lane):
        return fail_loud_response(response, lane)
    return finalize_response(response, lane)
```

### 3.0.2 Benchmark Evaluation Harness (Diagram + Pseudocode)

The validation battery executes an external evaluation loop over router responses.

```text
Prompt item + condition
  ↓
Router call
  ↓
Format/contract check (harness)
  ↓
If invalid: one bounded retry (harness)
  ↓
Rubric scoring (harness)
  ↓
Taxonomy mapping + reporting artifacts
```

```python
def eval_one_item(item, condition):
    resp1 = call_router(item, condition)
    ok1 = validate_format(item.category, resp1)
    if not ok1:
        resp2 = call_router(item, condition)  # single bounded retry in harness
        resp = resp2
    else:
        resp = resp1
    scores = score_item(item, resp)  # includes hallucination_flag
    return scores
```

### 3.1 Legacy Hivemind Routed Batteries

- 240 battery hallucination-flag rate: `3.3% -> 0.0%`
- 1000 battery hallucination-flag rate: `1.4% -> 0.2%`
- refusal correctness: `1.60 -> 2.00` (240), `1.83 -> 2.00` (1000)

Legacy tradeoff signal under plus_scratch:

- contradiction detection `2.00 -> 0.00`
- contradiction source prioritization `2.00 -> 1.00`
- contradiction uncertainty `1.98 -> 1.00`

### 3.2 Pre-Policy Attribution (Qwen/Hivemind)

Raw baselines:

- Qwen2507 raw: `4/500` (`0.8%`)
- Hivemind raw: `9/500` (`1.8%`)

Routed Qwen2507 (`1000` paired):

- no_scratch: `1/500` (`0.2%`)
- plus_scratch: `2/500` (`0.4%`)
- format retries: `249/1000` (`24.9%`)

### 3.3 Post-Policy Core Reruns

Completed blocks:

- Qwen2507 raw `100`: `0/100` hallucination-flag count
- Hivemind raw `100`: `0/100` hallucination-flag count
- routed Hivemind `600`: `0/600` hallucination-flag count, `0` retries
- routed Qwen2507 `600`: `0/600` hallucination-flag count, `0` retries

Coverage note: the `600 + 600` routed slices covered `reversal/tom/evidence/retraction`; contradiction and negative_control were closed in missing-lane runs.

### 3.4 Missing-Lane Closures (Qwen2507/Hivemind)

Each closure run targets only `contradiction + negative_control` with paired no_scratch/plus_scratch runs.

- routed Qwen2507 closure (`332`):
  - hallucination-flag count: `0/332`
  - format retries: `0/332`

- routed Hivemind closure (`332`):
  - hallucination-flag count: `0/332`
  - format retries: `0/332`

### 3.5 Cross-Family Expansion Campaign (`3300`)

#### granite-4.0-micro-ablit

- raw `100`: hallucination-flag count `0`, errors `0`
- routed `1000`: hallucination-flag count `11`, errors `0`, format retries `1`
- routed category split: contradiction `1`, negative_control `10`

#### Phi-4-mini

- raw `100`: hallucination-flag count `2`, errors `2` (read timeouts)
- routed `1000`: hallucination-flag count `46`, errors `0`, format retries `173`
- routed category split: contradiction `2`, negative_control `44`

#### SmolLM3

- raw `100`: hallucination-flag count `0`, errors `0`
- routed `1000`: hallucination-flag count `78`, errors `0`, format retries `707`
- routed condition split: `no_scratch 3/500`, `plus_scratch 75/500`
- routed category split: contradiction `52`, reversal `17`, negative_control `8`, evidence `1`

### 3.6 Post-Patch Targeted Revalidation (`160`)

A surgical sandbox patch was applied to lane contracts and strict output handling for affected categories.

- Granite fixcheck `40`: flags `0`, errors `0`
- Phi fixcheck `40`: flags `0`, errors `0`
- SmolLM3 fixcheck `80`: flags `0`, errors `0`
- aggregate: `160/160` clean

This is targeted affected-lane validation, not a full replacement rerun of all three routed `1000` blocks.

## 4. Error Taxonomy (Consolidated)

Final campaign classification (see `prepub/ERRATA.md` for artifact-level detail):

- `runtime/transport`:
  - routed blocks: `0`
  - observed only in Phi raw (`2` read timeouts)
- `lexical/contract`:
  - dominant in Granite/Phi negative-control failures
- `rubric-coupling`:
  - contradiction rows with strong subscores but conservative `hallucination_flag` coupling
- `lane-quality degradation`:
  - dominant in SmolLM3 routed plus-scratch contradiction/reversal lanes

No row was promoted to confirmed true fabrication without manual adjudication in this audit pass.

### 4.1 Hallucination Scoring Protocol (Explicit, Harness-Level)

Rubric inputs (per run):

- prompt metadata: model, condition, lane/category, run id
- model output payload (contracted response fields)
- parser/validator outcome (schema/header/label compliance)
- scorer fields (category sub-scores + final `hallucination_flag`)

Decision flow (validation harness):

1. Evaluate transport/runtime status.
2. Validate contract compliance (required labels/headers/schema).
3. Compute rubric category subscores from final response.
4. Assign preliminary `hallucination_flag` from rubric output.
5. Map flagged rows into taxonomy (`runtime/transport`, `lexical/contract`, `rubric-coupling`, `lane-quality`, or `confirmed hallucination` after adjudication).

Adjudication boundary:

- automated: all run-level rubric and taxonomy preclassification
- manual: promotion to `confirmed true hallucination` and final dispute resolution

Interpretation rule:

- reported hallucination-flag rows are rubric outputs, not automatic proof of fabrication.
- confirmed fabrication requires adjudication beyond parser/contract and rubric artifacts.

### 4.2 Why Zero Observed Hallucination Is Plausible (Bounded Regime)

Zero-observed slices are plausible in this setup because generation is constrained by:

- lane contracts that narrow admissible output states
- bounded grounding policy that suppresses unsupported free generation
- fail-loud retry logic that rejects contract-violating responses
- narrow task scope and fixed runtime settings (`8K`, low temperature)

Therefore, zero-observed outcomes are interpreted as bounded protocol behavior under this harness, not a universal claim of intrinsic model truthfulness.

## 5. Interpretation

Strongest bounded position from current evidence:

1. Policy-constrained routed orchestration can suppress observed hallucination-flag rates to floor-level in several routed slices.
2. Outcomes are model-policy interaction effects, not universal model-family guarantees.
3. Cross-family expansion demonstrates that plus-scratch can reveal model-policy lane brittleness (SmolLM3 contradiction/reversal) even when other families remain near floor in routed conditions.
4. Failure modes are auditable and classifiable; they are not opaque runtime instability.
5. Surgical lane-targeted policy changes can materially clear affected lanes in targeted revalidation.

This is a control-layer reliability result under bounded tasks. It is not evidence that base model truthfulness is universally solved.

## 6. Limitations

- Context regime is `8K`; no direct external validity to `32K+` from this pack.
- Reported outcomes are rubric-automated, not yet two-rater blinded adjudication outcomes.
- Post-patch evidence currently uses targeted reruns (`160`), not full `3 x 1000` routed replacement runs.
- Results are benchmark-bounded and should not be generalized to all prompt classes.

## 7. Separate Clinical Workflow Stream (Out-of-Band)

Workflow-stability stream totals `210` runs (`45 + 165`) with published artifacts.

Boundary condition:

- `>>cliniko` generation is deterministic Python sidecar logic.
- `>>cliniko review` uses `Qwen2.5-1.5B`.
- This stream is operational workflow evidence, not Qwen3-4B scratch attribution evidence.

## 8. Conclusion

This project is best represented as a bounded reliability harness with auditable failure modes and measurable policy effects.

Across all benchmark campaigns in this draft (`8764` runs), evidence supports a strong system-level control claim under bounded tasks: policy/routing can drive observed hallucination-flag rates to floor in tested slices, but behavior remains sensitive to model-policy fit by lane. The SmolLM3 routed block demonstrates that this fit can degrade sharply without lane-scoped hardening; the post-patch targeted rerun demonstrates that these degradations are actionable and correctable.

## 9. Evidence Pack (Backmatter)

### 9.1 Legacy routed batteries and attribution core

- `prepub/VALIDATION_BATTERY_REPORT.md`
- `prepub/VALIDATION_BATTERY_REPORT2.md`
- `prepub/VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md`
- `prepub/validation_battery_raw2.jsonl`
- `prepub/validation_battery_scored2.jsonl`
- `prepub/validation_battery_raw2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_scored2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_meta_v2_1000_20260309T130027Z.json`
- `prepub/prompts_manifest2_v2_1000_20260309T130027Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT.md`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-postpolicy-1400-20260312T150117Z.md`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_qwen2507_raw_1000_20260309T175805Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_qwen2507_raw_1000_20260309T175805Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_hivemind_raw_1000_20260309T191228Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_hivemind_raw_1000_20260309T191228Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_1000_20260309T232905Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_qwen2507_routed_1000_20260309T232905Z.jsonl`

### 9.2 Missing-lane closures

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-missing-lanes-332-hivemind-20260313T091848Z.md`
- `TEST_ARTIFACTS_VALIDATION/missing_lanes_332_hivemind_meta_20260313T091848Z.json`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_252_20260313T094659Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_252_20260313T094659Z.jsonl`

### 9.3 Cross-family campaign (`3300`)

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-campaign-3300-queue.md`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_granite40_micro_ablit_routed_1000_20260313T114724Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_phi4_mini_routed_1000_20260313T161354Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_smollm3_routed_1000_20260313T203358Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_granite40_micro_ablit_raw_100_20260313T112451Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_phi4_mini_raw_100_20260313T145626Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_smollm3_raw_100_20260313T200801Z.jsonl`

### 9.4 Post-patch targeted validation

- `TEST_ARTIFACTS_VALIDATION/fixval_scored_granite_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_phi_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_smol_fixcheck_20260314T084930Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_summary_20260314T084930Z.json`
- `prepub/ERRATA.md`

### 9.5 Clinical workflow stream

- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`

## 10. Consolidated Campaign Summary (Explicit Ledger)

### 10.1 Total Run Accounting

- Core benchmark corpus before latest expansion: `5304`
- Cross-family expansion campaign: `3300` (completed)
- Post-patch targeted validation: `160` (completed)
- Core benchmark total now: `8764`
- Separate workflow stream (out-of-band): `210` (`45 + 165`)
- Grand total including workflow stream: `8974`

### 10.2 Legacy Routed Hivemind Batteries

- `240` battery:
  - hallucination-flag rate: `3.3% -> 0.0%` (`no_scratch -> plus_scratch`)
- `1000` battery:
  - hallucination-flag rate: `1.4% -> 0.2%`
- Tradeoff signal observed in legacy runs:
  - contradiction handling weakened under plus-scratch in that phase.

### 10.3 Pre-Policy Attribution Block (Qwen/Hivemind)

- Raw baselines:
  - Qwen2507 raw: `4/500` (`0.8%`)
  - Hivemind raw: `9/500` (`1.8%`)
- Routed Qwen2507 (`1000`):
  - `no_scratch`: `1/500` (`0.2%`)
  - `plus_scratch`: `2/500` (`0.4%`)
  - major overhead: `249/1000` format retries (`24.9%`)

### 10.4 Post-Policy Resample + Missing-Lane Closures

- Post-policy resample (`1400`):
  - Qwen2507 raw `100`: `0/100`
  - Hivemind raw `100`: `0/100`
  - routed Hivemind `600`: `0/600`, retries `0`
  - routed Qwen2507 `600`: `0/600`, retries `0`
- Missing-lane closures:
  - routed Qwen2507 `332`: flags `0/332`, retries `0`
  - routed Hivemind `332`: flags `0/332`, retries `0`

### 10.5 Cross-Family Expansion Campaign (`3300`)

#### Granite-4.0-micro-ablit

- Raw `100`: flags `0`, errors `0`
- Routed `1000`: flags `11`, errors `0`, retries `1`
- Flags mostly `negative_control` lexical/contract family + `1` contradiction rubric-coupling anomaly.

#### Phi-4-mini

- Raw `100`: flags `2`, errors `2` (both transport timeouts)
- Routed `1000`: flags `46`, errors `0`, retries `173`
- Routed flags mostly `negative_control` lexical/contract family + `2` contradiction rubric-coupling anomalies.

#### SmolLM3

- Raw `100`: flags `0`, errors `0`
- Routed `1000`: flags `78`, errors `0`, retries `707`
- Condition split:
  - `no_scratch`: `3/500`
  - `plus_scratch`: `75/500`
- Category split:
  - `contradiction 52`, `reversal 17`, `negative_control 8`, `evidence 1`
- This is the strongest model-policy lane-quality degradation signal (not transport).

### 10.6 Errata Taxonomy (Well-Characterized)

- `runtime/transport`:
  - only `2` observed (Phi raw), none in finalized routed blocks.
- `lexical/contract`:
  - major family in Granite/Phi negative-control rows.
- `rubric-coupling`:
  - contradiction anomalies with strong subscores but conservative flag coupling.
- `lane-quality degradation`:
  - dominant in SmolLM3 routed plus-scratch contradiction/reversal lanes.
- `confirmed true hallucination`:
  - none promoted as confirmed in current errata pass without manual adjudication.

### 10.7 Surgical Patch + Targeted Revalidation (Sandbox)

- Patch scope: lane-scoped hardening in:
  - `chat_postprocess.py`
  - `chat_finalize.py`
  - `router_fastapi.py`
- Targeted revalidation (`160`):
  - Granite `40/40`: flags `0`, errors `0`
  - Phi `40/40`: flags `0`, errors `0`
  - SmolLM3 `80/80`: flags `0`, errors `0`
- Result: targeted affected-lane failures were eliminated in sandbox checks.

### 10.8 What This Shows

- Outcomes are **model-harness-policy interaction**, not a simple universal scratch toggle effect.
- Failures are diagnosable and correctable with narrow policy/contract changes.
- Routed operational stability is strong (transport near-zero in finalized routed blocks).

### 10.9 What Is Still Bounded

- No universal claim across all tasks/contexts.
- Long-context (`>8K`) generalization is not established by this evidence pack.
- Full post-patch `3 x 1000` replacement rerun is not yet done; current post-patch evidence is targeted (`160`) for fast engineering validation.

