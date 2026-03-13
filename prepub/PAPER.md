# Preprint Draft v4: Reliability of a Local 4B Router-Grounded Stack

Status: pre-publication draft for peer-review preparation. Updated 2026-03-13.

## Abstract

This draft reports bounded reliability testing of a local 4B stack at fixed settings (`temperature=0.2`, `top_p=0.9`, `max_tokens=768`, `--ctx 8192`) across raw and routed conditions.

Benchmark runs analyzed in this draft: `5304`.

- Routed Hivemind batteries: `240` + `1000`
- Attribution blocks: Qwen2507 raw `500`, Hivemind raw `500`, routed Qwen2507 `1000`
- Post-policy resample: `1400`
- Missing-lane closure: routed Qwen2507 `332`, routed Hivemind `332`

Separate workflow-stability evidence adds `210` runs (`45 + 165`) and is reported as an out-of-band stream.

Main findings:

1. Early routed Hivemind batteries showed hallucination suppression under grounding (`3.3% -> 0.0%` at 240; `1.4% -> 0.2%` at 1000), with measured contradiction tradeoffs.
2. Pre-policy routed Qwen2507 did not show scratch uplift (`0.2%` no_scratch vs `0.4%` plus_scratch) and incurred `24.9%` format retries.
3. Post-policy routed resample removed retry overhead in tested routed tracks (`0/1200` retries).
4. Latest missing-lane closures (contradiction + negative_control only) finished at:
   - Qwen2507 routed: `0/332` hallucination flags, `0/332` retries
   - Hivemind routed: `0/332` hallucination flags, `0/332` retries

Interpretation: reliability effects are model-policy dependent, not model-agnostic. Claims are bounded to this benchmark family and `8K` context.

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

Operational controls used in extension campaigns:

- thermal guard `86C` trigger / `87C` hard / `82C` resume
- 10-minute inter-block cooldown
- sequential execution (single active serving path)

## 2. Task Dimensions

The benchmark prompt set covers six categories:

- `reversal`
- `tom`
- `evidence`
- `retraction`
- `contradiction`
- `negative_control`

## 3. Results

### 3.1 Routed Hivemind Baseline and Replication (Legacy)

- 240 battery hallucination flags: `3.3% -> 0.0%`
- 1000 battery hallucination flags: `1.4% -> 0.2%`
- refusal correctness: `1.60 -> 2.00` (240), `1.83 -> 2.00` (1000)

Legacy tradeoff signal under plus_scratch:

- contradiction detection `2.00 -> 0.00`
- contradiction source prioritization `2.00 -> 1.00`
- contradiction uncertainty `1.98 -> 1.00`

### 3.2 Attribution Block (Pre-Policy)

Raw baselines:

- Qwen2507 raw: `4/500` (`0.8%`)
- Hivemind raw: `9/500` (`1.8%`)

Routed Qwen2507 (1000 paired):

- no_scratch: `1/500` (`0.2%`)
- plus_scratch: `2/500` (`0.4%`)
- format retries: `249/1000` (`24.9%`)

### 3.3 Post-Policy Resample (1400)

Completed blocks:

- Qwen2507 raw `100`: `0/100` hallucination flags
- Hivemind raw `100`: `0/100` hallucination flags
- routed Hivemind `600`: `0/600` hallucination flags, `0` retries
- routed Qwen2507 `600`: `0/600` hallucination flags, `0` retries

Coverage note: the `600 + 600` routed slices covered `reversal/tom/evidence/retraction`; contradiction and negative_control were closed in missing-lane runs.

### 3.4 Missing-Lane Closure (Latest)

Each closure run targets only `contradiction + negative_control` with paired no_scratch/plus_scratch runs.

- routed Qwen2507 closure (`332`):
  - hallucination flags: `0/332`
  - format retries: `0/332`

- routed Hivemind closure (`332`, latest completed block ending `2026-03-13T10:40:43Z`):
  - hallucination flags: `0/332`
  - format retries: `0/332`
  - category counts: contradiction `0/166`, negative_control `0/166`

### 3.5 Operational Readout

- Major pre-policy routed cost was format retries (`24.9%`) in Qwen2507 routed.
- In post-policy routed runs plus both missing-lane closures, observed retries were `0`.
- Thermal envelope in routed tracks remained stable in the mid/high 70s C in captured reports.

## 4. Interpretation

Strongest bounded position from current evidence:

1. Routed grounding can suppress benchmark hallucination flags in this setup.
2. Outcomes are model-policy interaction effects, not universal model-family guarantees.
3. Contradiction handling is sensitive to policy wording and route behavior.
4. Retry burden is an explicit operational metric, not a hidden artifact.

## 5. Limitations

- Context regime is `8K`; no direct external validity to `32K+` from this pack.
- Reported outcomes are rubric-automated, not yet two-rater blinded adjudication outcomes.
- Results are benchmark-bounded and should not be generalized to all prompt classes.

## 6. Separate Clinical Workflow Stream (Out-of-Band)

Workflow-stability stream totals `210` runs (`45 + 165`) with published artifacts.

Boundary condition:

- `>>cliniko` generation is deterministic Python sidecar logic.
- `>>cliniko review` uses `Qwen2.5-1.5B`.
- This stream is operational workflow evidence, not Qwen3-4B scratch attribution evidence.

## 7. Conclusion

This project is best represented as a bounded reliability harness with auditable failure modes and measurable policy effects.

The latest closure runs materially strengthen the operational claim for routed stability in the tested lanes (`0/332` and `0/332`, both with `0` retries), while preserving the core caveat: transferability across models, contexts, and task families is not guaranteed without direct reruns.

## 8. Evidence Pack (Backmatter)

### 8.1 Legacy routed batteries

- `prepub/VALIDATION_BATTERY_REPORT.md`
- `prepub/VALIDATION_BATTERY_REPORT2.md`
- `prepub/VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md`
- `prepub/validation_battery_raw2.jsonl`
- `prepub/validation_battery_scored2.jsonl`
- `prepub/validation_battery_raw2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_scored2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_meta_v2_1000_20260309T130027Z.json`
- `prepub/prompts_manifest2_v2_1000_20260309T130027Z.jsonl`

### 8.2 Attribution and post-policy artifacts

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT.md`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-postpolicy-1400-20260312T150117Z.md`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_qwen2507_raw_1000_20260309T175805Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_qwen2507_raw_1000_20260309T175805Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_hivemind_raw_1000_20260309T191228Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_hivemind_raw_1000_20260309T191228Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_1000_20260309T232905Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_qwen2507_routed_1000_20260309T232905Z.jsonl`

### 8.3 Missing-lane closure (publication-labeled artifacts)

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-missing-lanes-332-hivemind-20260313T091848Z.md`
- `TEST_ARTIFACTS_VALIDATION/missing_lanes_332_hivemind_meta_20260313T091848Z.json`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_252_20260313T094659Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_252_20260313T094659Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/VALIDATION_BATTERY_REPORT2_hivemind_routed_80_20260313T091901Z.md`
- `TEST_ARTIFACTS_VALIDATION/VALIDATION_BATTERY_REPORT2_hivemind_routed_252_20260313T094659Z.md`

### 8.4 Clinical workflow stream

- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`
