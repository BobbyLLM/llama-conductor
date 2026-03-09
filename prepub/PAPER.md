# Preprint Draft: Reliability Evaluation of a Local 4B LLM using deterministic router grounding

Status: pre-publication draft for peer-review preparation. Updated 2026-03-10 (WIP)

## Abstract

This preprint evaluates a local 4B configuration (`Qwen3-4B-Hivemind`) under two matched conditions:

1. `hivemind_no_scratch` (prompt-only evidence)
2. `hivemind_plus_scratch` (scratch-grounded evidence path)

Across two matched batteries:

- baseline: `120 prompts x 2 conditions = 240` runs
- replication: `500 prompts x 2 conditions = 1000` runs

Grounded mode (`plus_scratch`) reduced hallucination flags versus no-scratch in both batteries:

- 240 battery: `4/120 (3.3%) -> 0/120 (0.0%)`
- 1000 battery: `7/500 (1.4%) -> 1/500 (0.2%)`

Refusal correctness also strengthened under grounded mode:

- 240 battery: `1.60 -> 2.00`
- 1000 battery: `1.83 -> 2.00`

Tradeoff: contradiction handling regressed in the 1000 battery (source prioritization `2.00 -> 1.00`, uncertainty appropriateness `1.98 -> 1.00`). Evidence classification improved (`0.92 -> 2.34`). This indicates grounding reduced hallucination exposure while introducing a measurable contradiction-handling cost in this benchmark.

Claims are explicitly bounded to this benchmark design, runtime settings, and scoring rubric.

Blinded-rating note:
- A two-rater blinded adjudication protocol exists in project docs, but it was not included for the results in this evidence pack.
- Therefore, claims here remain bounded to the automated rubric/scoring outputs and published artifacts.
- This will be updated prior to submitting for publication.

## 1. Evidence Pack Included in This Repo (`/prepub`)

- `prepub/VALIDATION_BATTERY_REPORT.md`
- `prepub/VALIDATION_BATTERY_REPORT2.md`
- `prepub/validation_battery_raw2.jsonl`
- `prepub/validation_battery_scored2.jsonl`
- `prepub/VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md`
- `prepub/validation_battery_raw2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_scored2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_meta_v2_1000_20260309T130027Z.json`
- `prepub/prompts_manifest2_v2_1000_20260309T130027Z.jsonl`
- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`

These files are the basis for all numeric claims below.

## 2. Experimental Setup (frozen per battery)

From `VALIDATION_BATTERY_REPORT2.md`:

- model: `Qwen3-4B-Hivemind`
- temperature: `0.2`
- top_p: `0.9`
- max_tokens: `768`
- context target: `8192`
- runs: `120 prompts x 2 conditions = 240`

From `VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md` and `validation_battery_meta_v2_1000_20260309T130027Z.json`:

- model: `Qwen3-4B-Hivemind`
- temperature: `0.2`
- top_p: `0.9`
- max_tokens: `768`
- context target: `8192`
- runs: `500 prompts x 2 conditions = 1000`

### 2.1 Benchmark dimensions (what was actually tested)

- `reversal`: tests whether the model can re-adjudicate when frame assumptions invert, instead of sticking to the first answer by inertia.
- `tom`: tests perspective separation (who knows what, who believes what) without collapsing roles into one blended view.
- `evidence`: tests claim-label discipline (`VERIFIED`/`SUPPORTED`/`ASSERTED`) and suppression of unsupported upgrades.
- `retraction`: tests whether correction is incorporated cleanly when new information invalidates an earlier answer.
- `contradiction`: tests conflict handling when sources disagree: detect conflict, prioritize source, and express uncertainty appropriately.
- `negative_control`: tests refusal floor behavior when evidence is insufficient, to reduce fabricated completions.

## 3. Core Results

### 3.1 Run Integrity + Latency

240 battery:
- total runs: `240`
- errors: `0`
- format retries: `40`

1000 battery:
- total runs: `1000`
- errors: `0`
- format retries: `220`

Latency (240 battery):

- no_scratch: `p50=7.81s`, `p95=13.46s`, `p99=14.95s`, `mean=8.02s`
- plus_scratch: `p50=7.66s`, `p95=12.70s`, `p99=14.19s`, `mean=8.15s`

Interpretation: reliability constraints did not produce a large latency penalty in this setup.

### 3.2 Hallucination/Refusal Signals (Both Batteries)

240 battery:
- hallucination flags:
  - no_scratch: `4/120 (3.3%)`
  - plus_scratch: `0/120 (0.0%)`
- refusal correctness (negative controls):
  - `1.60 -> 2.00` (delta `+0.40`, 95% CI `[+0.10, +0.80]`)

1000 battery (replication):
- hallucination flags:
  - no_scratch: `7/500 (1.4%)`
  - plus_scratch: `1/500 (0.2%)`
- refusal correctness (negative controls):
  - `1.83 -> 2.00` (delta `+0.17`)

Interpretation: hallucination reduction was observed in both batteries. Grounded mode consistently reduced observed hallucination flags and improved refusal behavior under this benchmark setup.

### 3.3 Tradeoffs Observed (Regression Signal)

240 battery tradeoffs:
- reversal hedging: `0.15 -> 0.70` (worse)
- contradiction uncertainty appropriateness: `1.90 -> 1.15` (worse)
- evidence classification accuracy: `3.60/6 -> 3.45/6` (small negative shift)

1000 battery tradeoffs:
- contradiction detection: `2.00 -> 0.00` (worse under plus_scratch)
- contradiction source prioritization: `2.00 -> 1.00` (worse)
- contradiction uncertainty appropriateness: `1.98 -> 1.00` (worse)
- evidence classification improved: `0.92 -> 2.34` (better under plus_scratch)
- reversal hedging: `0.24 -> 0.26` (near-flat)

Interpretation: grounded mode achieved hallucination suppression with weaker contradiction handling in this battery. This may be acceptable for some safety-prioritized workflows, but it is not a universal improvement across all reasoning dimensions.

## 4. Statistical Bound (Zero-Event Caveat)

For the 240 battery `0/120` hallucination observation in grounded mode, the Rule-of-Three 95% upper bound is approximately:

- `3 / 120 ~= 2.5%`

For the 1000 battery grounded condition (`1/500`), observed rate is `0.2%` (non-zero).

So this is not a universal zero-hallucination claim. It is a bounded observation under tested conditions.

## 5. Mechanism Hypothesis

Observed behavior is consistent with a constrained-generation mechanism:

- evidence-bounded context reduces open recall pressure
- structured output constraints reduce free-form drift
- low-temperature decoding reduces sampling variance
- explicit refusal paths avoid forced completion when support is weak

This likely explains gains in hallucination/refusal metrics and may also contribute to the contradiction calibration tradeoff.

## 6. Clinical Workflow Validation (Test-Case Evidence)

This repository also includes a separate workflow-stability battery using clinical-style test cases.

From `prepub/_STRESS-CLINIKO-REVIEW-v1.md`:

- total runs: `45`
- passed: `45`
- failed: `0`
- errors: `0`
- pass rate: `100.0%`
- off-target term runs: `0` (`0.0%`)

Per-case repeatability (15 repeats each):

- `TEST-CASE-LBP.txt`: stable diagnosis `Mechanical low back pain` (`15/15`)
- `TEST-CASE-Shoulder.txt`: stable diagnosis `Rotator cuff tendinopathy` (`15/15`)
- `TEST-CASE-cervical.txt`: stable diagnosis `Mechanical neck pain` (`15/15`)

Expanded replay (same command-path contract, larger case set) from `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`:

- protocol: `>>cliniko auto` with pasted `TEST-*.txt` case -> `>>cliniko review` -> `>>flush` -> guard check (`>>cliniko review` should fail-loud with no scaffold)
- case set: `11` core test files (`TEST-CASE-*.txt`)
- repeats: `15` per case
- total runs: `165`
- passed: `165`
- failed: `0`
- errors: `0`
- unknown: `0`
- pass rate: `100.0%`
- off-target term runs: `0` (`0.0%`)
- flush-guard failures: `0`

From `prepub/GPT-AUTO-TEST-v3.md`:

- overall average (successful runs): `85.82`
- previous-cohort average shift: `78.83 -> 85.33` (`+6.50`)
- review validation pass count: `11/11 -> 11/11`
- auto-header duplication: fixed (`avg 2.0 -> 1.0`)
- stress-run review pass rate: `45/45` (`100.0%`)

Interpretation boundary:

- these are workflow consistency and structural validity signals
- they are not prospective clinical efficacy outcomes

Rule-of-Three notes for zero-event stress observations:

- legacy stress subset (`0/45`): 95% upper bound ~`6.7%`
- expanded replay (`0/165`): 95% upper bound ~`1.8%`

## 7. What This Does and Does Not Establish

### Supported by current data pack

1. In this benchmark, grounded path reduced observed hallucination flags (replicated in both batteries).
2. In this benchmark, grounded path improved refusal correctness (replicated in both batteries).
3. Grounded path traded weaker contradiction handling for hallucination suppression in the larger replication battery.
4. Clinical workflow test cases showed full repeatability across expanded case set (`165/165`) for the tested protocol.

### Not established by this pack alone

1. Universal behavior outside this benchmark shape.
2. Open-domain long-horizon reliability guarantees.
3. Cross-model invariance ("model-agnostic guarantee") without dedicated multi-model replications.
4. Real-world impact of contradiction regressions without domain-specific adjudication.

## 8. Reproducibility Notes

This repository contains:

- raw run outputs (`validation_battery_raw2.jsonl`)
- scored outputs (`validation_battery_scored2.jsonl`)
- replication raw/scored/meta/prompts (`validation_battery_*_v2_1000_20260309T130027Z.*`)
- summary reports and CIs

Next peer-review step is to publish the benchmark runner and rubric scripts as a turnkey reproduction package.
Next validation step is to include the two-rater blinded adjudication path and publish inter-rater agreement outputs.

## 9. Conclusion

Under bounded grounding constraints via Python router, this local 4B setup demonstrates the following reliability-oriented shift:

- hallucination suppression: fewer observed hallucination flags in the tested battery (`3.3% -> 0.0%` at `n=120`; `1.4% -> 0.2%` at `n=500`)
- refusal strength: stronger refusal behavior on insufficient-evidence prompts (`1.60 -> 2.00`; `1.83 -> 2.00`)
- measured tradeoff: weaker contradiction detection/source prioritization under grounding in the larger replication battery

The hallucination reduction signal replicated at larger sample size in this bounded benchmark. Contradiction regression was also observed and should be treated as an explicit deployment tradeoff requiring use-case-specific review.
