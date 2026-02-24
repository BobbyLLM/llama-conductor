# Preprint Draft: Reliability Evaluation for a Local 4B System via Deterministic router grounding 

Status: pre-publication draft for peer-review preparation.

## Abstract

This preprint evaluates a local 4B configuration (`Qwen3-4B-Hivemind`) under two matched conditions:

1. `hivemind_no_scratch` (prompt-only evidence)
2. `hivemind_plus_scratch` (scratch-grounded evidence path)

Across 120 prompts x 2 conditions (240 total runs), grounded mode (`plus_scratch`) reduced observed hallucination flags in this battery from `4/120 (3.3%)` to `0/120 (0.0%)`, and improved refusal correctness on negative controls from `1.60` to `2.00`. Tradeoffs were also observed: slightly higher reversal hedging (`0.15 -> 0.70`) and slightly weaker contradiction uncertainty appropriateness (`1.90 -> 1.15`).

Claims are explicitly bounded to this benchmark design, runtime settings, and scoring rubric.

## 1. Evidence Pack Included in This Repo (`/prepub`)

- `prepub/VALIDATION_BATTERY_REPORT.md`
- `prepub/VALIDATION_BATTERY_REPORT2.md`
- `prepub/validation_battery_raw2.jsonl`
- `prepub/validation_battery_scored2.jsonl`
- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`

These files are the basis for all numeric claims below.

## 2. Experimental Setup (frozen)

From `VALIDATION_BATTERY_REPORT2.md`:

- model: `Qwen3-4B-Hivemind`
- temperature: `0.2`
- top_p: `0.9`
- max_tokens: `768`
- context target: `8192`
- runs: `120 prompts x 2 conditions = 240`

## 3. Core Results

### 3.1 Run Integrity + Latency

- total runs: `240`
- errors: `0`
- format retries: `40`

Latency:

- no_scratch: `p50=7.81s`, `p95=13.46s`, `p99=14.95s`, `mean=8.02s`
- plus_scratch: `p50=7.66s`, `p95=12.70s`, `p99=14.19s`, `mean=8.15s`

Interpretation: reliability constraints did not produce a large latency penalty in this setup.

### 3.2 Hallucination/Refusal Signals

- hallucination flags:
  - no_scratch: `4/120 (3.3%)`
  - plus_scratch: `0/120 (0.0%)`
- refusal correctness (negative controls):
  - `1.60 -> 2.00` (delta `+0.40`, 95% CI `[+0.10, +0.80]`)

### 3.3 Tradeoffs Observed

- reversal hedging: `0.15 -> 0.70` (worse)
- contradiction uncertainty appropriateness: `1.90 -> 1.15` (worse)
- evidence classification accuracy: `3.60/6 -> 3.45/6` (small negative shift)

This is the key message: better bounded reliability on hallucination/refusal dimensions, with measurable calibration tradeoffs on some adversarial dimensions.

## 4. Statistical Bound (zero-event caveat)

For the `0/120` hallucination observation in grounded mode, the Rule-of-Three 95% upper bound is approximately:

- `3 / 120 ~= 2.5%`

So this is **not** a universal zero-hallucination claim. It is a bounded observation under tested conditions.

## 5. Mechanism Hypothesis

Observed behavior is consistent with a constrained-generation mechanism:

- evidence-bounded context reduces open recall pressure
- structured output constraints reduce free-form drift
- low-temperature decoding reduces sampling variance
- explicit refusal paths avoid forced completion when support is weak

This likely explains the gains in hallucination/refusal metrics and part of the calibration tradeoff profile.

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

From `prepub/GPT-AUTO-TEST-v3.md`:

- overall average (successful runs): `85.82`
- previous-cohort average shift: `78.83 -> 85.33` (`+6.50`)
- review validation pass count: `11/11 -> 11/11`
- auto-header duplication: fixed (`avg 2.0 -> 1.0`)
- stress-run review pass rate: `45/45` (`100.0%`)

Interpretation boundary:

- these are workflow consistency and structural validity signals
- they are not prospective clinical efficacy outcomes

Rule-of-Three note for zero-event stress observations (`0/45`):

- 95% upper bound approximately `6.7%`

## 7. What This Does and Does Not Establish

### Supported by current data pack

1. In this benchmark, grounded path reduced observed hallucination flags.
2. In this benchmark, grounded path improved refusal correctness.
3. Tradeoffs in hedging/calibration are measurable non zero.

### Not established by this pack alone

1. Universal behavior outside this benchmark shape.
2. Open-domain long-horizon reliability guarantees.
3. Cross-model invariance ("model-agnostic guarantee") without dedicated multi-model replications.

## 8. Reproducibility Notes

This repository contains:

- raw run outputs (`validation_battery_raw2.jsonl`)
- scored outputs (`validation_battery_scored2.jsonl`)
- summary reports and CIs

Next peer-review step is to publish the benchmark runner + rubric scripts as a turnkey reproduction package.

## 9. Conclusion

Under bounded grounding constraints, via python router, this local 4B setup demonstrates following reliability-oriented shift:

- fewer observed hallucination flags in the tested battery
- stronger refusal behavior on insufficient-evidence prompts
- explicit tradeoffs in adversarial calibration dimensions

This is enough to justify peer-review submission as an architecture-and-evaluation study, provided claims remain bounded to the published evidence.
