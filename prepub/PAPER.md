# Preprint Draft v2: Reliability Evaluation of a Local 4B LLM with Attribution Extension

Status: pre-publication draft for peer-review preparation. Updated 2026-03-12 (WIP)

## Abstract

This draft extends prior reliability results for a local 4B setup by adding a model-vs-harness attribution series.

Prior matched batteries (same runtime contract, routed conditions) using `Qwen3-4B-Hivemind` showed:

- 240 runs (`120 x 2`): hallucination flags `3.3% -> 0.0%` (`no_scratch -> plus_scratch`)
- 1000 runs (`500 x 2`): hallucination flags `1.4% -> 0.2%`

with consistent tradeoffs in contradiction handling under `plus_scratch`.

New attribution series (2026-03-10) added:

1. `Qwen3-4B-Instruct-2507` raw (`500`)
2. `Qwen3-4B-Hivemind` raw (`500`)
3. `Qwen3-4B-Instruct-2507` routed (`500 no_scratch + 500 plus_scratch`)

Key findings:

- Raw baseline favored `Qwen3-4B-Instruct-2507` on hallucination flags (`0.8%` vs `1.8%`).
- In routed `Qwen3-4B-Instruct-2507`, `plus_scratch` did not improve hallucination flags (`0.2% no_scratch`, `0.4% plus_scratch`).
- Contradiction handling under `plus_scratch` remained weak (`detection 2.00 -> 0.00`) in routed batteries.

Interpretation: grounding uplift is real in some model/mode combinations, but not model-agnostic. Model selection materially affects baseline and routed outcomes.

Claims are explicitly bounded to this benchmark design and scoring rubric.

Blinded-rating note:

- A two-rater blinded adjudication protocol has been created but it was not executed for these reported runs.
- Claims in this draft therefore remain bounded to automated rubric/scoring artifacts.

## 1. Context and Scope

This project positions reliability as constrained truthfulness/faithfulness under grounded tasks, not universal open-domain intelligence.

This draft keeps that scope and adds attribution analysis:

- How much comes from base model choice?
- How much comes from routing + scratch grounding policy?

## 2. Evidence Pack Used

### 2.1 Prior routed batteries (Hivemind track)

- `prepub/VALIDATION_BATTERY_REPORT.md`
- `prepub/VALIDATION_BATTERY_REPORT2.md`
- `prepub/validation_battery_raw2.jsonl`
- `prepub/validation_battery_scored2.jsonl`
- `prepub/VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md`
- `prepub/validation_battery_raw2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_scored2_v2_1000_20260309T130027Z.jsonl`
- `prepub/validation_battery_meta_v2_1000_20260309T130027Z.json`
- `prepub/prompts_manifest2_v2_1000_20260309T130027Z.jsonl`

### 2.2 Attribution extension (2026-03-10 campaign)

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT.md`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_qwen2507_raw_1000_20260309T175805Z.jsonl` (500 runs)
- `TEST_ARTIFACTS_VALIDATION/validation_scored_qwen2507_raw_1000_20260309T175805Z.jsonl` (500 runs)
- `TEST_ARTIFACTS_VALIDATION/validation_raw_hivemind_raw_1000_20260309T191228Z.jsonl` (500 runs)
- `TEST_ARTIFACTS_VALIDATION/validation_scored_hivemind_raw_1000_20260309T191228Z.jsonl` (500 runs)
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_1000_20260309T232905Z.jsonl` (1000 runs)
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_qwen2507_routed_1000_20260309T232905Z.jsonl` (1000 runs)
- `TEST_ARTIFACTS_VALIDATION/VALIDATION_BATTERY_REPORT2_qwen2507_routed_1000_20260309T232905Z.md`
- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`

## 3. Experimental Setup

Common generation settings for all compared runs:

- temperature: `0.2`
- top_p: `0.9`
- max_tokens: `768`
- context target: `8192`

Additional run controls in attribution campaign:

- thermal guard: trigger `86C`, hard-stop `87C`, resume `82C`
- 10-minute cooldown between major blocks
- sequential execution (no parallel model servers)

## 4. Benchmark Dimensions

- `reversal`: re-adjudication under frame inversion.
- `tom`: perspective separation under role-conditioned prompts.
- `evidence`: label discipline (`VERIFIED`/`SUPPORTED`/`ASSERTED`) and over-upgrade suppression.
- `retraction`: correction handling after invalidating updates.
- `contradiction`: conflict detection + source priority + calibrated uncertainty.
- `negative_control`: refusal behavior under insufficient evidence.

## 5. Results

### 5.1 Prior routed results (Hivemind track)

Hallucination flags:

- 240 battery: `3.3% -> 0.0%` (`no_scratch -> plus_scratch`)
- 1000 battery: `1.4% -> 0.2%`

Refusal correctness:

- 240 battery: `1.60 -> 2.00`
- 1000 battery: `1.83 -> 2.00`

Tradeoff pattern:

- 240 battery: reversal hedging `0.15 -> 0.70`; contradiction uncertainty `1.90 -> 1.15`
- 1000 battery: contradiction detection `2.00 -> 0.00`; source prioritization `2.00 -> 1.00`; uncertainty `1.98 -> 1.00`
- evidence classification shifted from slight negative in 240 (`3.60 -> 3.45`) to positive in 1000 (`0.92 -> 2.34`)

Interpretation:

- routed grounding yielded strong hallucination suppression in this model track.
- gains were accompanied by measurable contradiction regressions.

### 5.2 Attribution extension (model-vs-harness)

Raw baselines:

- `Qwen3-4B-Instruct-2507` raw: `4/500` hallucination flags (`0.8%`)
- `Qwen3-4B-Hivemind` raw: `9/500` hallucination flags (`1.8%`)

Routed `Qwen3-4B-Instruct-2507` (1000 paired runs):

- `no_scratch`: `1/500` (`0.2%`)
- `plus_scratch`: `2/500` (`0.4%`)

Category-level note in routed Qwen2507 run:

- contradiction under `plus_scratch`: `detection 2.00 -> 0.00`

Interpretation:

- model choice materially shifts raw baseline reliability.
- scratch uplift was not reproduced in routed Qwen2507 run.
- contradiction regression under plus_scratch appears robust across routed tracks.

### 5.3 Operational profile (attribution extension)

From `INTERIM-REPORT.md`:

- Raw Qwen2507 block thermal p50/p95/max: `81/82/83 C`
- Raw Hivemind block thermal p50/p95/max: `81/82/83 C`
- Routed Qwen2507 block thermal p50/p95/max: `74/76/77 C`

Latency (routed report):

- no_scratch mean: `12.65s`
- plus_scratch mean: `10.61s`

Format retries:

- Raw blocks: `0`
- Routed Qwen2507 block: `249/1000` (`24.9%`)

Interpretation:

- no thermal penalty observed for routing in this run set.
- format-retry burden is a non-trivial routed-path operational cost.

## 6. Relation to arXiv:2603.08274

Reference: `https://arxiv.org/html/2603.08274`

Alignment with observed outcomes in this study:

1. Model selection matters:
   - Raw baselines differ by model (`0.8%` vs `1.8%` hallucination flags).
2. Grounding and fabrication are distinct:
   - Lower hallucination can coexist with weaker contradiction handling.
3. Claims must remain bounded:
   - Behavior is setup- and policy-dependent, not universal across all modes/models.

Non-equivalence caveat:

- The cited arXiv work emphasizes long-context fabrication regimes.
- This battery is constrained, rubric-scored, and mostly fixed-context task design.
- All reported runs in this draft were executed at `--ctx 8192`; external validity to `32K+` contexts is not established by this evidence pack.
- Qwen overlap does not remove this boundary: `Qwen3-4B-Instruct-2507` appears in both programs, but context regime and evaluation protocol differ.
- Therefore, the relationship is conceptual alignment, not direct benchmark comparability.

Decoupling interpretation for routed Qwen2507:

- The routed Qwen2507 run showed low hallucination rates but elevated format retries (`24.9%`) and weaker contradiction handling under `plus_scratch`.
- This is consistent with capability decoupling: grounding/refusal behavior can remain strong while contradiction adjudication and schema-conformant conflict reasoning degrade under strict policy constraints.
- In this framing, retry spikes are treated as a symptom of decoupled capability under constraint pressure, not random instability.
- This directly motivates contradiction-aware gating and dual-pass adjudication as principled mitigations rather than ad-hoc patching.

## 7. Statistical Bound (Zero-Event Caveat)

For the prior Hivemind 240 battery, grounded `0/120` implies a Rule-of-Three 95% upper bound of approximately:

- `3 / 120 ~= 2.5%`

For the prior Hivemind 1000 battery grounded result (`1/500`), observed rate is:

- `0.2%` (non-zero)

For the attribution extension raw blocks:

- Qwen2507 raw (`4/500`) and Hivemind raw (`9/500`) are low-rate observations, but not zero-event guarantees.

For expanded cliniko stress replay:

- `0/165` implies a Rule-of-Three upper bound of approximately `1.8%`.

These are bounded observations under this benchmark family, not universal guarantees.

## 8. Clinical Workflow Validation (Test-Case Evidence)

Separate from the contradiction/reversal battery, workflow-stability artifacts show:

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

- protocol: `>>cliniko auto` with pasted `TEST-*.txt` case -> `>>cliniko review` -> `>>flush` -> guard check (`>>cliniko review` fails loud with no scaffold)
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
- auto-header duplication fix: `avg 2.0 -> 1.0`
- stress-run review pass rate: `45/45` (`100.0%`)

Interpretation boundary:

- these are workflow consistency and structural validity signals
- these are not prospective clinical efficacy outcomes

Rule-of-Three notes for zero-event stress observations:

- legacy stress subset (`0/45`): 95% upper bound approximately `6.7%`
- expanded replay (`0/165`): 95% upper bound approximately `1.8%`

## 9. What This Does and Does Not Establish

Supported:

1. In this benchmark family, routed grounding can reduce hallucination flags in specific model/mode tracks.
2. Model choice is a first-order factor in raw and routed outcomes.
3. Contradiction-handling degradation under strict grounding is measurable and should be treated as an explicit tradeoff.
4. Clinical workflow stress tests remain structurally stable in the included protocol artifacts (`165/165` expanded replay).

Not established:

1. Universal scratch uplift across all compatible 4B models.
2. Universal contradiction robustness under strict grounding.
3. Open-domain or long-horizon guarantees.

## 10. Reproducibility Notes

This draft references raw, scored, report, and interim artifacts for:

- prior Hivemind routed batteries
- attribution raw baselines
- attribution routed Qwen2507 battery
- clinical workflow stress and replay artifacts

Next reproducibility step:

- publish a turnkey benchmark runner + rubric package for independent reruns with fixed settings and explicit model swaps.
- execute the two-rater blinded adjudication path and publish inter-rater agreement outputs before final publication.

## 11. Mechanism Hypothesis (Revised)

Observed behavior is consistent with interaction effects:

- Base model prior behavior sets the raw reliability floor.
- Grounding contracts reduce unconstrained generation pressure.
- The same constraints can reduce contradiction adjudication flexibility.
- Net effect depends on model-policy fit (not just policy design alone).

This reframes prior claims from "general grounding uplift" to "conditional uplift with model-dependent interaction."

## 12. Practical Next Steps

1. Add contradiction-aware policy gating:
   - route contradiction tasks to `no_scratch` or dual-pass adjudication by default.
2. Keep scratch default for evidence/refusal-heavy tasks where gains are consistent.
3. Track format-retry rate as a required operational KPI in all routed benchmarks.
4. Add long-context stress extension to align more directly with long-context fabrication literature.
5. Execute two-rater blinded adjudication and inter-rater agreement before final publication.

## 13. Conclusion

- evidence now shows more clearly that model choice and grounding policy interact.
- Prior Hivemind routed gains remain valid as bounded observations.
- New Qwen2507 routed evidence shows those gains do not automatically transfer.

The strongest defensible position is:

- reliability gains are real in specific constrained settings,
- tradeoffs are explicit and measurable,
- and cross-model generalization must be demonstrated, not assumed.
