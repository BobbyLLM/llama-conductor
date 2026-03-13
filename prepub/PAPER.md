# Preprint Draft v3: Reliability Evaluation of a Local 4B LLM with Post-Policy Replication

Status: pre-publication draft for peer-review preparation. Updated 2026-03-13 (WIP)

## Abstract

This preprint evaluates reliability behavior for a local 4B stack under fixed settings (`temperature=0.2`, `top_p=0.9`, `max_tokens=768`, `--ctx 8192`) using rubric-scored grounded tasks.

Benchmark evidence in this draft includes `5304` runs:

1. Routed Hivemind battery A: `120 prompts x 2 conditions = 240`
2. Routed Hivemind battery B: `500 prompts x 2 conditions = 1000`
3. Attribution block: `Qwen3-4B-Instruct-2507` raw (`500`)
4. Attribution block: `Qwen3-4B-Hivemind` raw (`500`)
5. Attribution block: routed `Qwen3-4B-Instruct-2507` (`1000`)
6. Post-policy resample campaign (`1400`)
7. Missing-lane closure: Qwen2507 routed (`332`)
8. Missing-lane closure: Hivemind routed (`332`)

Separate workflow-stability evidence adds `210` runs (`45 + 165`) under the clinical command-path protocol. This stream is out-of-band for Qwen3-4B attribution claims: `>>cliniko` generation is deterministic Python sidecar logic and `>>cliniko review` runs through `Qwen2.5-1.5B`.

Core findings:

- Prior routed Hivemind batteries still show bounded hallucination suppression (`3.3% -> 0.0%` in 240; `1.4% -> 0.2%` in 1000) with stronger refusal behavior.
- Prior routed Qwen2507 run showed a format-retry burden (`24.9%`) and weaker contradiction behavior under strict grounding.
- Post-policy routed resample removed the retry burden in tested tracks (`0` retries across `1200` routed runs).
- Missing-lane closure confirms contradiction stability in both models (`0` contradiction flags in both 332-run closures).
- Hivemind missing-lane run recorded `4/332` rubric hallucination flags, all in `negative_control` and all `no_scratch`, all consistent with lexical refusal misses rather than fabricated specifics.

Interpretation: reliability uplift is conditional on model-policy interaction, and scoring sensitivity (especially refusal lexicon) is a measurable factor. Claims remain bounded to this benchmark family and `8K` context.

Blinded-rating note:

- A two-rater blinded adjudication protocol has been created but was not executed for the reported runs.
- Claims in this draft remain bounded to automated rubric/scoring artifacts.

## 1. Context and Scope

This project targets constrained truthfulness/faithfulness under grounded tasks, not universal open-domain intelligence.

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

### 2.2 Attribution + post-policy extension

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT.md`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-postpolicy-1400-20260312T150117Z.md`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_qwen2507_raw_1000_20260309T175805Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_raw_hivemind_raw_1000_20260309T191228Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_1000_20260309T232905Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_600_20260312T160431Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_600_20260312T183751Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-missing-lanes-332-20260313T023647Z.md`
- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-missing-lanes-332-hivemind-20260313T072632Z.md`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_80_20260313T023703Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_252_20260313T030405Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_80_20260313T072645Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_qwen2507_routed_252_20260313T075523Z.jsonl`

### 2.3 Clinical workflow stream (separate model path)

- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`

Boundary note:

- `>>cliniko` is deterministic Python sidecar workflow.
- `>>cliniko review` uses `Qwen2.5-1.5B`.
- This stream validates command-path/workflow stability, not Qwen3-4B scratch effect size.

## 3. Experimental Setup

Common settings for compared runs:

- temperature: `0.2`
- top_p: `0.9`
- max_tokens: `768`
- context target: `8192`

Operational controls in extension campaigns:

- thermal guard: trigger `86C`, hard-stop `87C`, resume `82C`
- 10-minute cooldown between major blocks
- sequential execution (single active model server path)

## 4. Benchmark Dimensions

- `reversal`: re-adjudication under frame inversion
- `tom`: perspective separation under role-conditioned prompts
- `evidence`: label discipline and over-upgrade suppression
- `retraction`: correction handling after invalidating updates
- `contradiction`: conflict detection + source priority + calibrated uncertainty
- `negative_control`: refusal behavior under insufficient evidence

## 5. Results

### 5.1 Prior routed Hivemind results (baseline and replication)

- 240 battery hallucination flags: `3.3% -> 0.0%`
- 1000 battery hallucination flags: `1.4% -> 0.2%`
- refusal correctness: `1.60 -> 2.00` (240), `1.83 -> 2.00` (1000)
- contradiction metrics regressed under strict grounding in prior batteries:
  - detection `2.00 -> 0.00`
  - source prioritization `2.00 -> 1.00`
  - uncertainty `1.98 -> 1.00`

### 5.2 Attribution extension (pre-policy)

Raw baselines:

- Qwen2507 raw: `4/500` (`0.8%`)
- Hivemind raw: `9/500` (`1.8%`)

Routed Qwen2507 (`1000` paired runs):

- no_scratch: `1/500` (`0.2%`)
- plus_scratch: `2/500` (`0.4%`)
- format retries: `249/1000` (`24.9%`)

Interpretation:

- model choice materially shifts raw baseline reliability
- policy-task interaction materially shifts routed behavior

### 5.3 Post-policy resample (`1400`)

Completed blocks:

- Qwen2507 raw (`100`): hallucination `0/100`, retries `0`
- Hivemind raw (`100`): hallucination `0/100`, retries `0`
- routed Hivemind (`600`): hallucination `0/600`, retries `0`
- routed Qwen2507 (`600`): hallucination `0/600`, retries `0`

Important composition note:

- the two `600` routed blocks covered `reversal/tom/evidence/retraction`
- `contradiction/negative_control` were not included in those `600` slices due to prompt-selection ordering
- those lanes were closed explicitly in the missing-lane campaign below

### 5.4 Missing-lane closure (`332 + 332`)

Qwen2507 routed missing lanes:

- canary `80` + top-up `252` = `332`
- category coverage: contradiction `166`, negative_control `166`
- hallucination flags: `0/332`
- format retries: `0/332`

Hivemind routed missing lanes:

- canary `80` + top-up `252` = `332`
- category coverage: contradiction `166`, negative_control `166`
- hallucination flags: `4/332` (`1.20%`)
- format retries: `0/332`
- flag distribution:
  - contradiction: `0/166`
  - negative_control: `4/166` (all in `no_scratch`)

### 5.5 Interpreting the `4/332` Hivemind flags

Artifact inspection of all four flagged rows shows:

- no invented causal details
- no invented thresholds
- refusal-style responses using phrasing such as:
  - "mechanism is not described"
  - "does not confirm or deny"

Current negative-control rubric marks hallucination when refusal lexicon does not include one of a strict token set (for example `insufficient`, `cannot`, `not enough`, `undocumented`).  
Therefore these four flags are best interpreted as **rubric lexical refusal misses**, not clear factual fabrication events.

This distinction is reported explicitly; raw artifacts and rubric outputs are preserved unchanged.

## 6. Operational Profile

From extension campaigns:

- post-policy routed tracks ran at thermal p50/p95/max around `75-77C`
- no routed format retries were observed in post-policy routed tracks (`0/1200`) and missing-lane closures (`0/664`)
- this contrasts with pre-policy routed Qwen2507 (`249/1000` retries)

Interpretation:

- major retry burden observed pre-policy did not reproduce in post-policy reruns
- residual flagged events were concentrated in negative-control lexical refusal scoring (Hivemind no_scratch), not contradiction output collapse

## 7. Relation to arXiv:2603.08274

Reference: `https://arxiv.org/html/2603.08274`

Alignment:

1. Model selection matters.
2. Grounding and fabrication resistance are partially decoupled capabilities.
3. Bounded claims are required; context regime and protocol differences matter.

Non-equivalence caveat:

- this benchmark is at `--ctx 8192`
- direct equivalence to `32K+` long-context studies is not established by this evidence pack

## 8. Clinical Workflow Validation Stream (Separate Path)

Combined clinical workflow stream (`45 + 165 = 210`) remains structurally stable:

- pass/fail stability in reported command-path protocol
- off-target term runs: `0`
- expanded replay flush-guard failures: `0`

Rule-of-Three for combined zero-event stream:

- `0/210` implies 95% upper bound approximately `1.43%`

Boundary:

- this stream is operational/workflow evidence, not Qwen3-4B scratch-attribution evidence

## 9. What This Supports and What It Does Not

Supported:

1. Prior routed Hivemind batteries show bounded hallucination suppression with tradeoffs.
2. Model-policy interaction is first-order; outcomes are not model-agnostic.
3. Post-policy runs materially improved routed operational stability (retry collapse).
4. Missing contradiction/negative-control lanes are now explicitly covered for both model paths.
5. Hivemind residual flags in missing lanes are currently concentrated in lexical refusal scoring, not contradiction fabrication.

Not established:

1. Universal uplift across all model families and prompts.
2. Universal contradiction robustness under strict grounding.
3. Long-context (`32K+`) external validity from this pack alone.
4. Human-adjudicated clinical efficacy claims without blinded rater execution.

## 10. Reproducibility Notes

This draft references raw, scored, report, and interim artifacts for:

- prior routed batteries
- attribution extension
- post-policy rerun
- missing-lane closure for both model paths
- separate clinical workflow stream

Next steps before final publication:

1. execute two-rater blinded adjudication and report inter-rater agreement
2. publish turnkey runner + rubric package for independent reruns
3. refine negative-control refusal rubric to reduce lexical false positives while preserving fail-loud behavior

## 11. Conclusion

The strongest defensible position after the full post-policy and missing-lane closure set is:

- reliability gains are real but conditional
- tradeoffs are explicit and measurable
- operational costs can be reduced materially with policy-level fixes
- scoring design (especially refusal lexicon) must be treated as part of the reliability system, not an afterthought

This project should be interpreted as a bounded reliability harness with explicit failure modes and auditable artifacts, not as a universal model-agnostic guarantee.

