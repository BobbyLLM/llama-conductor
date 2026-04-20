# Policy-Constrained Output Control in Local LLM Orchestration: Bounded Reliability Evidence Across 8,764 Runs

Status: automated-rubric evidence pack, pre-publication draft. Two-rater blinded adjudication is declared future work (see Section 4.1). Updated 2026-03-26.

---

## Abstract

This paper reports bounded reliability testing of a local 4B-parameter router-grounded stack at fixed settings (`temperature=0.2`, `top_p=0.9`, `max_tokens=768`, `--ctx 8192`) across raw and routed conditions, evaluated over `8764` benchmark runs spanning five model families.

The central claim is a **systems control claim**: routing policy and lane contracts can drive rubric-flag rates to floor-level outcomes under bounded task conditions. This is not a claim about intrinsic model truthfulness, nor is it a claim that the rubric constitutes ground-truth hallucination detection. It is a claim that architectural control layers — contract-bounded generation, explicit grounding constraints, and fail-loud behavior — can dominate observed output quality for small local models under the conditions tested.

Reported outcomes are automated-rubric outputs. Single-rater adjudication of all flagged rows plus a stratified unflagged sample is planned prior to formal submission; full two-rater replication is future work.

Benchmark runs analyzed: `8764`.

- Legacy + attribution + post-policy core corpus: `5304`
- Cross-family expansion campaign: `3300`
- Post-patch targeted revalidation: `160`

Separate workflow-stability evidence adds `210` runs (`45 + 165`) and is reported as an out-of-band stream.

Main findings:

1. Post-policy routed Qwen2507 and Hivemind reached floor-level rubric-flag outcomes (`0/1864` across all six task categories) with zero format retries, compared to `24.9%` format retries under pre-policy routing. This improvement was achieved through policy/contract changes alone, with no model retraining or fine-tuning.
2. Cross-family expansion across three additional model families (Granite, Phi-4-mini, SmolLM3) demonstrated that the routing layer generalizes beyond the models it was developed against, while also revealing model-policy interaction effects that vary by family.
3. The harness detected, classified, and enabled correction of a lane-quality degradation in SmolLM3 (`78/1000` flags concentrated in contradiction/reversal under plus_scratch). A surgical lane patch resolved all affected categories (`160/160` clean in targeted revalidation). This detect-classify-correct cycle is itself evidence of system observability.
4. All rubric-flagged rows across campaigns were taxonomically classified. No row was promoted to confirmed fabrication. The dominant failure families — `lexical/contract` compliance and `rubric-coupling` anomalies — are contract-enforcement gaps, not model-generated confabulation.

Interpretation: this evidence supports a bounded system-level control claim — policy/routing can dominate observed rubric-flag outcomes in tested tasks under this harness — while also demonstrating that failures are auditable, classifiable, and correctable without model-level intervention.

---

## 1. Scope and Protocol

The target claim is **behavioral consistency under routing policy in bounded, grounded tasks**. This is not a claim about universal open-domain reasoning or intrinsic model reliability.

Common run settings:

- temperature `0.2`
- top_p `0.9`
- max_tokens `768`
- context target `8192`

Operational controls in extension campaigns:

- thermal guard `86°C` trigger / `87°C` hard / `82°C` resume
- 10-minute inter-block cooldown
- sequential execution (single active serving path)

### 1.1 Runtime vs Evaluation Responsibilities

| Component | Runtime system | Evaluation harness |
|---|---|---|
| Lane selection | yes | no |
| Contract/policy enforcement | yes | no |
| Deterministic fail-loud behavior | yes | no |
| Format retry | no (handled by lane logic) | yes (single bounded retry in battery runner) |
| Rubric scoring | no | yes |
| Rubric-flag assignment | no | yes |
| Taxonomy aggregation/reporting | no | yes |

### 1.2 Contributions

This work makes three claims, each supported by the evidence in Sections 3–4:

1. **Policy dominance over observed output quality.** Under bounded tasks with fixed runtime settings, routing policy and lane contracts can suppress rubric-flag rates to floor across multiple model families. Post-policy routed slices for Qwen2507 and Hivemind achieved `0/1864` flags with `0` format retries. This is a systems-engineering result: the control layer, not model scale, is the dominant factor in observed output quality under these conditions.

2. **Cross-family generalization with characterized interaction effects.** The routing layer was developed against Qwen2507 and Hivemind, then tested on Granite, Phi-4-mini, and SmolLM3 without per-model tuning. Granite and Phi showed low flag rates (`11/1000` and `46/1000` respectively, dominated by contract-compliance gaps rather than confabulation). SmolLM3 showed a pronounced model-policy interaction (`78/1000` flags, `75/78` under plus_scratch), demonstrating that policy generalization is not uniform and that specific model-policy pairings can degrade in identifiable lanes.

3. **System observability: detection, classification, and correction of failure modes.** The harness taxonomy classified all flagged rows into four families (`runtime/transport`, `lexical/contract`, `rubric-coupling`, `lane-quality degradation`) with no row promoted to confirmed fabrication. The SmolLM3 lane-quality degradation was detected by the rubric, localized by taxonomy to specific lanes and conditions, and resolved by a surgical patch to three files (`chat_postprocess.py`, `chat_finalize.py`, `router_fastapi.py`), yielding `160/160` clean in targeted revalidation. This closed-loop correctability — without model retraining — is a practical contribution for local LLM deployment.

---

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

- `reversal`: initial claim adjudication, then explicit inversion of key premise. Expected: revise answer and state why prior conclusion no longer holds.
- `tom`: multiple actors with different belief/knowledge states. Expected: keep role-conditioned beliefs separate without leakage.
- `evidence`: mixed support strength from provided context. Expected: preserve label discipline (`VERIFIED`/`SUPPORTED`/`ASSERTED`) without unsupported upgrades.
- `retraction`: follow-up correction invalidates earlier assumption. Expected: update answer state and retire prior invalid claim.
- `contradiction`: conflicting statements/sources within bounded context. Expected: detect conflict, prioritize source, express uncertainty when unresolved.
- `negative_control`: insufficient support by design. Expected: explicit refusal/insufficient-evidence response, no fabrication.

### 2.2 Worked Example (Artifact-Grounded)

Example category: `negative_control` (from errata-reviewed rows).

- Prompt intent: ask for causal mechanism/threshold not present in provided evidence.
- Expected behavior: explicit insufficient-evidence refusal.
- Observed response family (flagged rows in pre-fix runs): short scratch acknowledgement/refusal-like phrasing without strict refusal contract tokenization.
- Rubric outcome: `rubric_flag=1` with `refusal_correctness=0`.
- Taxonomy classification: `lexical/contract` (not auto-promoted to confirmed fabrication without manual adjudication).

Reference: `prepub/ERRATA.md`

---

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
        resp2 = call_router(item, condition)  # single bounded retry
        resp = resp2
    else:
        resp = resp1
    scores = score_item(item, resp)  # includes rubric_flag
    return scores
```

### 3.1 Legacy Hivemind Routed Batteries

- 240 battery rubric-flag rate: `3.3% -> 0.0%`
- 1000 battery rubric-flag rate: `1.4% -> 0.2%`
- refusal correctness: `1.60 -> 2.00` (240), `1.83 -> 2.00` (1000)

Legacy tradeoff signal under plus_scratch:

- contradiction detection `2.00 -> 0.00`
- contradiction source prioritization `2.00 -> 1.00`
- contradiction uncertainty `1.98 -> 1.00`

These contradiction tradeoffs motivated the policy revisions tested in subsequent campaigns.

### 3.2 Pre-Policy Attribution (Qwen/Hivemind)

Raw baselines:

- Qwen2507 raw: `4/500` (`0.8%`)
- Hivemind raw: `9/500` (`1.8%`)

Routed Qwen2507 (`1000` paired):

- no_scratch: `1/500` (`0.2%`)
- plus_scratch: `2/500` (`0.4%`)
- format retries: `249/1000` (`24.9%`)

The `24.9%` format-retry rate, while not affecting final rubric-flag counts, indicated that the routing layer was producing contract-valid outputs only after substantial harness-level correction. This motivated the policy revisions in Section 3.3.

### 3.3 Post-Policy Core Reruns

Completed blocks:

- Qwen2507 raw `100`: `0/100` rubric-flag count
- Hivemind raw `100`: `0/100` rubric-flag count
- routed Hivemind `600`: `0/600` rubric-flag count, `0` retries
- routed Qwen2507 `600`: `0/600` rubric-flag count, `0` retries

Coverage note: the `600 + 600` routed slices covered `reversal/tom/evidence/retraction`; contradiction and negative_control were closed in missing-lane runs.

The format-retry rate dropped from `24.9%` (pre-policy, Section 3.2) to `0%` across `1200` post-policy routed runs. This delta is evidence that the policy changes improved generation-time contract compliance, not merely rubric-level scoring.

### 3.4 Missing-Lane Closures (Qwen2507/Hivemind)

Each closure run targets only `contradiction + negative_control` with paired no_scratch/plus_scratch runs.

- routed Qwen2507 closure (`332`): rubric-flag count `0/332`, format retries `0/332`
- routed Hivemind closure (`332`): rubric-flag count `0/332`, format retries `0/332`

Combined with Section 3.3, this completes full six-category coverage for both core models under post-policy routing: `0/1864` flags, `0/1864` retries.

### 3.5 Cross-Family Expansion Campaign (`3300`)

The routing layer was developed against Qwen2507 and Hivemind. This campaign tests generalization to three model families the routing policy was not tuned for.

#### granite-4.0-micro-ablit

- raw `100`: rubric-flag count `0`, errors `0`
- routed `1000`: rubric-flag count `11`, errors `0`, format retries `1`
- routed category split: contradiction `1`, negative_control `10`
- Taxonomy: `10/11` flags are `lexical/contract` (negative-control refusal phrasing that did not satisfy strict contract tokenization); `1/11` is `rubric-coupling` (contradiction subscore pattern). None are confabulation.

#### Phi-4-mini

- raw `100`: rubric-flag count `2`, errors `2` (read timeouts)
- routed `1000`: rubric-flag count `46`, errors `0`, format retries `173`
- routed category split: contradiction `2`, negative_control `44`
- Taxonomy: `44/46` flags are `lexical/contract` (same negative-control pattern as Granite); `2/46` are `rubric-coupling`. None are confabulation. The `173` format retries indicate Phi-4-mini's generation is less format-compliant under these contracts than Granite, but the retry mechanism contained the impact.

#### SmolLM3

- raw `100`: rubric-flag count `0`, errors `0`
- routed `1000`: rubric-flag count `78`, errors `0`, format retries `707`
- routed condition split: `no_scratch 3/500`, `plus_scratch 75/500`
- routed category split: contradiction `52`, reversal `17`, negative_control `8`, evidence `1`
- Taxonomy: dominated by `lane-quality degradation` — the model's outputs under plus_scratch in contradiction and reversal lanes degraded in ways the contracts did not catch at generation time but the rubric detected post-hoc. The `25:1` condition split (`75` vs `3`) is the strongest model-policy interaction signal in the campaign. The `707/1000` format-retry rate further indicates poor model-policy fit under these contracts.

This result is informative precisely because it failed: the harness detected the degradation, the taxonomy localized it, and the subsequent patch (Section 3.6) resolved it — all without model retraining.

### 3.6 Post-Patch Targeted Revalidation (`160`)

A surgical sandbox patch was applied to lane contracts and strict output handling for affected categories. Patch scope was limited to three files: `chat_postprocess.py`, `chat_finalize.py`, `router_fastapi.py`.

- Granite fixcheck `40`: flags `0`, errors `0`
- Phi fixcheck `40`: flags `0`, errors `0`
- SmolLM3 fixcheck `80`: flags `0`, errors `0`
- aggregate: `160/160` clean

This is targeted affected-lane validation, not a full replacement rerun of all three routed `1000` blocks. Full replacement reruns are planned but not yet completed. The result demonstrates that the identified failure modes were attributable to contract/policy gaps and were correctable at the routing layer.

---

## 4. Error Taxonomy (Consolidated)

Final campaign classification (see `prepub/ERRATA.md` for artifact-level detail):

- `runtime/transport`: routed blocks `0`; observed only in Phi raw (`2` read timeouts)
- `lexical/contract`: dominant in Granite/Phi negative-control failures — the model produced semantically correct refusals that did not match strict contract tokenization requirements
- `rubric-coupling`: contradiction rows with strong subscores but conservative rubric-flag coupling — these may represent rubric conservatism rather than true output failure
- `lane-quality degradation`: dominant in SmolLM3 routed plus-scratch contradiction/reversal lanes — genuine model-policy interaction failures detected by the rubric

No row was promoted to confirmed fabrication without manual adjudication in this audit pass. The absence of confirmed fabrication across `~135` flagged rows is consistent with the system design: contract-bounded generation with fail-loud behavior is intended to produce contract-compliance failures (detectable, correctable) rather than silent confabulation.

### 4.1 Rubric Scoring Protocol and Adjudication Scope

**This draft reports automated-rubric outputs.** The rubric assigns a `rubric_flag` based on contract compliance, label discipline, and category subscores. A `rubric_flag=1` is a rubric output, not an automatic determination of fabrication.

Rubric inputs (per run):

- prompt metadata: model, condition, lane/category, run id
- model output payload (contracted response fields)
- parser/validator outcome (schema/header/label compliance)
- scorer fields (category subscores + final `rubric_flag`)

Decision flow (validation harness):

1. Evaluate transport/runtime status.
2. Validate contract compliance (required labels/headers/schema).
3. Compute rubric category subscores from final response.
4. Assign preliminary `rubric_flag` from rubric output.
5. Map flagged rows into taxonomy (`runtime/transport`, `lexical/contract`, `rubric-coupling`, `lane-quality`, or `confirmed fabrication` after adjudication).

**Planned adjudication (pre-submission):** Single-rater structured adjudication will be conducted on (a) all rubric-flagged rows across campaigns (~135 flagged rows in finalized routed blocks after taxonomy preclassification) and (b) a stratified random sample of `n=100` unflagged rows to establish a false-negative floor estimate. The adjudication protocol will be disclosed in full. Inter-rater reliability caveat will be explicit: single-rater adjudication is acknowledged as a limitation; full two-rater blinded replication is declared future work.

Interpretation rule: reported rubric-flag rows are harness outputs, not automatic proof of fabrication. The central claim does not require them to be — it requires that the system behave consistently under its own contracts, which the data supports.

### 4.2 Why Zero Observed Rubric-Flag Is Plausible Under This Harness

Zero-flag slices are plausible in this setup because generation is constrained by:

- lane contracts that narrow admissible output states
- bounded grounding policy that suppresses unsupported free generation
- fail-loud retry logic that rejects contract-violating responses
- narrow task scope and fixed runtime settings (8K context, low temperature)

Zero-flag outcomes are therefore interpreted as bounded protocol behavior under this harness, not a universal claim of intrinsic model truthfulness.

---

## 5. Interpretation

The central claim is a **systems control claim**: routing policy and lane contracts can drive rubric-flag rates to floor-level outcomes under bounded task conditions. The evidence does not require the rubric to be ground truth; it requires the system to behave consistently under its own contracts, which the data supports across five model families and `8764` runs.

### 5.1 What the Evidence Supports

1. **Policy dominance is empirically demonstrated.** Post-policy routed Qwen2507 and Hivemind achieved `0/1864` rubric flags with `0` format retries across all six task categories. The pre-policy to post-policy transition — from `24.9%` format retries to `0%`, and from nonzero flags to floor — was achieved through routing-layer changes alone. This is direct evidence that the control layer, not the base model, is the dominant factor in observed output quality under these conditions.

2. **Cross-family generalization holds with caveats.** Granite (`11/1000`) and Phi (`46/1000`) showed low flag rates under routing developed for different models, with flags concentrated in contract-compliance gaps rather than confabulation. SmolLM3 (`78/1000`) showed that generalization is not automatic — specific model-policy pairings can degrade in identifiable lanes.

3. **Failures are observable, classifiable, and correctable.** The SmolLM3 result demonstrates a complete detect-classify-correct cycle: the rubric detected degradation, the taxonomy localized it to specific lanes and conditions, and a surgical patch resolved it (`160/160` clean). This closed-loop property — where system failures are handled at the routing layer rather than requiring model retraining — is a practical requirement for local deployment.

4. **The failure taxonomy is informative about system design.** The dominance of `lexical/contract` failures (model produced correct behavior but wrong format) over confabulation suggests that the primary reliability bottleneck under this architecture is contract specification, not model capability. This has implications for where engineering effort should be directed.

### 5.2 What the Evidence Does Not Support

- A claim of intrinsic model truthfulness or universal hallucination suppression.
- Generalization beyond 8K context or beyond the six task categories tested.
- That the rubric constitutes ground-truth hallucination detection.
- That SmolLM3-class models are unsuitable — only that they require per-model lane hardening that Qwen2507 and Hivemind did not need under the tested contracts.

---

## 6. Limitations

- Context regime is `8K`; no direct external validity to `32K+` from this evidence pack.
- Reported outcomes are automated-rubric outputs. Single-rater adjudication on flagged rows and stratified unflagged sample is planned pre-submission; two-rater replication is future work.
- Post-patch evidence uses targeted reruns (`160`), not full `3 x 1000` replacement runs.
- Results are benchmark-bounded and should not be generalized to all prompt classes.
- Rubric design, task design, and evaluation harness are all first-party; no external benchmark comparison is included in this evidence pack.
- The routing layer was developed against Qwen2507 and Hivemind; cross-family results reflect zero-shot policy transfer, not per-model optimization. Better results for Granite/Phi/SmolLM3 may be achievable with family-specific tuning, but this is not tested.

---

## 7. Separate Clinical Workflow Stream (Out-of-Band)

Workflow-stability stream totals `210` runs (`45 + 165`) with published artifacts.

Boundary condition:

- `>>cliniko` generation is deterministic Python sidecar logic.
- `>>cliniko review` uses `Qwen2.5-1.5B`.
- This stream is operational workflow evidence, not Qwen3-4B scratch attribution evidence.

---

## 8. Conclusion

This project demonstrates that **architectural control layers can make small local models behave reliably under bounded tasks, without model scaling or retraining**.

Across `8764` benchmark runs spanning five model families, the evidence supports three results. First, routing policy and lane contracts drive observed rubric-flag rates to floor in well-fitted model-policy pairings (`0/1864` for post-policy Qwen2507/Hivemind, `0` format retries). Second, the routing layer generalizes to unseen model families with varying degrees of fit — from near-floor (Granite `11/1000`) to pronounced lane-quality degradation (SmolLM3 `78/1000` under plus_scratch) — and the variation is itself informative about model-policy interaction. Third, when failures occur, they are detectable by the harness, classifiable by taxonomy, and correctable by surgical policy changes without model retraining.

The claim is not that the models do not hallucinate. The claim is that the harness can control what they output under the conditions it was designed to enforce — and that when control fails, it fails observably and correctably.

---

## 9. Evidence Pack (Backmatter)

### 9.1 Legacy Routed Batteries and Attribution Core

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

### 9.2 Missing-Lane Closures

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-missing-lanes-332-hivemind-20260313T091848Z.md`
- `TEST_ARTIFACTS_VALIDATION/missing_lanes_332_hivemind_meta_20260313T091848Z.json`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_80_20260313T091901Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw2_hivemind_routed_252_20260313T094659Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_hivemind_routed_252_20260313T094659Z.jsonl`

### 9.3 Cross-Family Campaign (`3300`)

- `TEST_ARTIFACTS_VALIDATION/INTERIM-REPORT-campaign-3300-queue.md`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_granite40_micro_ablit_routed_1000_20260313T114724Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_phi4_mini_routed_1000_20260313T161354Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored2_smollm3_routed_1000_20260313T203358Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_granite40_micro_ablit_raw_100_20260313T112451Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_phi4_mini_raw_100_20260313T145626Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/validation_scored_smollm3_raw_100_20260313T200801Z.jsonl`

### 9.4 Post-Patch Targeted Validation

- `TEST_ARTIFACTS_VALIDATION/fixval_scored_granite_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_phi_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_smol_fixcheck_20260314T084930Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_summary_20260314T084930Z.json`
- `prepub/ERRATA.md`

### 9.5 Clinical Workflow Stream

- `prepub/_STRESS-CLINIKO-REVIEW-v1.md`
- `prepub/GPT-AUTO-TEST-v3.md`
- `prepub/cliniko_stress_replay_expanded_20260308T194459Z.json`
- `prepub/cliniko_live_sweep_20260226T110651Z.json`
- `prepub/cliniko_live_ankle_random_battery_20260228T141106Z.json`

---

## 10. Consolidated Campaign Summary (Explicit Ledger)

### 10.1 Total Run Accounting

- Core benchmark corpus before latest expansion: `5304`
- Cross-family expansion campaign: `3300` (completed)
- Post-patch targeted validation: `160` (completed)
- Core benchmark total: `8764`
- Separate workflow stream (out-of-band): `210` (`45 + 165`)
- Grand total including workflow stream: `8974`

### 10.2 Legacy Routed Hivemind Batteries

- `240` battery: rubric-flag rate `3.3% -> 0.0%` (`no_scratch -> plus_scratch`)
- `1000` battery: rubric-flag rate `1.4% -> 0.2%`
- Tradeoff signal: contradiction handling weakened under plus-scratch in legacy phase.

### 10.3 Pre-Policy Attribution Block (Qwen/Hivemind)

- Raw baselines: Qwen2507 raw `4/500` (`0.8%`), Hivemind raw `9/500` (`1.8%`)
- Routed Qwen2507 (`1000`): no_scratch `1/500` (`0.2%`), plus_scratch `2/500` (`0.4%`), format retries `249/1000` (`24.9%`)

### 10.4 Post-Policy Resample + Missing-Lane Closures

- Post-policy resample (`1400`): Qwen2507 raw `0/100`, Hivemind raw `0/100`, routed Hivemind `0/600` (retries `0`), routed Qwen2507 `0/600` (retries `0`)
- Missing-lane closures: routed Qwen2507 `0/332` (retries `0`), routed Hivemind `0/332` (retries `0`)
- Combined post-policy routed total: `0/1864` flags, `0/1864` retries

### 10.5 Cross-Family Expansion Campaign (`3300`)

#### Granite-4.0-micro-ablit
- Raw `100`: flags `0`, errors `0`
- Routed `1000`: flags `11`, errors `0`, retries `1`
- Flags: negative_control (lexical/contract) `10`, contradiction (rubric-coupling) `1`

#### Phi-4-mini
- Raw `100`: flags `2`, errors `2` (transport timeouts)
- Routed `1000`: flags `46`, errors `0`, retries `173`
- Flags: negative_control (lexical/contract) `44`, contradiction (rubric-coupling) `2`

#### SmolLM3
- Raw `100`: flags `0`, errors `0`
- Routed `1000`: flags `78`, errors `0`, retries `707`
- Condition split: no_scratch `3/500`, plus_scratch `75/500`
- Category split: contradiction `52`, reversal `17`, negative_control `8`, evidence `1`

### 10.6 Errata Taxonomy

- `runtime/transport`: `2` observed (Phi raw only); none in finalized routed blocks
- `lexical/contract`: major family in Granite/Phi negative-control rows
- `rubric-coupling`: contradiction anomalies with strong subscores but conservative flag coupling
- `lane-quality degradation`: dominant in SmolLM3 routed plus-scratch contradiction/reversal lanes
- `confirmed fabrication`: none promoted without manual adjudication

### 10.7 Surgical Patch + Targeted Revalidation

- Patch scope: `chat_postprocess.py`, `chat_finalize.py`, `router_fastapi.py`
- Targeted revalidation (`160`): Granite `40/40` clean, Phi `40/40` clean, SmolLM3 `80/80` clean

### 10.8 What This Shows

- Policy dominance: post-policy Qwen2507/Hivemind achieved `0/1864` flags with `0` retries across all six task categories.
- Cross-family transfer: routing layer generalized to three unseen model families with varying fit.
- System observability: failures were detected, classified, and corrected at the routing layer without model retraining.
- Failure characterization: dominant failure families are contract-compliance gaps, not confabulation.

### 10.9 What Is Still Bounded

- No universal claim across all tasks or context lengths.
- Long-context (`>8K`) generalization is not established by this evidence pack.
- Full post-patch `3 x 1000` replacement rerun is not yet done; current post-patch evidence is targeted (`160`) for engineering validation.
- Rubric, task design, and harness are all first-party; no external benchmark comparison is included.
- Adjudication is automated-rubric only in this draft; single-rater adjudication on flagged rows and stratified unflagged sample is planned pre-submission.
- Cross-family results reflect zero-shot policy transfer; per-model optimization is untested and may improve outcomes.
