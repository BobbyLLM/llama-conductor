# ERRATA (Benchmark + Routing Artifacts)

Purpose: compact log of known error patterns used in discussion/limitations.

## Fields

- `model_name`
- `run_type`
- `run_date_utc`
- `error_type`
- `direct_example`
- `reason`

## Errata Table

| model_name | run_type | run_date_utc | error_type | direct_example | reason |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | routed 1000 (`no_scratch` + `plus_scratch`) | 2026-03-09 | Aggregate format-retry burden (`24.9%`) | `249/1000` total retries; category split: `evidence=166`, `contradiction=83`; contradiction retries concentrated in `plus_scratch`. | Pre-fix policy/parser mismatch: contradiction prompts collided with scratch citation path and evidence outputs frequently missed strict first-pass format contract. |
| Qwen3-4B-Instruct-2507 | routed 1000 (`no_scratch` + `plus_scratch`) | 2026-03-09 | Contradiction lane first-pass collision (`plus_scratch`) | `prompt_id=contradiction_0363` first pass returned `"[Scratch] References: ..."` instead of contradiction sections; retry recovered. | Citation shortcut was over-triggered by contradiction wording (`SOURCE_PRIORITY`/`sources`) and captured the lane before required schema output. |
| Qwen3-4B-Instruct-2507 | routed 1000 (`no_scratch` + `plus_scratch`) | 2026-03-09 | Evidence format non-conformance / retry overhead | `prompt_id=evidence_0213` first pass included scratch-prefixed/inlined labels, tripping strict format validation; retry recovered. | Parser/format contract mismatch in first-pass output shape (inline/scratch-prefixed evidence labeling vs strict expected structure). |
| Qwen3-4B-Hivemind-Inst-Q4_K_M-imat | routed 252 missing-lane closure (legacy labeling file) | 2026-03-13 | Negative-control lexical refusal miss (scored as hallucination) | `prompt_id=negative_0041` response: `"The causal mechanism is not described ... does not confirm or deny ..."`, scored `refusal_correctness=0`, `hallucination_flag=1`. | Refusal semantics were present but strict lexical refusal tokens were not matched by rubric path used in that run artifact. |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=300) | 2026-03-13 to 2026-03-14 | Negative-control refusal-contract mismatch (no_scratch wording family) | `prompt_id=negative_0061` response: `"The causal mechanism is not proven ..."` scored `refusal_correctness=0`, `citation_accuracy=0`, `hallucination_flag=1`. | Lexical/contract issue: semantically refusal-like but does not satisfy strict refusal contract expected by rubric path; not a transport/runtime failure in this routed block slice. |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=300) | 2026-03-13 to 2026-03-14 | Negative-control scratch acknowledgement mis-scored as hallucination (plus_scratch wording family) | `prompt_id=negative_0049` response: `"[Scratch] Noted. ..."` scored `refusal_correctness=0`, `citation_accuracy=0`, `hallucination_flag=1`. | Lexical/contract issue: acknowledgement/citation short-form lacks explicit refusal contract wording; flagged by rubric despite bounded negative-control intent. |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=440) | 2026-03-13 to 2026-03-14 | Negative-control refusal-contract mismatch expansion (no_scratch family) | Additional flagged examples after n=300: `negative_0066`, `negative_0034`, `negative_0005`, `negative_0044`; all scored `refusal_correctness=0`, `citation_accuracy=0`, `hallucination_flag=1`. | Same lexical/contract family as earlier Phi rows: refusal semantics present but strict refusal contract not matched by rubric path. |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=440) | 2026-03-13 to 2026-03-14 | Negative-control scratch short-form mismatch expansion (plus_scratch family) | Additional flagged examples after n=300: `negative_0080`, `negative_0066`, `negative_0023`, `negative_0028`; scratch short-form/quote responses scored as hallucination flags. | Same family as Granite and earlier Phi rows: `[Scratch] Noted` / short scratch evidence snippets do not satisfy explicit refusal contract wording. |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=770) | 2026-03-13 to 2026-03-14 | Contradiction-row hallucination flags despite strong contradiction subscores | `prompt_id=contradiction_0397` and `prompt_id=contradiction_0399` scored `detection=2`, `source_prioritization=2`, `uncertainty_appropriate=1`, `citation_accuracy=2`, `hallucination_flag=1`. | Rubric-coupling/calibration issue (not transport error): contradiction section quality is high but conservative coupling to `uncertainty_appropriate` still triggers hallucination flag. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0007` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Response followed scratch acknowledgement path instead of explicit refusal contract; scored as weak refusal in negative-control rubric. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0039` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family as `negative_0007`: acknowledgement/citation style output instead of explicit refusal contract form. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0001` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family: scratch acknowledgement short-form output with weak refusal scoring under rubric contract. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control refusal-contract mismatch | `prompt_id=negative_0021` response refused causal proof but retained scratch/citation framing; scored `refusal_correctness=0`, `hallucination_flag=1`. | Semantically aligned refusal content but contract parser/rubric path did not accept output form as strong refusal. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0062` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family: acknowledgement/citation output, weak refusal-contract score in negative-control rubric. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control refusal-contract mismatch | `prompt_id=negative_0006` response refused unsupported mechanism/threshold but retained scratch framing; scored `refusal_correctness=0`, `hallucination_flag=1`. | Semantically refusal-like answer not accepted as strong refusal under current contract parser/rubric combination. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0048` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family as prior `Noted` rows. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0063` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family as prior `Noted` rows. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control scratch acknowledgement mis-scored as hallucination | `prompt_id=negative_0041` response: `"[Scratch] Noted. ... Confidence: high | Source: Scratchpad"`, scored `hallucination_flag=1`. | Same failure family as prior `Noted` rows. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Negative-control refusal-contract mismatch | `prompt_id=negative_0064` response refused unsupported mechanism/threshold but retained scratch/citation framing; scored `refusal_correctness=0`, `hallucination_flag=1`. | Semantically refusal-like answer not accepted as strong refusal under current contract parser/rubric combination. |
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 (in-progress campaign) | 2026-03-13 | Contradiction-row hallucination flag despite strong contradiction subscores | `prompt_id=contradiction_0364` scored `detection=2`, `source_prioritization=2`, `uncertainty_appropriate=1`, `citation_accuracy=2`, `hallucination_flag=1`. | Apparent rubric coupling/threshold behavior in contradiction scoring path; requires adjudication review to separate true fabrication from conservative contradiction-flag policy. |
| SmolLM3-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=515) | 2026-03-13 to 2026-03-14 | Contradiction lane-quality collapse under `plus_scratch` | Example: `prompt_id=contradiction_0357` (`plus_scratch`) scored `detection=0`, `source_prioritization=1`, `uncertainty_appropriate=2`, `citation_accuracy=2`, `hallucination_flag=1`; response pattern repeated as `"[Scratch] [Scratch] Noted..."` with scratch quote block instead of robust contradiction adjudication. | Primarily task-lane noncompliance (weak contradiction detection) rather than transport failure; appears as behavioral lane-quality issue in this model-policy pairing. |
| SmolLM3-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=515) | 2026-03-13 to 2026-03-14 | Reversal lane contamination under `plus_scratch` | Example: `prompt_id=reversal_0038` (`plus_scratch`) scored `distinct_args=2`, `adjudication=2`, `hedged=0`, `frame_contamination=1`, `citation_accuracy=2`, `hallucination_flag=1`. | Structured answer mostly present, but frame-contamination trigger indicates lane-quality degradation in reversal handling under current scratch policy envelope. |
| SmolLM3-Q4_K_M | routed 1000 (in-progress campaign, snapshot at n=515) | 2026-03-13 to 2026-03-14 | Negative-control lexical/contract refusal mismatch | Example: `prompt_id=negative_0061` (`plus_scratch`) responded `"[Scratch] Noted..."`, scored `refusal_correctness=0`, `citation_accuracy=0`, `hallucination_flag=1`. | Same lexical/contract family seen in Granite/Phi: refusal semantics are weakly expressed in short scratch acknowledgement format and miss strict refusal contract path. |
| SmolLM3-Q4_K_M | routed 1000 (in-progress campaign, expanded snapshot at n=985) | 2026-03-13 to 2026-03-14 | Plus-scratch dominant lane-quality failure cluster (`73/76` flags in `plus_scratch`) | Category split at snapshot: `contradiction=51`, `reversal=17`, `negative_control=7`, `evidence=1`; condition split: `no_scratch=3`, `plus_scratch=73`; runtime transport errors `0`. | Not a transport issue; indicates model-policy interaction where scratch path disproportionately degrades contradiction/reversal lane compliance for SmolLM3 under current wording/contract envelope. |

## Notes

- This file records artifact-observed failure signatures, not final adjudication.
- Where retries recovered, failures still count as operational overhead and are retained here.
- File/label naming in some legacy artifacts may contain inherited `qwen2507` prefixes despite actual model role corrections noted in interim reports.
- Pre-fix aggregate retry stats are retained intentionally to preserve before/after causality with post-policy reruns.
- Review annotation: `contradiction_0364` should be treated as a rubric-calibration review item (possible false-positive hallucination flag under conservative contradiction coupling) before final publication claims.
- Phi routed annotation (current snapshot at n=300): `14/300` flagged rows are all `negative_control` refusal-contract/lexical families (`no_scratch` "not proven" phrasing; `plus_scratch` "[Scratch] Noted" short-form), with `0` runtime transport errors in that routed slice.
- Phi routed annotation (updated snapshot at n=440): `22/440` flagged rows; still `100%` in `negative_control`, with `0` runtime transport errors; all observed rows match refusal-contract lexical families (`no_scratch` "not proven" phrasing, `plus_scratch` scratch short-form acknowledgements/quotes).
- Phi routed annotation (updated snapshot at n=575): `28/575` flagged rows; still `100%` in `negative_control`, with `0` runtime transport errors; all observed rows remain in the same lexical/contract refusal families (`no_scratch` "not proven" phrasing, `plus_scratch` `[Scratch] Noted`/short quote acknowledgements).
- Phi routed annotation (updated snapshot at n=770): `37/770` flagged rows total (`35` negative-control lexical/contract refusal mismatches, `2` contradiction rubric-coupling anomalies); `0` runtime transport errors observed in this routed slice.
- SmolLM3 routed annotation (snapshot at n=515): `35/515` flagged rows with `0` runtime transport errors; category split: `25` contradiction (lane-quality/detection weakness, mostly plus_scratch), `7` reversal (frame-contamination flags), `3` negative-control lexical/contract refusal mismatches.
- SmolLM3 routed annotation (expanded snapshot at n=985): `76/985` flagged rows with `0` runtime transport errors. Category split: `51` contradiction, `17` reversal, `7` negative_control, `1` evidence. Condition split: `3/490` (`no_scratch`) vs `73/495` (`plus_scratch`) at snapshot time.

## Hivemind Scratch vs No-Scratch (Pre vs Post Policy)

Hallucination and retry rates for Hivemind-routed artifacts:

| phase | run | condition | n | hallucinations | hallucination_rate | format_retries | retry_rate |
|---|---|---|---:|---:|---:|---:|---:|
| pre-policy | legacy routed 240 | no_scratch | 120 | 4 | 3.33% | 20 | 16.67% |
| pre-policy | legacy routed 240 | plus_scratch | 120 | 0 | 0.00% | 20 | 16.67% |
| pre-policy | legacy routed 1000 | no_scratch | 500 | 7 | 1.40% | 82 | 16.40% |
| pre-policy | legacy routed 1000 | plus_scratch | 500 | 1 | 0.20% | 138 | 27.60% |
| post-policy | routed 600 (label-corrected Hivemind block) | no_scratch | 300 | 0 | 0.00% | 0 | 0.00% |
| post-policy | routed 600 (label-corrected Hivemind block) | plus_scratch | 300 | 0 | 0.00% | 0 | 0.00% |
| post-policy | missing-lane closure 332 | no_scratch | 166 | 0 | 0.00% | 0 | 0.00% |
| post-policy | missing-lane closure 332 | plus_scratch | 166 | 0 | 0.00% | 0 | 0.00% |

Interpretation note:

- Pre-policy artifacts show measurable scratch-linked hallucination reduction in Hivemind tracks, with non-trivial retry overhead.
- Post-policy artifacts show floor-level outcomes in both conditions, so incremental scratch lift is not separable in those slices.

## Campaign-3300 Final Error Classification (Post-Run Audit)

Finalized blocks audited from:
- `validation_*_granite40_micro_ablit_*`
- `validation_*_phi4_mini_*`
- `validation_*_smollm3_*`

Classification labels used:
- `runtime/transport`: request timeout or execution failure (`error=1` in raw/scored rows)
- `lexical/contract`: refusal/format wording misses strict rubric contract (semantic refusal present but parser contract not satisfied)
- `rubric-coupling`: high subscores in target lane but conservative coupling still raises hallucination flag
- `lane-quality degradation`: model-policy interaction degrades required lane behavior (for example contradiction detection collapse, reversal frame contamination)
- `unadjudicated true hallucination`: only to be used if direct fabrication is confirmed by manual adjudication (none confirmed in this audit pass)

| model | block | n | hallucination_flags | runtime/transport | lexical/contract | rubric-coupling | lane-quality degradation | unadjudicated true hallucination |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| granite-4.0-micro-ablit.Q5_K_M | routed 1000 | 1000 | 11 | 0 | 10 (`negative_control`) | 1 (`contradiction`) | 0 | 0 |
| Phi-4-mini-instruct-Q4_K_M | routed 1000 | 1000 | 46 | 0 | 44 (`negative_control`, mostly lexical/contract refusal family) | 2 (`contradiction`) | 0 | 0 |
| SmolLM3-Q4_K_M | routed 1000 | 1000 | 78 | 0 | 8 (`negative_control` family) | 1 (`contradiction`) | 69 (`contradiction` + `reversal` + 1 `evidence`) | 0 |
| Phi-4-mini-instruct-Q4_K_M | raw 100 | 100 | 2 | 2 (read timeout to `127.0.0.1:8011`) | 0 | 0 | 0 | 0 |

Audit note:
- In the finalized routed blocks above, `runtime/transport` errors are `0`.
- The only transport faults in this campaign were in `phi4_mini_raw_100` (`2` read timeouts).
- No row in this audit pass was promoted to confirmed "true hallucination" without manual adjudication.

## Post-Patch Targeted Revalidation (Sandbox)

Patch context:
- Surgical lane-scoped hardening was applied in sandbox to address:
  - negative-control lexical/contract misses (`[Scratch] Noted` short-form and weak refusal wording)
  - contradiction/reversal lane-quality drift under plus-scratch (notably SmolLM3)
- Changed files in sandbox patch:
  - `llama_conductor/chat_postprocess.py`
  - `llama_conductor/chat_finalize.py`
  - `llama_conductor/router_fastapi.py`

Targeted rerun outcome (all post-patch):
- Granite fixcheck: `40/40`, hallucination flags `0`, errors `0`
- Phi fixcheck: `40/40`, hallucination flags `0`, errors `0`
- SmolLM3 fixcheck: `80/80`, hallucination flags `0`, errors `0`
- Aggregate: `160/160`, hallucination flags `0`, errors `0`

Artifacts:
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_granite_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_phi_fixcheck_20260314T083236Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_scored_smol_fixcheck_20260314T084930Z.jsonl`
- `TEST_ARTIFACTS_VALIDATION/fixval_summary_20260314T084930Z.json`

Why no immediate 3000-run full rerun:
- The patch was intentionally narrow and lane-targeted; the rerun was intentionally lane-targeted to validate causal fix impact with minimal confound and turnaround time.
- Operational decision: use targeted post-patch confirmation for engineering sign-off now, defer full `3 x 1000` routed parity rerun unless publication parity/comparability requires it.
- This preserves iteration speed while keeping claims bounded: post-patch success is currently supported by targeted affected-lane evidence, not yet by full cross-model 1000-per-model replacement runs.
