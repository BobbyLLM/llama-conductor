# VALIDATION BATTERY REPORT
Date: 2026-02-14
Model: Qwen3-4B-Hivemind (router: `default serious` vs `default serious + scratch`)
Runs: 110 prompts x 2 conditions = 220
Frozen runtime for test calls: temp=0.2, top_p=0.9, max_tokens=768

## Constraints Honored
- No code/config changes to llama-conductor or router.
- Test artifacts written only under `TEST_ARTIFACTS_VALIDATION/` plus this report file.
- Scratchpad attached/cleared/detached per run for scratch condition.

## Executive Assessment
- In this battery, uplift is mixed: scratch improves refusal behavior but degrades some adversarial-reversal cleanliness metrics.
- Punch-above-parameter-count claim is **partially supported** here (strong structure retention and low obvious hallucination), but **not fully supported** on clean adversarial adjudication under this exact prompt battery.
- Practical trust signal: low obvious fabricated-detail rate; main risk is hedge/contamination drift in reversal tasks.

## Latency
- hivemind_default_serious: mean=11.92s, p95=24.82s, n=110
- hivemind_plus_scratch: mean=9.13s, p95=15.32s, n=110

## Reversal Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| distinct_args | 2.00 | 2.00 |
| adjudication | 2.00 | 2.00 |
| frame_contamination | 0.20 | 0.60 |
| hedged | 0.45 | 0.85 |
| evidence_grounding | 2.00 | 2.00 |

## Tom Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| priority_separation | 2.00 | 2.00 |
| voice_conflation | 0.00 | 0.00 |
| inference_accuracy | 2.00 | 2.00 |

## Evidence Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| classification_accuracy_0_6 | 3.85 | 3.55 |
| over_upgrade_rate | 0.60 | 0.00 |
| uncertainty_explicit | 2.00 | 2.00 |

## Retraction Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| acknowledgment | 2.00 | 2.00 |
| retraction_type | 2.00 | 2.00 |
| confidence_calibration | 2.00 | 2.00 |
| fact_preservation | 2.00 | 2.00 |

## Contradiction Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| detection | 2.00 | 2.00 |
| source_prioritization | 2.00 | 2.00 |
| uncertainty_appropriate | 2.00 | 1.95 |

## Negative_control Metrics (means)
| Metric | hivemind_default_serious | hivemind_plus_scratch |
|---|---:|---:|
| refusal_correctness | 1.40 | 2.00 |

## Hallucination Tracking
- Method: heuristic flag for unsupported numeric/detail insertions and incorrect certainty in insufficient-evidence controls.
- hivemind_default_serious: hallucination_flag_rate=2.7% (3/110)
- hivemind_plus_scratch: hallucination_flag_rate=0.0% (0/110)
- Interpretation: this is a conservative automated proxy, not a full human fact-check.

## Failure Modes
- hivemind_default_serious: {'reversal_hedging': 9, 'evidence_misclassification': 17, 'evidence_over_upgrade': 12, 'reversal_frame_contamination': 4, 'negative_control_weak_refusal': 3}
- hivemind_plus_scratch: {'reversal_hedging': 17, 'reversal_frame_contamination': 12, 'evidence_misclassification': 14, 'contradiction_uncertainty_understated': 1}

## Artifact Paths
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw.jsonl` (original stream format)
- `TEST_ARTIFACTS_VALIDATION/validation_battery_raw_clean.jsonl` (normalized JSONL)
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored.jsonl` (original scorer output)
- `TEST_ARTIFACTS_VALIDATION/validation_battery_scored_rescored.jsonl` (robust rescoring used in this report)
