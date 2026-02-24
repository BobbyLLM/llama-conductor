# VALIDATION BATTERY REPORT 2
Date: 2026-02-15
Model: Qwen3-4B-Hivemind
Conditions: hivemind_no_scratch vs hivemind_plus_scratch
Runs: 120 prompts x 2 conditions = 240
Frozen runtime: temp=0.2, top_p=0.9, max_tokens=768, ctx=8192

## Run Integrity
- Total runs: 240
- Errors: 0
- Format retries used: 40

## Latency
- hivemind_no_scratch: p50=7.81s, p95=13.46s, p99=14.95s, mean=8.02s
- hivemind_plus_scratch: p50=7.66s, p95=12.70s, p99=14.19s, mean=8.15s

## Summary By Category
### reversal
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| distinct_args | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| adjudication | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| frame_contamination | 0.80 | 0.85 | +0.05 | [-0.15, +0.25] |
| hedged | 0.15 | 0.70 | +0.55 | [+0.25, +0.80] |
| citation_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |

### tom
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| priority_separation | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| inference_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| voice_stability | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| citation_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |

### evidence
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| classification_accuracy_0_6 | 3.60 | 3.45 | -0.15 | [-1.30, +1.00] |
| over_upgrade_rate | 0.05 | 0.00 | -0.05 | [-0.15, +0.00] |
| uncertainty_explicit | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| citation_accuracy | 1.70 | 1.65 | -0.05 | [-0.35, +0.25] |

### retraction
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| acknowledgment | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| retraction_type | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| confidence_calibration | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| fact_preservation | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| citation_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |

### contradiction
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| detection | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| source_prioritization | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| uncertainty_appropriate | 1.90 | 1.15 | -0.75 | [-0.90, -0.55] |
| citation_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |

### negative_control
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| refusal_correctness | 1.60 | 2.00 | +0.40 | [+0.10, +0.80] |
| citation_accuracy | 1.60 | 2.00 | +0.40 | [+0.10, +0.80] |

## Hallucination and Failures
- hivemind_no_scratch: hallucination_flag_rate=3.3% (4/120), failures={'reversal_frame_contamination': 16, 'evidence_misclassification': 9, 'reversal_hedging': 3, 'evidence_over_upgrade': 1, 'negative_control_weak_refusal': 4}
- hivemind_plus_scratch: hallucination_flag_rate=0.0% (0/120), failures={'reversal_hedging': 14, 'reversal_frame_contamination': 17, 'evidence_misclassification': 10}

## Heuristic Comparator Bands (Not direct model benchmark)
Reference pattern only: 40B-like vs 70B-like grounded behavior bands.
| Metric | 40B-like band | 70B-like band | observed plus_scratch |
|---|---|---|---:|
| Reversal adjudication | 1.2-1.5 | 1.8-2.0 | 2.00 |
| ToM separation | 1.5-1.7 | 1.9-2.0 | 2.00 |
| Evidence accuracy (/6) | 2.5-3.5 | 4.5-5.5 | 3.45 |
| Retraction surgical | 1.5-1.7 | 1.9-2.0 | 2.00 |
| Refusal correctness | 1.0-1.4 | 1.8-2.0 | 2.00 |


