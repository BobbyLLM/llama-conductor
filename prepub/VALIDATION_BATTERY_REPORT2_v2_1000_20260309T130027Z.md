# VALIDATION BATTERY REPORT 2 (v2 1000-run)
Date: 2026-03-10
Model: Qwen3-4B-Hivemind
Conditions: hivemind_no_scratch vs hivemind_plus_scratch
Runs: 500 prompts x 2 conditions = 1000
Frozen runtime: temp=0.2, top_p=0.9, max_tokens=768, ctx=8192

## Run Integrity
- Total runs: 1000
- Errors: 0
- Format retries used: 220

## Latency
- hivemind_no_scratch: p50=9.25s, p95=16.88s, p99=18.00s, mean=9.88s
- hivemind_plus_scratch: p50=9.85s, p95=16.55s, p99=17.97s, mean=8.80s

## Summary By Category
### reversal
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| distinct_args | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| adjudication | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |
| frame_contamination | 0.70 | 0.88 | +0.18 | [+0.05, +0.31] |
| hedged | 0.24 | 0.26 | +0.02 | [-0.11, +0.17] |
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
| classification_accuracy_0_6 | 0.92 | 2.34 | +1.42 | [+1.02, +1.83] |
| over_upgrade_rate | 0.00 | 0.00 | +0.00 | [+0.00, +0.00] |
| uncertainty_explicit | 2.00 | 1.94 | -0.06 | [-0.12, -0.01] |
| citation_accuracy | 1.01 | 1.34 | +0.33 | [+0.23, +0.43] |

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
| detection | 2.00 | 0.00 | -2.00 | [-2.00, -2.00] |
| source_prioritization | 2.00 | 1.00 | -1.00 | [-1.00, -1.00] |
| uncertainty_appropriate | 1.98 | 1.00 | -0.98 | [-1.00, -0.94] |
| citation_accuracy | 2.00 | 2.00 | +0.00 | [+0.00, +0.00] |

### negative_control
| Metric | no_scratch | plus_scratch | delta | 95% bootstrap CI |
|---|---:|---:|---:|---|
| refusal_correctness | 1.83 | 2.00 | +0.17 | [+0.05, +0.29] |
| citation_accuracy | 1.83 | 2.00 | +0.17 | [+0.05, +0.29] |

## Hallucination and Failures
- hivemind_no_scratch: hallucination_flag_rate=1.4% (7/500), failures={'evidence_misclassification': 82, 'reversal_hedging': 20, 'reversal_frame_contamination': 59, 'negative_control_weak_refusal': 7}
- hivemind_plus_scratch: hallucination_flag_rate=0.2% (1/500), failures={'evidence_misclassification': 55, 'reversal_hedging': 22, 'reversal_frame_contamination': 74}

## Heuristic Comparator Bands (Not direct model benchmark)
Reference pattern only: 40B-like vs 70B-like grounded behavior bands.
| Metric | 40B-like band | 70B-like band | observed plus_scratch |
|---|---|---|---:|
| Reversal adjudication | 1.2-1.5 | 1.8-2.0 | 2.00 |
| ToM separation | 1.5-1.7 | 1.9-2.0 | 2.00 |
| Evidence accuracy (/6) | 2.5-3.5 | 4.5-5.5 | 2.34 |
| Retraction surgical | 1.5-1.7 | 1.9-2.0 | 2.00 |
| Refusal correctness | 1.0-1.4 | 1.8-2.0 | 2.00 |

## Assessment
- This report supports/weakens uplift claims only for this constrained grounded battery.
- 60B-70B equivalence remains heuristic unless confirmed against actual 40B/70B model runs on the same set.
