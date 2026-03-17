# GPT-AUTO-TEST-v3

Date: 2026-02-13 21:15:30
Method: Derived from retained v2 benchmark scores + pre/post structural artifacts + stress-run evidence.
Scoring basis: same heuristic family as v2 (GOLD proximity %); v3 case scores are unchanged where pre/post review outputs are identical.

## Summary Table

| Case | v3 Score | Previous | Trend vs Previous | PRE->POST Score Delta | Auto Header (PRE->POST) | Review Pass (PRE->POST) |
|---|---:|---:|---|---:|---|---|
| TEST-CASE-wrist-dequervains-tenosynovitis.txt | 88 | 72 | UP (+16) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-Shoulder.txt | 88 | 85 | UP (+3) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-LBP.txt | 85 | 63 | UP (+22) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-knee-patellofemoral-pain.txt | 88 | 86 | HOLD (+2) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-elbow-lateral-epicondylitis.txt | 87 | 88 | HOLD (-1) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-cervical.txt | 76 | 79 | DOWN (-3) | +0 | 2->1 | PASS->PASS |
| TEST-CASE-GTPS-ASCII.txt | 88 | - | NEW | +0 | 2->1 | PASS->PASS |
| TEST-CASE-ankle-achilles-tendinopathy.txt | 88 | - | NEW | +0 | 2->1 | PASS->PASS |
| TEST-CASE-MIDBACK.txt | 88 | - | NEW | +0 | 2->1 | PASS->PASS |
| TEST-RCT.txt | 84 | - | NEW | +0 | 2->1 | PASS->PASS |
| TEST-ACL.txt | 84 | - | NEW | +0 | 2->1 | PASS->PASS |

## Recalculated Averages

- Previous cohort (now): **85.33**
- Previous cohort (old baseline): **78.83**
- Delta (previous cohort avg): **+6.50**
- New tests cohort average: **86.40**
- Overall average (all successful runs): **85.82**

## PRE vs POST Structural Delta

- GOLD proximity score shift: **0.00** points overall (no regression, no uplift in this matrix).
- Auto header duplication: **fixed** (`avg 2.0 -> 1.0` blocks).
- Review validation pass count: **11/11 -> 11/11**.
- Off-target markers tracked in compare file: remained **0/11** cases before and after.

## Stress Run (Post-change)

- Dataset: `LBP`, `Shoulder`, `Cervical` x 15 repeats each (45 runs total).
- Review pass rate: **100.0%** (45/45).
- Validation failures: **0**
- Review errors: **0**
- Off-target term runs: **0** (0.0%).
- TEST-CASE-LBP.txt: stable diagnosis `Mechanical low back pain` in 15/15 runs.
- TEST-CASE-Shoulder.txt: stable diagnosis `Rotator cuff tendinopathy` in 15/15 runs.
- TEST-CASE-cervical.txt: stable diagnosis `Mechanical neck pain` in 15/15 runs.

