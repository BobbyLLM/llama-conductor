# Live Ankle Random Battery

- created_utc: `20260228T141106Z`
- seed: `1772287866`
- run_count: `8`
- gold_ref: `C:\moa-router1.2.1 TESTING\llama_conductor\test tools\GOLD-ANKLE.md`
- flush_guard_available_precheck: `True`

## Aggregate

- auto pass: `8/8`
- review pass: `8/8`
- flush guard pass: `8/8`
- expected diagnosis hit: `8/8`
- avg proximity composite vs GOLD-ANKLE: `0.4709`
- avg token_f1 vs GOLD-ANKLE: `0.4717`

## Case Results

| case | condition | dx hit | review ok | flush ok | proximity composite | token_f1 | cosine | seq |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | lateral_sprain | True | True | True | 0.4734 | 0.4802 | 0.5529 | 0.1599 |
| 2 | ankle_oa | True | True | True | 0.4708 | 0.4888 | 0.5331 | 0.1562 |
| 3 | posterior_impingement | True | True | True | 0.4853 | 0.4947 | 0.5713 | 0.1696 |
| 4 | syndesmosis | True | True | True | 0.4794 | 0.4566 | 0.6066 | 0.1871 |
| 5 | medial_deltoid | True | True | True | 0.4627 | 0.4636 | 0.5395 | 0.1618 |
| 6 | ocl_talus | True | True | True | 0.4539 | 0.4329 | 0.5926 | 0.1927 |
| 7 | peroneal | True | True | True | 0.4742 | 0.4799 | 0.5572 | 0.1628 |
| 8 | instability | True | True | True | 0.4675 | 0.4769 | 0.5275 | 0.1680 |

## Top 3 by Proximity

- case `3` `posterior_impingement`: composite `0.4853` (token_f1 `0.4947`)
- case `4` `syndesmosis`: composite `0.4794` (token_f1 `0.4566`)
- case `7` `peroneal`: composite `0.4742` (token_f1 `0.4799`)