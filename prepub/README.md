# prepub Evidence Bundle

This folder contains pre-publication benchmark artifacts used by `PAPER2.md`.

## Included files

- `PAPER.md` - preprint draft (bounded claims)
- `VALIDATION_BATTERY_REPORT.md` - original validation summary
- `VALIDATION_BATTERY_REPORT2.md` - expanded validation summary with CIs
- `validation_battery_raw2.jsonl` - raw paired run outputs
- `validation_battery_scored2.jsonl` - scored outputs
- `VALIDATION_BATTERY_REPORT2_v2_1000_20260309T130027Z.md` - 1000-run replication summary (v2)
- `validation_battery_raw2_v2_1000_20260309T130027Z.jsonl` - 1000-run replication raw outputs (v2)
- `validation_battery_scored2_v2_1000_20260309T130027Z.jsonl` - 1000-run replication scored outputs (v2)
- `validation_battery_meta_v2_1000_20260309T130027Z.json` - 1000-run replication metadata (v2)
- `prompts_manifest2_v2_1000_20260309T130027Z.jsonl` - 1000-run replication prompt manifest (v2)
- `_STRESS-CLINIKO-REVIEW-v1.md` - repeated clinical test-case stress report (45 runs)
- `GPT-AUTO-TEST-v3.md` - consolidated heuristic/structural stability report
- `cliniko_stress_replay_expanded_20260308T194459Z.json` - expanded Cliniko stress replay raw results (11 cases x 15 repeats)
- `cliniko_stress_replay_expanded_20260308T194459Z.md` - expanded Cliniko stress replay summary report
- `cliniko_live_sweep_20260226T110651Z.json` - full core live sweep results
- `cliniko_live_ankle_random_battery_20260228T141106Z.json` - randomized ankle live battery raw results
- `cliniko_live_ankle_random_battery_20260228T141106Z.md` - randomized ankle live battery summary

## Scope

These artifacts support constrained benchmark claims only. They do not claim universal model reliability outside this battery/setup.

Two-rater blinded note:
- A two-rater blinded adjudication protocol exists in project validation docs, but this `prepub` bundle does not include executed two-rater blinded outputs yet.
- Current claims are therefore bounded to the included automated rubric/scoring artifacts.

## Notes

- Test content is synthetic case-file style data (agency/council placeholders), not production user chat logs.
- Session IDs in raw files are run identifiers, not personal identities.
