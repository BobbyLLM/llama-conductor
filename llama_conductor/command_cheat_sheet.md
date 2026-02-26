# Router Command Cheat Sheet

## Core command prefixes
- `>>` router/system commands
- `!!` Vodka memory write/manage
- `??` Vodka memory query/rewrite
- `##` one-turn selectors (Mentats/Fun)
Tips:
- Prefixes are strict and non-interchangeable.
- Same keyword under different prefixes can do different things.

## Default operators (start here)
- `>>help` compact command list
- `>>status` inspect session state
- `>>attach <kb>` then `>>list_files`
- `>>lock SUMM_<name>.md` to force deterministic grounding
- ask your question normally
- `>>memory status` inspect memory pipeline health
- `>>preset fast|balanced|max-recall` quick runtime preset switch

Minimal examples:
- `>>preset fast`
- `>>memory status`
- `>>profile casual`

## Quick utilities (`>>`)
- `>>help` show compact core help
- `>>help advanced` show full sheet
- `>>status` show session state
- `>>memory status` show memory pipeline diagnostics
- `>>profile show` show interaction profile (session-ephemeral)
- `>>profile set <field>=<value>` set profile field
- `>>profile reset` reset profile defaults/scores
- `>>profile on` / `>>profile off` enable/disable profile adapter
- `>>profile <direct|neutral|softened>` quick correction style set
- `>>profile snark <low|medium|high>` quick snark tolerance set
- `>>profile sarcasm <off|low|medium|high>` quick sarcasm set
- `>>profile profanity <on|off>` quick profanity gate set
- `>>profile verbosity <compact|standard|expanded>` quick verbosity set
- `>>profile sensitive <on|off>` toggle sensitive-context override
- `>>profile <casual|feral|turbo>` quick preset (`turbo` = `feral`)
- `>>preset <fast|balanced|max-recall>` shorthand preset switch
- `>>preset show` show active memory/context preset
- `>>preset set <fast|balanced|max-recall>` set runtime preset for this session
- `>>preset reset` clear runtime override and use config default
- `>>calc <expression>` calculator (`+ - * / %`, `**` = power), parentheses, functions (`sqrt/log/sin/cos`)
- `>>wiki <topic>` Wikipedia summary fetch
- `>>exchange <query>` currency conversion (via Frankfurter API, real-time rates)
- `>>weather <location>` current weather (via Open-Meteo API; use single word or "City Country")
- `>>find <query>` search attached KB files
- `>>peek <query>` preview KB retrieval chunks for the query
- `>>flush` clear CTC history cache, reset session profile/style identity, and delete session-memory JSONL files (does not detach KBs)
Tips:
- Use these when you want a direct tool result rather than a free-form chat answer.
- API-backed commands (`wiki`, `exchange`, `weather`) can fail if upstream services are unavailable.
- Soft aliases are supported: `profile show|set|reset|on|off`.
- Soft aliases are supported: `preset show|set|reset`, `preset fast|balanced|max-recall`, `memory status`.
- Manual `>>profile set` values are pinned and take precedence over inferred updates until changed again or reset.

## KB attachment (`>>`)
- `>>list_kb` list known and attached KBs
- `>>list_files` list lockable `SUMM_*.md` files in attached filesystem KBs
- `>>attach <kb>` attach KB
- `>>attach all` attach all KBs
- `>>detach <kb>` detach KB
- `>>detach all` detach all KBs
- `>>lock SUMM_<name>.md` lock normal query grounding to one SUMM file (from attached filesystem KBs)
- `>>unlock` clear active SUMM file lock
Tips:
- After `>>attach <kb>`, normal chat queries are grounded against attached KB content until you detach it.
- Attachments persist for the session (sticky), so use `>>detach <kb>` or `>>detach all` to stop grounding.
- `>>detach all` also clears active lock state (same as `>>unlock`).
- `>>detach <kb>` also clears lock if the locked file belongs to that KB.
- While lock is active, normal chat grounding is deterministic and scoped to the locked SUMM file.
- Soft aliases (when a filesystem KB is attached): `lock SUMM_<name>.md` and `unlock`.
- Strict soft alias (filesystem KB must be attached): `list files` -> `>>list_files`.
- Partial lock alias (filesystem KB must be attached): `lock <partial_name>` -> router suggests one candidate as:
  - `Did you mean: >>lock SUMM_<name>.md ? [Y/N]`
  - Reply `Y` to lock, `N` to cancel.

## Scratchpad (`>>`) [Advanced operators]
Attach first:
- `>>attach scratchpad`
- `>>detach scratchpad` (detaches and deletes current session scratchpad data)

Full commands:
- `>>scratchpad status`
- `>>scratchpad list`
- `>>scratchpad show [query]`
- `>>scratchpad show all` (full dump of all stored scratchpad records for this session)
- `>>scratchpad clear` (alias: `>>scratchpad flush`)
- `>>scratchpad add <text>`
- `>>scratchpad delete <index|query>`

Aliases:
- `>>scratch` attaches scratchpad
- `>>scratch ...` is an alias root for all `>>scratchpad ...` commands
- `>>attach scratch` / `>>detach scratch` map to `scratchpad`
- `>>list scratchpad` -> `>>scratchpad list`
- bare `scratchpad show <query>` -> `>>scratchpad show <query>` (only when scratchpad is attached)
- bare `scratch show <query>` -> `>>scratchpad show <query>` (only when scratchpad is attached)
- When scratchpad is attached:
  - `>>add <text>` -> `>>scratchpad add <text>`
  - `>>list` shows scratchpad captures
- When scratchpad is not attached:
  - `>>list` shows known KBs

Tips:
- Scratchpad is session-ephemeral and used for grounded reasoning when attached.
- Selected tool outputs are auto-captured while attached.
- `>>detach all` also deletes scratchpad data when scratchpad is attached.
- `>>list` is context-sensitive: scratchpad list when attached, KB list when not attached.
- `>>list` may show more records than a normal follow-up reasoning turn uses; to query against all contents (eg: "compare xyz to abc"), run `>>scratchpad show all` first, then ask the question.
- Raw captures are stored at `total_recall/session_kb/<session_id>.jsonl`.

Two-step workflow:

Other:
Tips:
- Deterministic scaffold first, then optional stochastic polish.

## Sticky modes (`>>`)
- `>>fun` / `>>f` / `>>F` enable Fun mode
- `>>fun off` / `>>f off` / `>>F off` disable Fun mode
- `>>fr` / `>>FR` / `>>fun_rewrite` enable Fun Rewrite
- `>>fr off` / `>>FR off` / `>>fun_rewrite off` disable Fun Rewrite
- `>>raw` enable Raw mode
- `>>raw off` disable Raw mode
Tips:
- Sticky modes persist until turned off.
- Per-turn selectors can bypass sticky behavior for that turn.

## SUMM and Vault (`>>`) [Advanced operators]
- `>>summ new` summarize new raw files in attached KB folders
- `>>move to vault` embed attached KB summaries into Vault
- `>>move new to vault` embed only the newest `SUMM_*.md` per attached KB into Vault
Tips:
- `>>summ new` processes new raw docs in currently attached KB folders.
- `>>summ` is deterministic/extractive in current builds (no LLM call in the SUMM generation step).
- `>>move to vault` promotes summaries for Vault retrieval; it does not make Vault attachable via `>>attach`.
- `>>move_to_vault new` and `>>mtv new` are equivalent shorthand for newest-only promotion.
- SUMM mechanics are unchanged: creates `SUMM_*.md`, moves source docs into `/original/`, and keeps provenance headers.

## Tool recommendation (`>>`)
- `>>trust <query>` suggests best tool path (A/B/C)
Tips:
- `>>trust` is a router guide step; choose A/B/C to execute.
- Pending recommendation state is transient and replaced by newer routing actions.

## Vodka memory (`!!` / `??`) [Advanced operators]
Write/manage:
- `!! <text>` store memory
- `!! forget <query>` delete matching memories
- `!! nuke` delete all Vodka memories

Query:
- `?? <query>` retrieve memories and rewrite question with matched context
- `?? list` list Vodka memories (TTL, touch count, creation date)
Tips:
- `!!` writes/manages memory; `??` reads/rewrites with memory context.
- `!! nuke` clears Vodka memory only; `>>flush` clears CTC cache and resets profile/style runtime state.

## One-turn selectors (`##`) [Advanced operators]
- `##mentats <question>` Vault-only run
- `##m <question>` alias
- `##fun <question>` / `##f <question>` one-turn Fun
Tips:
- Selectors apply to the current turn only.
- Use selectors for explicit pipeline control when sticky modes are active.
- `##mentats` remains Vault-only and does not use filesystem lock scope.
- Non-Mentats `Confidence: ... | Source: ...` footer is router-normalized deterministically.
- Non-Mentats profile footer is deterministic and policy-gated:
  - `Profile: <correction_style> | Sarc: <level> | Snark: <level>`
  - default policy (`footer.profile.mode: non_default`) only shows this line when style is non-default.
  - How to read footer fields:
    - `Profile` = correction style axis (`softened|neutral|direct`)
      - `softened`: gentler correction framing
      - `neutral`: standard/default correction framing
      - `direct`: terse explicit correction framing
    - `Sarc` = sarcasm level (`off|low|medium|high`)
      - controls irony/playful mockery intensity
    - `Snark` = snark tolerance (`low|medium|high`)
      - controls how sharp/blunt combative framing is allowed
  - Example: `Profile: neutral | Sarc: medium | Snark: high`
    - means standard correction style, medium sarcasm, high sharpness tolerance.
  - Inference is rule/marker-based (deterministic), not LLM semantic inference.
  - `snark` and `sarcasm` are separate traits; profanity can raise snark faster than sarcasm.
  - So `Profile: neutral | Sarc: off | Snark: high` can be expected in some sessions.
- Mentats keeps its own `Sources: Vault` contract.

## Vision/OCR
- `>>vision` / `>>vl` / `>>v` with image: direct vision answer
- `>>ocr` / `>>read` with image: OCR text extraction
Tips:
- `>>vision`/`>>ocr` and `##vision`/`##ocr` route to the same vision pathway.
- You can simply attach an image and ask your question in natural language; the router will auto-run vision.
- Use `>>vision`/`>>ocr` (or `##vision`/`##ocr`) when you want to force a specific image-processing path.

## Status fields (`>>status`) [Advanced operators]
- `session_id` current chat/session identifier used by the router
- `attached_kbs` KBs currently attached for retrieval grounding
- `locked_summ_file` currently locked SUMM filename (empty = no lock)
- `locked_summ_kb` KB that owns the locked SUMM file (empty = no lock)
- `pending_lock_candidate` pending Y/N lock suggestion target (empty = none)
- `pending_sensitive_confirm_query` pending Y/N sensitive-confirm query (empty = none)
- `fun_sticky` whether sticky Fun mode is currently enabled
- `fun_rewrite_sticky` whether sticky Fun Rewrite mode is currently enabled
- `profile_enabled` whether style adapter is currently enabled
- `profile_confidence` inferred profile confidence score (`0.00-1.00`)
- `profile_effective_strength` runtime strength of profile constraints (`0.00-1.00`)
- `profile_output_compliance` rolling score of output/profile contract match (`0.00-1.00`)
- `profile_blocked_nicknames` session-blocked nicknames (from explicit user disallow prompts)
- `profile_last_updated_turn` latest user turn that updated profile state
- `serious_ack_reframe_streak` current consecutive serious-mode meta-ack streak
- `serious_repeat_streak` current consecutive serious-mode near-duplicate output streak
- `last_query` most recent filesystem-KB retrieval query
- `last_hits` retrieval hit indicator (`0`=no matches; `>0`=matches found)
- `vault_last_query` most recent Vault retrieval query
- `vault_last_hits` same idea, but for Vault retrieval
- Memory status fields (`>>memory status`):
  - `preset` active runtime memory preset
  - `enabled_summary` whether session-memory pipeline is active
  - `require_session_id` whether session id is required to use memory
  - `summary_every_n_user_msgs` update cadence
  - `unit_count` retained session-memory unit count
  - `last_update_turn` last turn where memory units were rebuilt
  - `last_inject_turn` last turn where memory retrieval ran
  - `last_inject_units` number of units injected/retrieved last turn
  - `last_candidate_count` shortlist size before final pick
  - `last_query` last query that hit memory retrieval
Tips:
- If an answer is weak and hits are `0`, the issue is likely missing retrieval; if hits are non-zero, the issue is more likely synthesis/interpretation quality.
- Hit counts are a quick clue, not a quality grade.
