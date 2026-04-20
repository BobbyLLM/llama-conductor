# What's New

### 1.9.5 (LATEST)

TL;DR: Codex grounding introduced. Now runs end-to-end from SUMM-backed source text, with live e5 + TinyBERT validation and router wiring complete. See FAQ.md for full implementation details. Which...I promise to write very soon so you know WTF this is. TL;DR: https://tinyurl.com/llmwiki 

- Codex grounding integration:
  - extracted `llama_conductor/codex_runtime.py` as the grounding runtime
  - two-stage grounding now uses e5 candidate selection + TinyBERT scoring
  - `_TINYBERT_THRESHOLD` calibrated to `4.0` on live SUMM-backed smoke runs
  - `raw_text` added to `codex_payload` so grounding uses the original SUMM content, not the rendered Codex page
  - `facts_block` wiring now appends grounded Codex content from `raw_text`
  - `chat_mode_execution.py` applies grounding after `_postguard(text)` across FUN, FUN REWRITE, serious, and raw


### 1.9.4

TL;DR: fixed sticky-vision carryover from conversation-history DOM image recapture in the shim.

- `>>vision` sticky fix (shim-side sent-URL dedup):
  - fixed conversation-history image re-injection loop in shim DOM fallback capture
  - added sent-URL tracking set (`window.__moaShimSentImageUrls`) with helper guards:
    `sentImageUrlsStore()`, `markImageUrlAsSent()`, `isImageUrlSent()`
  - `collectImagesFromDomFallback()` now skips URLs already injected on prior turns
  - fetch interceptor marks pre-inject pending URLs as sent after successful payload injection
  - behavior: Turn 1 image injects normally, Turn 2+ text turns no longer re-send prior images;
    sent URL set clears on page reload

### 1.9.3 

TL;DR: just some housekeeping / avoiding the real work. Though, in truth, this needed doing to make what comes next easier.


- Serious-mode hostile guard hardening (B2.6 / B2.7 / B2.8):
  - compound hostile clause detection: second-pass recheck after first-pass CLAUSE_TRIM
  - targeted imperative-attack trim (`_SERIOUS_IMPERATIVE_INSULT_TOKENS`, dynamic regex)
  - orphaned fragment cleanup after clause trim (leading sentence fragment detection + allowlist)
  - post-finalize telemetry snapshot added (`stage=serious_post_finalize`) for accurate guard diagnostics
  - silent exception in guard block now logged (`serious_guard_exception_logged` telemetry field)

- Order-status retrieval gate (Track-B):
  - context-free order queries (`"What's the status of my order?"`) now suppressed from web retrieval
  - symmetric inclusion: order queries WITH tracking ID / merchant context still route to web
  - extensible carrier/merchant set (`_ORDER_CONTEXT_CARRIER_TERMS`) including Australia Post, StarTrack, eBay

- Clarification phrase bank:
  - replaced single hardcoded clarifier phrase with randomised 6-phrase feral bank (`_CLARIFICATION_FALLBACK_PHRASES_FERAL`)
  - added serious/personal register variant (`_CLARIFICATION_FALLBACK_PHRASES_SERIOUS`) with KAIOKEN macro gating
  - deterministic slot fragment extractor appends context echo ("What order?", "Lost me too.", "Which thing?")
  - feral register weak-placeholder replacement (`FERAL_FALLBACK` action) on model-origin canned outputs

- Distress first-response tone sequencing:
  - `distress_turn_count` session field added; resets on `>>flush`
  - vent-opener probe suppressed on Turn 1 distress; fires from phrase bank on Turn 2+
  - task-forward fallback (`"What do you need help with right now?"`) guarded from non-distress clarification path (four overwrite sites)

- Model-origin canned phrase detection:
  - `_normalize_simple_phrase` apostrophe normalization fix
  - `SERIOUS_WEAK_PLACEHOLDER_FALLBACK` action on non-feral weak-placeholder turns


### Completed in 1.9.2

TL;DR: 

The headline is a new workflow: fetch something from the web, drop it in your scratchpad, reason over it - one command chain instead of copy-pasting between windows. 

Pick your retrieval mode (raw search, Wikipedia, or synthesised answer) and the results land directly in scratch, ready to lock and judge against.

Web synthesis (>>web synth) is now a real feature. It pulls from multiple independent sources, checks they're credible, and gives you a grounded answer with citations. If the evidence isn't good enough, it refuses rather than guesses. That's not a limitation — that's the point.

The model can no longer confidently hallucinate who currently holds a role or position when it doesn't actually know. It refuses instead.
Under the hood: groundwork laid for the upcoming swarm architecture, and the evidence audit trail got more transparent - you can now see exactly what evidence >>judge was working with on any run.

Additionally, added a new memory transparency command (>>vodka debug) shows you when and why the system trimmed your conversation context. No more wondering why something disappeared.

**Why care?** The system is getting better at knowing what it doesn't know - and saying so, rather than bullshitting confidently. For a local 4B model, that's punching well above its weight.

**What's next:** The road to 2.0.0 is swarm architecture (see upcoming blog posts). 

Before that lands, the codebase goes through a decomposition pass - breaking the monolithic modules into clean, testable units. 

No new feature releases are likely to ship until decomposition is done and swarm infrastructure is in place. 

2.0.0 is the target for the full swarm build. Keep an eye out on [Blog](https://bobbyllm.codeberg.page/llama-conductor/) 


- `>>web synth` lane fully wired and hardened:
  - evidence-constrained synthesis from gated top-N web rows
  - global gate: UGC strip -> independence (`>=2` distinct eTLD+1) -> credibility floor
  - one-shot post-refusal wiki fallback path
  - transport/router fail-loud behavior preserved (`Source: Model` on refusal)
  - grounded success path emits `See:` links + `Source: Web`

- Routing gate stabilization (v7.1):
  - embedded explicit sidecar commands extracted pre-gate and routed deterministically (`>>web` / `>>wiki` / `>>web synth`) while preserving full-turn profile context
  - retrieval trigger precision tightened to reduce lexical over-trigger on reasoning/casual turns
  - macro-transition retrieval-carry scrub added via `last_turn_kaioken_macro`
  - validation: full A+B+D matrix pass (`21/21`) + canary (`3/3`)

- Trust-domain expansion for general knowledge/policy:
  - `web_search.user_trust_domains` expanded in `router_config.yaml`
  - trust matching normalized to eTLD+1 roots (subdomains inherit)
  - live check confirmed grounded inflation synthesis (`Source: Web`)

- `who_is_current` anti-confabulation guard:
  - decomposed intent detector + retrieval miss protection
  - integrated with existing `who_won` durability pattern
  - validated with grounded/miss/non-regression probes

- Showtimes intent contract (first registered schema instance):
  - shared contract rendering across NLP alias and `>>web`
  - deterministic/fail-loud showtimes extraction lane
  - stable live validation on repeated runs

- `WebSearchHit` dataclass extension (struct-only groundwork):
  - added: `domain`, `source_type`, `serp_score`, `content_score`, `corroboration_score`, `fetch_status`, `canonical_url`
  - no scoring-logic behavior change in this step

- Item 4 operator pipeline:
  - `>>scratch online` (web/wiki/web synth) with interactive selector
  - `>>scratch load web` compatibility alias (snapshot/idempotent session handoff)
  - selector parity improvements (`1|2|3` and command aliases), disambiguation flags, and receipt cleanup

- Item 5 transparency lane:
  - `>>vodka debug` added
  - trim telemetry events surfaced (`JANITOR_EXPIRE`, `JANITOR_CAP`, `CLIP`, `NUKE`)
  - validation artifact: `docs/validation/VODKA-JANITOR-VERIFICATION-2026-04-09.md`

- Config migration notes (for existing local `router_config.yaml` installs):
  - new keys do not auto-appear in older local configs; merge from release baseline
  - synthesis policy expects global default block: `web_search.synth_policies.default`
  - `>>scratch online`/`>>scratch load web` rely on current `web_search` settings
  - judge JSONL now includes `scratch_locked_raw_chars` and `judge_evidence_chars`
  - operator rule: diff local config against release baseline after upgrade


## *** V1.9.1 

TL;DR:

Bugpatch release focused on retrieval scoring stability and live regression hardening.

- Confirmed Track B web trigger behavior so banter-adjacent web retrieval firing is documented and reproducible.
- Fixed `_quote_phrase_candidates` to generate structural sub-phrases (instead of whole-query exact matching), which restores phrase-score signal on natural-language web queries.
- No trust-weight tuning in this patch; scoring formula and thresholds are unchanged.
- Verified live improvements across the key regression set after the structural fix.

---

## *** V1.9.0

TL;DR:

Web retrieval sidecar with deterministic relevance gating. The model now searches the internet before making things up. Receipts included.

**`>>web` - provider-agnostic web retrieval**

New deterministic sidecar for live web search. `>>web <query>` returns ranked web results with strict relevance scoring and honest provenance. Provider-agnostic: ships with `ddg_lite` (DuckDuckGo, no API key needed), supports `tavily`, `searxng`, and `custom` adapters.

Results are scored deterministically - exact phrase match bonus, token-overlap ratio, and trusted-domain boost - with a hard threshold before anything counts as evidence. Garbage in, garbage out? No. Garbage in, fail-loud out.

**Automatic retrieval cascade**

The router no longer waits for you to type `>>web`. When cheatsheets and wiki can't answer a retrievable-fact query, web search fires automatically before model fallback.

Retrieval order: `Cheatsheets â†’ Wiki â†’ Web â†’ Model`. Each step only fires if the previous one missed. The model is last resort, not first call.

This is triggered by a broad `needs_web_retrieval` signal (WH-questions, named entities, attribution queries, date/event phrasing), not a narrow intent enum. If the question looks like something the internet could answer, the system tries before the model guesses.

**Refusal durability guard**

New safety guard for quote/source attribution queries. If retrieval already failed and the user pushes ("no, it's a quote, do you know the source?") without adding new evidence, the model is hard-blocked from inventing attributions. Previously, follow-up pressure caused the model to cave and confabulate. Now it holds the line.

Guard releases when new evidence is supplied (new entity tokens, URLs, pasted content). Safety constraint, not capability limiter.

**Deterministic source links**

Answers grounded from wiki or web now include a `See: <url>` line before the footer. The URL is injected deterministically from retrieval metadata - never model-generated. One click to verify.

- `Source: Wiki` -> `See: https://en.wikipedia.org/wiki/...`
- `Source: Web` -> `See: https://actual-source-url.com/...`
- `Source: Cheatsheets` / `Source: Model` -> no `See:` line (nothing external to link to).

**User trust domains**

New config extension: `web_search.user_trust_domains`. Add your own trusted domains (e.g. `bbc.co.uk`, `reuters.com`, `pubmed.ncbi.nlm.nih.gov`) and they get the same relevance scoring boost as built-in domains. Ships empty by default. Built-in domain list stays active and invisible.

```yaml
web_search:
  user_trust_domains: []
  # Example:
  # user_trust_domains:
  #   - bbc.co.uk
  #   - reuters.com
```

**What this is building toward**

`>>web` closes the last major retrieval gap before swarm. The system can now ground answers from your definitions (Cheatsheets), encyclopedic summaries (Wiki), live web evidence (Web), curated deep knowledge (Vault/Mentats), or your own pasted context (Scratchpad/Lock). Model fallback is truly last resort now.

See [FAQ](FAQ.md) for full `>>web` details and config options.

---

## *** V1.8.0

TL;DR: 

Real-time turn classification (KAIOKEN) plus deterministic static grounding (Cheatsheets). The model now knows what you're doing and has verified facts before it generates a single token.

Two systems shipped in this release. They are separate but complementary.

**KAIOKEN v1 â€” Behavioural shaping**

The model doesn't know what you're doing right now. KAIOKEN teaches it.

Every message gets classified before the model sees it. Macro label: working, casual, or personal. Subsignals: playful, friction, distress_hint, vulnerable_under_humour, kb_lookup_candidate. Those labels map to behavioural instructions injected into the thinker context before generation. Actual constraints â€” not metadata, not hints.

This is the first piece of the Claude in a Can architecture: externalising what frontier models do implicitly via scale into explicit, inspectable infrastructure. See [FAQ](FAQ.md) and [Claude-in-a-can](https://bobbyllm.github.io/llama-conductor/blog/claude-in-a-can-1/) for more details

**Cheatsheets â€” Deterministic knowledge grounding**

Cheatsheets is a static knowledge base - JSONL files in llama_conductor/cheatsheets/ - that fires silently on every turn. 

When you mention a known term, the router matches it, injects verified facts into the thinker context before generation, and the footer tells you it happened. 

No vector search, no LLM-generated context, no user invocation required. Drop a new file in the directory, it auto-loads. Remove it, it's gone. No restart.

Fires globally across all chat modes (serious, >>fun, >>fr, >>raw). 


**What this is building toward**
KAIOKEN is the steering wheel. Cheatsheets is the map. >>wiki integration and Swarm will be final pieces. 

The eventual goal of "Claude in a Can" is to achieve model decomposition, behavioural shaping and knowledge grounding on hardware Anthropic wouldn't use to play Tetris on. 

See [FAQ](FAQ.md) and [Claude-in-a-can](https://bobbyllm.github.io/llama-conductor/blog/claude-in-a-can-1/) 


---

*** V1.7.2

TL;DR:
Judge + Trust UX pass: clearer routing text, explicit ungrounded warning, visible winner line, and `>>flush` now clears scratchpad captures too. Plus a BONUS GOODY - see below.

- `>>trust` comparative output is now cleaner/scannable:
  - compact `A/B/C` blocks
  - plain-English `Why` lines
  - explicit `Cmd` line
- `>>trust` A-path paste prompt now reads:
  - `[PASTE EVIDENCE FOR JUDGE]`
  - `Paste source text now (or type CANCEL).`
  - `Enter anything else to run ungrounded >>judge.`
- `>>judge` output now includes `Winner:`:
  - single winner: `Winner: <item>`
  - tied top score: `Winner: TIE (<item1>, <item2>, ...)`
- Ungrounded `>>judge` runs now prepend a 3-line risk receipt:
  - no evidence loaded
  - confidence reflects model priors, not verified facts
  - recommends `>>scratch` pathway for better outcomes
- `>>flush` parity upgrade:
  - now clears current session scratchpad captures
  - clears scratch lock state
  - still clears session-memory and judge-audit artifacts

BONUS GOODY TIME!

**MoA Chat Bridge** - a Firefox extension that replaces Firefox's built-in "Ask AI Chatbot" context menu for localhost LLM providers.

**Problem:** Firefox's built-in AI Chatbot sidebar sends selected text via a `?q=` URL parameter. llama.cpp WebUI doesn't clear that param after consumption, so re-renders can re-read it and trigger a second generation. That's a llama.cpp WebUI bug, not Firefox.

**Solution:** A four-file extension (`manifest.json`, `background.js`, `sidebar.html`, `bridge.js`) that:

- adds a "MoA Chat" right-click menu with Summarize / Translate / Analyze Sentiment / Send to Chat
- opens llama.cpp WebUI in a sidebar iframe (`sidebar.html`)
- uses `browser.storage.local` to pass selected text + prompt prefix to the content script (`bridge.js`)
- injects text directly into the WebUI textarea and fires `input` so React picks it up
- never uses `?q=`, so no double-generation path

**Location:** `extras/firefox-extension/`

**XPI:** [moa chat bridge.xpi](extras/firefox-extension/moa%20chat%20bridge.xpi)

**Install:**
- temporary load via `about:debugging`
- or install signed `.xpi` via `about:addons` -> Install From File
- self-distributed, signed by Mozilla, not listed on public AMO

---

*** V1.7.1

TL;DR:
Judge is now meaningfully grounded when scratchpad is attached.
If you lock scratch evidence, judge uses that scope and fails closed when evidence is unusable.

- `>>judge` now consumes scratchpad evidence when attached, including lock-scoped records (`>>scratch lock <n>`).
- Added fail-closed contract for judge when locked scratch evidence does not support the criterion/options.
- Added reason/verdict mismatch guard with single strict verdict-only retry to reduce contradictory pair outputs.
- Judge verbose JSONL now includes evidence provenance fields:
  - `evidence_source`
  - `evidence_locked_indices`
  - `evidence_chars`

---

*** V1.7.0

TL;DR:
Heavy stabilization + release plumbing pass.
Less hidden jank, cleaner startup/runtime behavior, and a safer public promotion path.

- Added first-party `>>judge` v1 command path (deterministic pairwise rank worker with fail-loud parse behavior).
- Added live campaign bank/runner artifacts to validate `>>judge` on active router path.
- Added global startup janitor for `total_recall/session_kb` (not just lazy per-session pruning).
- Vodka memory storage now scopes to `total_recall/vodka/` with legacy read fallback.
- Added Vodka startup janitor pass for `facts.json` (startup + lazy runtime janitor now both active).
- Ran three hardening campaigns end-to-end: `>>judge` live router battery, Vodka/session-memory janitor migration + regression gates, and controlled public-refactor shakedown (rename/check + strip/compile/version parity).
- Added large-scale hallucination validation evidence (~9k run campaign) and tied release framing back to the prepub write-up: [PAPER.md](prepub/PAPER.md).

---

*** V1.6.0

TL;DR:
- Completed heavy stack cutover away from active llama-swap dependency.
- Backend path is now provider-driven via `router_config.yaml`:
  - `backend.provider` (`llama_cpp|vllm|ollama|custom`)
  - `backend.upstream_base_url`
  - `backend.upstream_chat_url`
- `llama_swap_url` remains as deprecated fallback for older configs only.
- Shim assets moved to `llama_conductor/shim` and launcher/docs repointed.

---

*** V1.5.4

TL;DR:
- Tightened deterministic source footer normalization for scratchpad-grounded answers.
- Scratchpad provenance remains explicit as `Source: Scratchpad` in normalized non-Mentats footer output.
- Patch release focused on footer consistency and provenance signal hygiene.

---

*** V1.5.3

TL;DR:
- Added deterministic >>faq navigator command for local FAQ navigation in installed builds.
- >>faq provides a numbered section index plus >>faq <n> section print path (webview-safe).
- Expanded FAQ navigator coverage to include key reference/ops sections outside the core FAQ block.
- Simplified core >>help wording to foreground >>faq and reduce operator clutter.

---

*** V1.5.2

TL;DR:
- Added deterministic >>define <word> sidecar for etymology lookup (Etymonline source).
- >>trust now recognizes etymology intent and recommends >>define first when appropriate.
- Trust output formatting is cleaner/scannable, and single-token lexical queries now include a >>define parity option.

---
*** V1.5.1

TL;DR:
Patch over `1.5.0`.
No shiny new toys. This one is about the router not doing dumb loopy stuff when you hit regenerate.

- Fixed deterministic follow-up stability (`you choose`, `idk`, repeat prompts).
- Added replay guard so regenerate keeps deterministic answers instead of falling into model waffle.
- Smoothed replay phrasing:
  - `Same branch: ...` -> `Still the same answer: ...`
- FUN transport kicker punctuation now intentional (`!`) and still zero extra latency.
- Net effect: less drift, less weirdness, same workflow.

---
*** V1.5.0

TL;DR:
Big under-the-hood cleanup release.
Less spaghetti, more structure, same operator-facing behavior.

- Decomposed a large chunk of router internals into clearer modules.
- Hardened deterministic reasoning path for transport/capacity/constraint-style problems.
- Added consistency/replay safeguards to reduce follow-up stochastic drift.
- Continued UX pass on command surfaces (`>>status`, `>>status full`, `>>help full`).
- Tightened public refactor process so private/internal stuff is less likely to leak.
- Net effect: easier to maintain, easier to test, less brittle under iteration.

---
*** V1.3.2

TL;DR
- Vodka recall is now cleaner, stricter, and more predictable under pressure.
- You get simpler controls (`>>preset ...`, `>>memory status`) without losing advanced options.
- Help/footers are less noisy, while fail-loud behavior stays intact.

Highlights
- Added recall presets:
  - `>>preset fast|balanced|max-recall`
  - `>>preset show|set|reset`
- Added memory observability:
  - `>>memory status`
  - active preset shown in `>>status` as `vodka_preset='<name>'`
- Improved help UX:
  - `>>help` = compact operator view
  - `>>help advanced` = full command sheet
- Reduced footer bloat:
  - profile footer line defaults to non-default styles only (`footer.profile.mode: non_default`)
- Unified Fun/Fun Rewrite selection:
  - one deterministic quote-selector core; different renderers preserved
- Canonicalized Vodka memory path:
  - single inlet flow (`capture -> retrieve -> render/inject`)
- Hardened fail-loud recall behavior:
  - uncertainty now returns explicit non-guess outputs
  - fuzzy near-match cases include a clear partial-token-trap signal
- Continued decomposition:
  - recall semantic layer isolated in `llama_conductor/recall_semantic.py`

---
*** V1.3.1 

TL;DR
- Big ticket: Vodka recall was refactored to be deterministic-first, less noisy, and far less likely to drift.
- Recall responses are now more consistent: concise, structured, and grounded to session evidence.
- Follow-up recall turns (for example `Nothing else?`) now resolve deterministically instead of drifting.

Recall contract hardening + global regression gate:
- Enforced deterministic recall output contracts:
  - list queries -> `You substituted the following:` + bullets
  - mention queries -> `Yes, you mentioned:` / explicit `No, ...`
  - mixed queries -> `You promised the following:` + `You also mentioned:`
- Reduced noisy software false positives in recall extraction.
- Added deterministic follow-up handling for shorthand recall turns (`Nothing else?`, `Anything else?`).
- Added a mixed-domain contract smoke gate to catch recall regressions early.
- Current validation target is passing at/above the 90-95% reliability threshold.

---
*** V1.2.5

Session-state interaction profile (deterministic, in-memory MVP):

- Added deterministic per-session interaction profile module: `llama_conductor/interaction_profile.py`.
- Added commands:
  - `>>profile show`
  - `>>profile set <field>=<value>`
  - `>>profile reset`
  - `>>profile on` / `>>profile off`
  - soft aliases: `profile show|set|reset|on|off`
- Added status fields in `>>status`:
  - `profile_enabled`
  - `profile_confidence`
  - `profile_last_updated_turn`
- Added profile reset on `>>detach all`.
- Added style constraints injection for normal + Fun + FR + Raw paths.
  - Style adapter is framing-only and does not alter lock/scratchpad/mentats grounding contracts.
- Added sensitive-context and profanity coercion rules:
  - `ack_reframe_style=feral` is coerced to `sharp` when `profanity_ok=false`.
- Hardened session fallback IDs in `router_fastapi.py`:
  - precedence now `x-chat-id` -> `x-session-id` -> `body.user` -> `sess-<ip>-<proc_counter>`.
- Extended smoke runner with executable PF matrix cases (`PF-001`..`PF-018`):
  - show/set/reset/on/off behavior
  - fail-loud validation checks
  - lock/scratchpad/mentats no-regression checks
  - detach-all reset behavior
- Added sensitive professional-context runtime confirmation gate:
  - prompt: `[router] This request is sensitive in a professional context. Continue anyway? [Y/N]`
  - non-`Y/N` next turn defaults to cancel (`[router] Sensitive request cancelled (default N)`).
- Added per-session nickname disallow memory:
  - detects explicit user disallow phrasing (`don't call me...`, `stop calling me...`, `I am not...`)
  - strips blocked nicknames from response framing.
- Added serious-mode anti-loop guard:
  - allows at most one consecutive meta ack/reframe response
  - repeated ack loop is coerced to deterministic task-forward fallback.
- Added deterministic profile footer on non-Mentats answers:
  - `Profile: <correction_style> | Sarc: <level> | Snark: <level>`
- Updated `>>flush` behavior:
  - still clears CTC cache
  - now also resets session profile/style identity runtime state
  - clears sticky style modes (`fun`, `fr`, `raw`) without detaching KBs.
- Added warm/supportive quote bucket in `llama_conductor/quotes.md`:
  - new tag line: `## warm supportive compassionate hopeful resilient`
  - 20 quotes normalized for punctuation/casing consistency.
- Increased `/serious` default generation cap from `256` to `384` tokens to reduce mid-answer truncation on longer analysis/edit requests.
- Reworked Vodka `enable_summary` path in TESTING:
  - replaced rolling concatenated summary reinjection with deterministic session-memory units
  - writes session-scoped memory units to `total_recall/session_memory/<session_id>.jsonl`
  - updates every `n` user turns (modulo gate), injects only when lexical relevance passes
  - capped injection budget (`summary_inject_max_units`, `summary_inject_max_chars`) to protect latency
- Fun and Fun Rewrite now share the same deterministic profile-aware quote prefilter:
  - both paths use profile state (`correction_style`, `sarcasm_level`, `snark_tolerance`) to constrain quote pool before random pick.
  - no additional model calls added.
- Validation summary:
  - profile/format matrix and FR checks passed.
  - recall quality and output-shape reliability improved versus prior baseline.
  - response stability improved under mixed workload runs.

---

*** V1.2.4

Deterministic confidence/source footer normalization (non-Mentats paths):

- Added `sources_footer.py` and integrated deterministic footer normalization into non-Mentats response paths.
- Footer shape is unchanged:
  - `Confidence: <low|medium|high|top> | Source: <Model|Docs|User|Contextual|Mixed>`
- Footer assignment is now router-rule-based (not model self-grading), using current grounding signals:
  - lock/scratchpad/facts grounding state
  - lock fallback state
  - retrieval hit indicators
- Model fallback confidence is normalized to `unverified` (with explicit `Source: Model`) to signal non-grounded answers without implying correctness.
- Existing explicit provenance lines remain intact:
  - lock grounding: `Source: Locked file (SUMM_<name>.md)`
  - lock fallback: `Source: Model (not in locked file)` + not-found note
  - scratchpad grounding: `Source: Scratchpad`
- Mentats contract is unchanged and still uses `Sources: Vault` (no footer override).
- No new dependencies added.
- Metadata/docs remediation (no command behavior change):
  - Router runtime version reporting now uses single-source `llama_conductor/__about__.py`.
  - `/healthz` version now resolves from `__version__` (no hardcoded drift).
  - `docs/index/DOCS-TRUTH-MAP.md` active list cleaned to remove stale `llama_conductor/SUMM.md`.
  - Added pre-release checker: `tests/pre_release_consistency_check.py` (version/license/docs consistency, fail-loud).

---

*** V1.2.3

TLDR: you can now >>lock a SUMM<name>.md file and reason ONLY over that file, similar to >>scratch pipeline...but without having to copy paste. Should the information NOT be within the locked file, the model will attempt to answer based on pre-trained data and will LOUDLY state "not in locked file; here's my best guess". I hope that this (along with the stdlib summary extraction method) further improves you confidence in provided answers. As always - glassbox, fail states LOUD, trust-but-verify. 

Locked SUMM grounding + lock-safe provenance behavior:

- Added `>>lock SUMM_<name>.md` and `>>unlock` for deterministic single-file SUMM grounding in normal query paths.
- Added `>>list_files` to enumerate lockable `SUMM_*.md` files from attached filesystem KBs.
- Added soft aliases (when a filesystem KB is attached):
  - `lock SUMM_<name>.md` -> `>>lock SUMM_<name>.md`
  - `unlock` -> `>>unlock`
  - `list files` -> `>>list_files` (strict exact phrase)
- Added partial lock confirmation flow:
  - `lock <partial_name>` -> `Did you mean: >>lock SUMM_<name>.md ? [Y/N]`
  - `Y` confirms lock, `N` cancels
- Lock behavior:
  - normal filesystem-grounded queries are scoped to the locked SUMM file only
  - facts are built deterministically from the locked file's `## Extracted Sentences` section
  - source labels normalize to `Source: Locked file (<SUMM_file>)` when grounded
  - when model fallback is used, source label is explicit:
    - `Source: Model (not in locked file)`
    - plus note: `[Not found in locked source <SUMM_file>. Answer based on pre-trained data.]`
- `##mentats` behavior is unchanged and remains Vault-only; lock scope does not affect Mentats.
- `>>detach all` now also clears lock state; `>>detach <kb>` clears lock when the locked file belongs to that KB.
- No SUMM pipeline changes:
  - `>>summ` mechanics and provenance remain unchanged
  - `>>move to vault` mechanics remain unchanged

---

*** V1.2.2

TLDR: >>SUMM is now entirely deterministic and *not* LLM summary. Faster and even more reflective of raw file (albeit somewhat larger).

Deterministic `>>summ` pipeline swap

- Replaced LLM-based summary generation in `>>summ` with deterministic stdlib extractive generation.
- Preserved existing command and pipeline mechanics:
  - `>>summ new|all|<kb>` behavior unchanged
  - `SUMM_<file>.md` creation unchanged
  - original-file move to `/original/` unchanged
  - `>>move to vault` ingestion path unchanged
- Preserved provenance wrapper behavior in generated SUMM files:
  - `source_sha256`
  - `summ_created_utc`
  - `pipeline: SUMM`
- No new dependencies added to `pyproject.toml`.
- Removed legacy `SUMM.md` sentinel dependency from `>>summ` / `>>move` command paths (no behavior change in pipeline mechanics).


---

*** V1.2.1

Scratchpad reliability and safety updates:

- `>>detach scratchpad` now detaches and deletes session scratchpad data, with record count in the response.
- `>>detach all` now also deletes scratchpad data when scratchpad is attached.
- `>>scratchpad delete <query>` now uses token-aware matching (prevents substring over-delete, e.g. `help` no longer deletes `helping`).
- Reasoning over scratchpad now uses full-context mode when a query narrows to exactly one matching record, even if other unrelated records exist.
- Existing exhaustive behavior (`show all` / `all facts`) remains unchanged.

---

*** V1.2.0

V1.2.0: monolithic code base turned modular

- V1.1.6 was centered around a large monolithic router file.
- V1.2.0 is organized into focused modules (commands, config, session state, helpers, model calls, streaming, vault ops), with a lean orchestration router.
- Practical impact:
  - clearer command behavior
  - safer/faster iteration when adding or changing features
  - lower regression risk and easier troubleshooting
- End result: more stability, less borkability

New in V1.2.0: Scratchpad (ephemeral grounded context)

Scratchpad is a session-level, temporary capture store that can be attached as context and used for grounded reasoning during the current session.
This means that you can activate the scratchpad and reason/compare etc over contents thereof. The scratchpad is auto-deleted every 60 minutes.

What users can do:

- Turn on / off:
  - `>>attach scratchpad` (or `>>scratch`)
  - `>>detach scratchpad` (or `>>detach scratch`)

- Inspect and manage captured context:
  - `>>scratch status`
  - `>>scratch list`
  - `>>scratch show [query]`
  - `>>scratch clear` / `>>scratch flush`
  - `>>scratch add <text>`
  - `>>scratch delete <index|query>`

- Operational behavior:
  - selected tool outputs are auto-captured while scratchpad is attached
  - raw captures are stored at `total_recall/session_kb/<session_id>.jsonl`
  - inline command hints now point users to `>>help` for the full list

---

*** V1.1.6

Trust Pipeline Enhancements (Wiki & Exchange Routing)

This release refines the trust pipeline to correctly surface authoritative sidecars for encyclopedic and exchange-style queries, without violating any project invariants.

The trust pipeline has been extended to:

Recommend >>wiki for encyclopedia-shaped questions (people, places, concepts, lore)

Recommend >>exchange for currency conversion / FX-rate queries

Preserve all existing behavior for >>calc, >>weather, ##mentats, KB attachment, and serious mode

No auto-execution was added. All changes remain recommendation-only and fully explicit.


*** (v1.1.5)

***  New Feature: >>trust Mode (Tool Recommendation)

**Tool recommendation pipeline** `>>trust <query>` analyzes your query and suggests the best tools to use
- Recommends ranked options (A, B, C) with confidence levels
- You choose explicitly by typing A, B, or C
- No auto-execution preserves user control
- Helps eliminate tool-selection guesswork

**Examples:**

Math query:
```
>>trust What's 15% of 80?
â€™ A) >>calc (HIGH) - deterministic calculation
â€™ B) serious mode (LOW) - model estimation
```

Factual query:
```
>>trust What software was pivotal for Amiga?
â€™ A) ##mentats (HIGH) - verified reasoning using Vault
â€™ B) >>attach all + query (MEDIUM) - search filesystem KBs
â€™ C) serious mode (LOW) - model knowledge only
```

Complex reasoning:
```
>>trust Compare microservices vs monolithic architecture
â€™ A) ##mentats (HIGH) - 3-pass verification
â€™ B) serious mode with KBs (MEDIUM) - grounded reasoning
â€™ C) serious mode (LOW) - model knowledge
```

**How it works:**
1. Type `>>trust <your question>`
2. System shows ranked tool recommendations
3. Type A/B/C to execute your chosen option
4. Get your answer

**Under the hood:**
- Pattern detection (math, weather, factual, complex reasoning, etc.)
- Resource checking (what KBs are attached)
- Confidence ranking (HIGH/MEDIUM/LOW)
- Zero auto-execution you're always in control

**Design principles:**
-  Explicit control (you choose A/B/C)
-  Router stays dumb (just routes to trust_pipeline)
-  No auto-escalation (suggestions only)
-  Transparent and predictable

**Documentation

**Updated command cheat sheet** Added >>trust documentation under Help & tool selection
**Technical specification** Full architecture and design docs included in release

---

** v1.1.4 (CRITICAL FIX)

*** Critical Bugfix

**Vault attachment prevention** Fixed critical bug where `>>attach vault` was allowed but caused silent failures
- Previously: `>>attach vault` succeeded but Serious mode filtered it out â€™ empty FACTS_BLOCK â€™ hallucinations
- Now: `>>attach vault` properly rejected with helpful error message
- Vault (Qdrant) can only be accessed via `##mentats` (as designed)
- Filesystem KBs (amiga, c64, dogs, etc.) still attach normally via `>>attach <kb>`

**Impact:** Prevents model hallucinations when vault is incorrectly treated as filesystem KB

---

*** v1.1.3

**Code cleanup** Removed legacy fun.py code and reduced duplicate calls

---

** v1.1.2

**Streaming keepalive** Added keepalive pings to prevent streaming timeouts in long-running responses

---

** v1.1.1

** New Sidecars (Deterministic API Tools)

**Wikipedia summaries** `>>wiki <topic>` fetches article openings via free Wikipedia JSON API
- Full paragraph summaries (~500 chars)
- No hallucination, requires valid article name
- Example: `>>wiki Albert Einstein` â€™ full paragraph on relativity, Nobel Prize, etc.

**Currency conversion** `>>exchange <query>` fetches real-time rates via Frankfurter API
- Supports natural language: `>>exchange 1 USD to EUR`, `>>exchange GBP to JPY`
- Common currency aliases (USD, EUR, GBP, JPY, AUD, CAD, etc.)
- Example: `>>exchange 1 AUD to USD` â€™ `0.70 USD`

**Weather** `>>weather <location>` fetches current conditions via Open-Meteo API
- Uses geocoding to find location, then fetches current weather
- Returns: temperature, condition (Clear/Cloudy/Rainy), humidity, wind speed
- No rate limiting (Open-Meteo free tier is generous)
- Works best with city names or "City Country" format
- Example: `>>weather Perth` â€™ `Perth, Australia: 22Ã‚Â°C, Clear sky, 64% humidity`

---

** v1.0.11

***  Core Improvements

**RAW mode context fix** `>>raw` mode now includes CONTEXT block (conversation history) so queries like "what have we discussed?" work correctly instead of hallucinating. Previously `>>raw` was outputting non-Serious answers but without conversation context, causing it to fabricate topics from training data.

---

** v1.0.10

**Fun mode quote pool** Fixed fun_pool to include ALL quotes from all tags instead of just one tone. Fun mode now retrieves actual quotes from quotes.md instead of hallucinating.

**Mentats critic temperature** Added wrapper forcing temperature=0.1 for critic role (Step 2 fact-checking). Eliminates hallucinations where critic accepts made-up facts.

**Vodka warm-up** Eager-load facts.json on Filter initialization instead of lazy-loading. Fixed quirk where `?? list` showed only KBs on first call, then showed full memory store after any write operation.

**Vision mode persistence** Fixed auto-vision detection checking entire history instead of just current message. After `##ocr` or image processing, router now correctly reverts to serious mode on next message.

**Vision/Fun sticky reversion** Fixed sticky modes (`>>fun`, `>>fr`, `>>raw`) now correctly revert when switching modes or completing pipelines.

**Vault chunking optimization** Added configurable chunk size for Qdrant ingestion:
```yaml
vault:
  chunk_words: 250        # Down from 600 (better semantic matching)
  chunk_overlap_words: 50  # Prevents context loss at boundaries
```

---

** v1.0.9-debug

- Fix: Fun/FR blocks after Mentats (checks only recent 5 turns, not all history)
- Fix: Images drop Fun/FR sticky modes and auto-route to vision
- Fix: Mentats selector actually works (was being skipped)
- Fix: Clear errors when conflicts occur

---

** v1.0.4

- Auto-vision detection: Images automatically trigger vision pipeline
- Session commands (>>) skip vision even if image is present (prevents UI attachment bugs)
- All other behavior unchanged

---

** Known Issues / Limitations

** RAG Semantic Retrieval
- e5-small-v2 embedding model struggles with broad, abstract queries
- `##mentats "cultural impact on music"` may return zero facts even if docs contain the info
- **Workaround:** Use more specific queries or `>>find` for exact text matching

** Chunking Trade-offs
- Smaller chunks (250 words) improve semantic matching but increase Qdrant store size
- Larger chunks (600+ words) reduce storage but hurt retrieval precision
- Current setting (250) optimized for medical/knowledge domains

** Weather Locations
- Open-Meteo geocoding works best with city names or "City Country" format
- Long/complex location strings may not resolve (e.g. "Perth Western Australia" â€™ use "Perth")

---

** Older hot-fixes 

**New in v1.1.5:** 
- Tool recommendation pipeline: `>>trust <query>` helps you choose the best tool
- Choose recommendations by typing A, B, or C
- No configuration changes needed works out of the box

**New in v1.1.4:**
- Critical fix: `>>attach vault` now properly rejected (prevents hallucinations)

**New in v1.1.1:** 
- Three new API sidecars: `>>wiki`, `>>exchange`, `>>weather`

**New in v1.2.5 (stability patch):**
- Added fail-soft exception guard for `/v1/chat/completions`:
  - unhandled router exceptions now return a structured router error content instead of transport-level hard failure.
  - traceback is emitted to router console for debugging.
- Replayed `serious7` and `fun7` sequences end-to-end after patch:
  - result: `serious7 16/16`, `fun7 19/19`.

**Profile override fix (manual set precedence):**
- Fixed bug where inferred profile updates could overwrite explicit `>>profile set` values on the next turn.
- Explicitly set fields are now sticky manual overrides until changed again or reset.
- Preset shortcuts (`>>profile override ...` / `>>profile turbo`) now register the same sticky manual overrides.

---

## Questions?

- Run `>>help` in-chat for command reference
- Try `>>trust <your question>` to get tool recommendations
- Check `mentats_debug.log` for deep reasoning traces
- See [FAQ](FAQ.md) for architecture & troubleshooting



