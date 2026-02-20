# What's New

*** V1.2.4 (latest)

Deterministic confidence/source footer normalization:

- Added `sources_footer.py` and integrated deterministic footer normalization.
- Footer shape is unchanged:
  - `Confidence: <low|medium|high|top> | Source: <Model|Docs|User|Contextual|Mixed>`
- Footer assignment is now *router-rule-based* (not model self-grading), using current grounding signals:
  - lock/scratchpad/facts grounding state
  - lock fallback state
  - retrieval hit indicators
- Model fallback confidence is normalized to `medium` (with explicit `Source: Model`) to avoid pessimistic under-confidence on known-stable facts.
- Existing explicit provenance lines remain intact:
  - lock grounding: `Source: Locked file (SUMM_<name>.md)`
  - lock fallback: `Source: Model (not in locked file)` + not-found note
  - scratchpad grounding: `Source: Scratchpad`
- Mentats contract is unchanged and still uses `Sources: Vault` (no footer override).
- No new dependencies added.
- **TLDR:Confidence now represents DIRECT sourcing, per above. No more 'good enough' indicators

---

*** V1.2.3 (latest)

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
  - ** TLDR: you can now >>lock a SUMM<name>.md file and reason ONLY over that file, similar to >>scratch pipeline...but without having to copy paste. Should the information NOT be within the locked file, the model will attempt to answer based on pre-trained data and will LOUDLY state "not in locked file; here's my best guess". I hope that this (along with the stdlib summary extraction method) further improves you confidence in provided answers. As always - glassbox, fail states LOUD, trust-but-verify. 

---

*** V1.2.2

Deterministic `>>summ` pipeline swap (surgical refactor):

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
- `SUMM.md` compatibility checks remain in place to avoid command-path behavior drift.
-  ** TLDR: >>SUMM is now entirely deterministic and *not* LLM summary. Faster and even more reflective of raw file (albeit somewhat larger).

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


## (v1.1.5)

### 🎯 New Feature: >>trust Mode (Tool Recommendation)

**Tool recommendation pipeline** — `>>trust <query>` analyzes your query and suggests the best tools to use
- Recommends ranked options (A, B, C) with confidence levels
- You choose explicitly by typing A, B, or C
- No auto-execution — preserves user control
- Helps eliminate tool-selection guesswork

**Examples:**

Math query:
```
>>trust What's 15% of 80?
→ A) >>calc (HIGH) - deterministic calculation
→ B) serious mode (LOW) - model estimation
```

Factual query:
```
>>trust What software was pivotal for Amiga?
→ A) ##mentats (HIGH) - verified reasoning using Vault
→ B) >>attach all + query (MEDIUM) - search filesystem KBs
→ C) serious mode (LOW) - model knowledge only
```

Complex reasoning:
```
>>trust Compare microservices vs monolithic architecture
→ A) ##mentats (HIGH) - 3-pass verification
→ B) serious mode with KBs (MEDIUM) - grounded reasoning
→ C) serious mode (LOW) - model knowledge
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
- Zero auto-execution — you're always in control

**Design principles:**
- ✅ Explicit control (you choose A/B/C)
- ✅ Router stays dumb (just routes to trust_pipeline)
- ✅ No auto-escalation (suggestions only)
- ✅ Transparent and predictable

### 📚 Documentation

**Updated command cheat sheet** — Added >>trust documentation under Help & tool selection
**Technical specification** — Full architecture and design docs included in release

---

## v1.1.4 (CRITICAL FIX)

### 🚨 Critical Bugfix

**Vault attachment prevention** — Fixed critical bug where `>>attach vault` was allowed but caused silent failures
- Previously: `>>attach vault` succeeded but Serious mode filtered it out → empty FACTS_BLOCK → hallucinations
- Now: `>>attach vault` properly rejected with helpful error message
- Vault (Qdrant) can only be accessed via `##mentats` (as designed)
- Filesystem KBs (amiga, c64, dogs, etc.) still attach normally via `>>attach <kb>`

**Impact:** Prevents model hallucinations when vault is incorrectly treated as filesystem KB

---

## v1.1.3

**Code cleanup** — Removed legacy fun.py code and reduced duplicate calls

---

## v1.1.2

**Streaming keepalive** — Added keepalive pings to prevent streaming timeouts in long-running responses

---

## v1.1.1

### 🔍 New Sidecars (Deterministic API Tools)

**Wikipedia summaries** — `>>wiki <topic>` fetches article openings via free Wikipedia JSON API
- Full paragraph summaries (~500 chars)
- No hallucination, requires valid article name
- Example: `>>wiki Albert Einstein` → full paragraph on relativity, Nobel Prize, etc.

**Currency conversion** — `>>exchange <query>` fetches real-time rates via Frankfurter API
- Supports natural language: `>>exchange 1 USD to EUR`, `>>exchange GBP to JPY`
- Common currency aliases (USD, EUR, GBP, JPY, AUD, CAD, etc.)
- Example: `>>exchange 1 AUD to USD` → `0.70 USD`

**Weather** — `>>weather <location>` fetches current conditions via Open-Meteo API
- Uses geocoding to find location, then fetches current weather
- Returns: temperature, condition (Clear/Cloudy/Rainy), humidity, wind speed
- No rate limiting (Open-Meteo free tier is generous)
- Works best with city names or "City Country" format
- Example: `>>weather Perth` → `Perth, Australia: 22°C, Clear sky, 64% humidity`

---

## v1.0.11

### 🎯 Core Improvements

**RAW mode context fix** — `>>raw` mode now includes CONTEXT block (conversation history) so queries like "what have we discussed?" work correctly instead of hallucinating. Previously `>>raw` was outputting non-Serious answers but without conversation context, causing it to fabricate topics from training data.

---

## v1.0.10

**Fun mode quote pool** — Fixed fun_pool to include ALL quotes from all tags instead of just one tone. Fun mode now retrieves actual quotes from quotes.md instead of hallucinating.

**Mentats critic temperature** — Added wrapper forcing temperature=0.1 for critic role (Step 2 fact-checking). Eliminates hallucinations where critic accepts made-up facts.

**Vodka warm-up** — Eager-load facts.json on Filter initialization instead of lazy-loading. Fixed quirk where `?? list` showed only KBs on first call, then showed full memory store after any write operation.

**Vision mode persistence** — Fixed auto-vision detection checking entire history instead of just current message. After `##ocr` or image processing, router now correctly reverts to serious mode on next message.

**Vision/Fun sticky reversion** — Fixed sticky modes (`>>fun`, `>>fr`, `>>raw`) now correctly revert when switching modes or completing pipelines.

**Vault chunking optimization** — Added configurable chunk size for Qdrant ingestion:
```yaml
vault:
  chunk_words: 250        # Down from 600 (better semantic matching)
  chunk_overlap_words: 50  # Prevents context loss at boundaries
```

---

## v1.0.9-debug

- Fix: Fun/FR blocks after Mentats (checks only recent 5 turns, not all history)
- Fix: Images drop Fun/FR sticky modes and auto-route to vision
- Fix: Mentats selector actually works (was being skipped)
- Fix: Clear errors when conflicts occur

---

## v1.0.4

- Auto-vision detection: Images automatically trigger vision pipeline
- Session commands (>>) skip vision even if image is present (prevents UI attachment bugs)
- All other behavior unchanged

---

## Known Issues / Limitations

### RAG Semantic Retrieval
- e5-small-v2 embedding model struggles with broad, abstract queries
- `##mentats "cultural impact on music"` may return zero facts even if docs contain the info
- **Workaround:** Use more specific queries or `>>find` for exact text matching

### Chunking Trade-offs
- Smaller chunks (250 words) improve semantic matching but increase Qdrant store size
- Larger chunks (600+ words) reduce storage but hurt retrieval precision
- Current setting (250) optimized for medical/knowledge domains

### Weather Locations
- Open-Meteo geocoding works best with city names or "City Country" format
- Long/complex location strings may not resolve (e.g. "Perth Western Australia" → use "Perth")

---

## Roadmap (Ideas)

- [ ] Better embedding model (nomic-embed-text or bge-large-en-v1.5)
- [ ] User-provided reference documents without summarization
- [ ] Session-level fact deduplication
- [ ] Multi-language support (Vodka memory)
- [ ] Web UI integration (hooks for `>>` and `??` commands)
- [ ] More sidecars (>>define, >>convert units, etc.)

---

## Installation / Upgrading

```bash
# Fresh install
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git

# Upgrade existing
pip install --upgrade git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

**New in v1.1.5:** 
- Tool recommendation pipeline: `>>trust <query>` helps you choose the best tool
- Choose recommendations by typing A, B, or C
- No configuration changes needed — works out of the box

**New in v1.1.4:**
- Critical fix: `>>attach vault` now properly rejected (prevents hallucinations)

**New in v1.1.1:** 
- Three new API sidecars: `>>wiki`, `>>exchange`, `>>weather`

---

## Questions?

- Run `>>help` in-chat for full command reference
- Try `>>trust <your question>` to get tool recommendations
- Check `mentats_debug.log` for deep reasoning traces
- See [FAQ](FAQ.md) for architecture & troubleshooting
