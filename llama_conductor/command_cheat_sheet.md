#  Router Command Cheat Sheet 


## Sidecar utilities (deterministic, no LLM) — start with `>>`

### Calculator
- `>>calc <expression>` — evaluate math expressions safely
  - Supports: `+`, `-`, `*`, `/`, `%`, `**`, parentheses, functions (`sqrt`, `sin`, `cos`, `log`, `floor`, `ceil`, `abs`, `round`, `min`, `max`, `pow`)
  - Natural language: `>>calc 30% of 79.95` → `23.99`
  - Natural language: `>>calc 14 per 20` → `0.70`

### Knowledge lookup (free APIs, no auth)
- `>>wiki <topic>` — Wikipedia summary (via free JSON API, max 500 chars)
  - Example: `>>wiki Albert Einstein` → opening paragraph + title
  - Works best with: single topics or well-known names
  - Deterministic (no hallucination), context-light
  - Note: Requires valid Wikipedia article name (e.g. "Albert Einstein" not "albert einstein physicist")
  
- `>>exchange <query>` — Currency conversion (via Frankfurter API, real-time rates)
  - Examples: `>>exchange 1 USD to EUR`, `>>exchange GBP to JPY`, `>>exchange convert AUD to CAD`
  - Returns: `1.0 USD = 0.92 EUR`
  - Supports common currency aliases: USD, EUR, GBP, JPY, AUD, CAD, etc.
  
- `>>weather <location>` — Current weather (via Open-Meteo, compact format)
  - Examples: `>>weather Perth`, `>>weather London`, `>>weather New York`
  - Works best with: city names or short location names (single word, or "City Country")
  - Returns: `Perth: 22°C, Partly cloudy`
  - Note: Long location strings may timeout. Use city name only if possible.

### Memory & search
- `>>list` — list all stored Vodka memories with metadata (TTL, touch count, creation date)
- `>>flush` — reset the CTC (Cut-The-Crap) message history cache for next turn
- `>>find <query>` — search attached filesystem KBs for exact text matches
  - Returns file location, line number, and snippet context
  - Example: `>>find unit price` → finds "unit price" in KB files
  - Requires: KBs must be attached first with `>>attach <kb>`

---

## Session commands (sticky) — start with `>>`

### Help & tool selection
- `>>help` — show this cheat sheet in-chat
- `>>status` — show session state (sticky modes, attachments, last RAG stats)
- `>>trust <query>` — **tool recommendation** (analyzes query, suggests best tools)
  - Shows ranked options (A, B, C) with confidence levels
  - Type A/B/C to execute chosen recommendation
  - Example: `>>trust What's 15% of 80?` → suggests `>>calc` (A) or serious mode (B)
  - Helps you choose the right tool without guessing

### Fun modes (sticky)
- `>>fun` / `>>f` / `>>F` — enable sticky Fun mode
- `>>fun off` / `>>f off` / `>>F off` — disable sticky Fun mode

### Fun Rewrite (FR) modes (sticky)
- `>>fun_rewrite` / `>>FR` / `>>fr` — enable sticky Fun Rewrite mode
- `>>fun_rewrite off` / `>>FR off` / `>>fr off` — disable sticky Fun Rewrite mode

**Nb:** If you invoke `##mentats` while Fun/FR is active, the router will hard-disable Fun/FR and run Mentats in isolation.

### Raw mode (sticky)
- `>>raw` — enable sticky Raw mode (answers without Serious formatting, includes context)
- `>>raw off` — disable sticky Raw mode

### KB attachment (filesystem KBs)
- `>>list_kb` — list known KB names + attached KBs
- `>>attach <kb>` — attach a KB for this session
- `>>attach all` — attach all known KBs
- `>>detach <kb>` — detach a specific KB
- `>>detach all` — detach all KBs

### SUMM workflow (KB curation)
You keep KBs as **folders of files** on disk.

**Supported raw file types for summarization:** `.md`, `.txt`, `.pdf`, `.htm`, `.html`

**What happens when you SUMM:**
- Raw files in the KB folder are summarized using `SUMM.md`.
- A `SUMM_<filename>.md` is produced in the KB folder.
- The original raw file is moved into `<kb_folder>/original/`.
- Retrieval from KBs uses `SUMM_*.md` (not the raw docs).

**Commands:**
- `>>summ new` (recommended; command matching is case-insensitive) — summarize NEW raw files in all currently attached KBs
  - "new" means: files that are **not** already named `SUMM_*`.
  - Uses provenance + SHA-based dedupe rules from the pipeline.

> Note: Mentats does **not** stream status. Await response, mortal!

### Vault promotion (Qdrant)
- `>>move to vault` — take `SUMM_*.md` from **all attached KBs**, chunk+embed, and write them into Qdrant as `kb="vault"`.

### KB peek (assumes KBs attached)
- `>>peek <query>` — preview the would-be FACTS block (debugging)

---

## Vodka memory commands (per-turn control) — start with `!!` or `??`

### Store & manage
- `!! <text>` — manually store text as a Vodka memory (highlight for later recall)
- `!! nuke` / `nuke !!` — **delete all Vodka memories** (hard reset, early return)
- `!! forget <query>` — delete Vodka memories matching the query text

### Query & expand
- `?? <query>` — search Vodka memories, rewrite your question using matched facts as context
  - Model sees stored facts injected into the message
  - Useful for: follow-ups, referencing prior notes
- `?? list` — list all stored Vodka memories with metadata (TTL, touch count, creation date)

**Important distinction:**
- `!! nuke` — deletes **Vodka memory store** (persistent facts you've saved)
- `>>flush` — resets **CTC cache** (temporary message history trimming); does NOT delete memories

---

## Per-turn selectors (one message) — start with `##`

### Mentats (Vault-only)
- `##mentats <question>`
- alias: `##m`

Mentats function:
- Detaches all KBs.
- Auto-attaches Vault for the run, then detaches.
- Must not receive Vodka or chat history.
- If Vault returns no facts for the query, Mentats refuses cleanly.
- Mentats always logs raw step outputs to `mentats_debug.log` (in the router working directory).

### Fun (one turn)
- `##fun <question>` / `##f <question>` — answer in Serious, then style it.

---

## Vision / Screenshot commands (per-turn; works when an image is present)

### Option A: Direct VLM (routes straight to roles.vision)
- `>>vision` / `>>vl` / `>>v` + <text> -> Direct vision answer (VLM sees the image + your question; answer based on image)

### Option B: OCR
- `>>OCR` / `>>read` -> OCR extract text from image (if present) 

- `image present, no vision command given` -> Defaults to caption/OCR first, then runs selected pipeline

---

## Output markers

- `[trust]` — Tool recommendations from >>trust mode
- `[FUN]` — Fun mode outputs (with seed quote)
- `[FUN REWRITE]` — Fun Rewrite outputs (with seed quote)
- `[ZARDOZ HATH SPOKEN]` — Mentats outputs (with Sources: Vault)
- `[calc]`, `[wiki]`, `[exchange]`, `[weather]`, `[find]`, `[list]`, `[flush]` — Sidecar outputs
