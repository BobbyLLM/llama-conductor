#  Router Command Cheat Sheet 


## Sidecar utilities (deterministic, no LLM) — start with `>>`

### Calculator
- `>>calc <expression>` — evaluate math expressions safely
  - Supports: `+`, `-`, `*`, `/`, `%`, `**`, parentheses, functions (`sqrt`, `sin`, `cos`, `log`, `floor`, `ceil`, `abs`, `round`, `min`, `max`, `pow`)
  - Natural language: `>>calc 30% of 79.95` → `23.99`
  - Natural language: `>>calc 14 per 20` → `0.70`

### Memory management
- `>>list` — list all stored Vodka memories with metadata (TTL, touch count, creation date)
- `>>flush` — reset the CTC (Cut-The-Crap) message history cache for next turn

### Knowledge search
- `>>find <query>` — search attached filesystem KBs for exact text matches
  - Returns file location, line number, and snippet context
  - Example: `>>find unit price` → finds "unit price" in KB files
  - Requires: KBs must be attached first with `>>attach <kb>`

---

## Session commands (sticky) — start with `>>`

### Help / status
- `>>help` — show this cheat sheet in-chat
- `>>status` — show session state (sticky modes, attachments, last RAG stats)

### Fun modes (sticky)
- `>>fun` / `>>f` / `>>F` — enable sticky Fun mode
- `>>fun off` / `>>f off` / `>>F off` — disable sticky Fun mode

### Fun Rewrite (FR) modes (sticky)
- `>>fun_rewrite` / `>>FR` / `>>fr` — enable sticky Fun Rewrite mode
- `>>fun_rewrite off` / `>>FR off` / `>>fr off` — disable sticky Fun Rewrite mode

**Nb:** If you invoke `##mentats` while Fun/FR is active, the router will hard-disable Fun/FR and run Mentats in isolation.

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
  - “new” means: files that are **not** already named `SUMM_*`.
  - Uses provenance + SHA-based dedupe rules from the pipeline.

> Note: Mentats does **not** stream status. Await response, mortal!

### Vault promotion (Qdrant)
- `>>move to vault` — take `SUMM_*.md` from **all attached KBs**, chunk+embed, and write them into Qdrant as `kb="vault"`.

### RAG peek (debug)
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

### Option B: OCR/
- `>>OCR` / `>>read` -> OCR extract text from image (if present) 

- `image present, no vision command given` -> Defaults to caption/OCR first, then runs selected pipeline

---

## Output markers

- Fun outputs include: `[FUN] "<seed quote>"`
- Fun Rewrite outputs include: `[FUN REWRITE] "<seed quote>"`
- Mentats outputs include: `[ZARDOZ HATH SPOKEN]` and `Sources: Vault`
