# MoA Router Command Cheat Sheet (v1.0.3.1)

This router exposes an OpenAI-style `/v1/chat/completions` endpoint.

## Core idea
You are routing **workflows**, not models.

- **Serious** is default (boring, reliable).
- **Mentats** is deliberate reasoning, **Vault-only**, isolated.
- **Fun** is post-processing style only.
- **Fun Rewrite (FR)** is style rewrite with a tone-matched seed quote.

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

**Invariant:** If you invoke `##mentats` while Fun/FR is active, the router will hard-disable Fun/FR and run Mentats in isolation.

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

## Per-turn selectors (one message) — start with `##`

### Mentats (Vault-only)
- `##mentats <question>`
- aliases: `##m`, `##zardoz`, `##z`

Mentats invariants:
- Detaches all KBs.
- Auto-attaches Vault for the run, then detaches.
- Must not receive Vodka or chat history.
- If Vault returns no facts for the query, Mentats refuses cleanly.
- Mentats always logs raw step outputs to `mentats_debug.log` (in the router working directory).

### Fun (one turn)
- `##fun <question>` / `##f <question>` — answer in Serious, then style it.

---

## Config (`router_config.yaml`)

- `kb_paths:` map of KB name → folder path
- `vault_kb_name:` name of the Vault KB (Mentats uses this)
- `rag:` Qdrant + embedder + reranker settings
- `vodka:` TTL/touch + context shaping settings

---

## Output markers

- Fun outputs include: `[FUN] "<seed quote>"`
- Fun Rewrite outputs include: `[FUN REWRITE] "<seed quote>"`
- Mentats outputs include: `[ZARDOZ HATH SPOKEN]` and `Sources: Vault`
