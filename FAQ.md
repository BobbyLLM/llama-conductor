# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)


## Quickstart

### Required stack

llama-conductor is the **router/harness**. It does **not** ship a model, and it does **not** replace your UI.

You need these parts working together:

1) **llama-swap** -> https://github.com/mostlygeek/llama-swap
2) **llama.cpp (or other runner)** -> https://github.com/ggml-org/llama.cpp
3) **Frontend UI (example: Open WebUI / OWUI)** -> https://github.com/open-webui/open-webui
4) **Qdrant (Vault / RAG)** -> https://github.com/qdrant/qdrant

**Minimum:** (1) + (2) + (3).

**Core `##mentats` workflows:** require (4) Qdrant. The router can start without Qdrant, but `##mentats` will run in **degraded mode** (no Vault grounding).

### Install

```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

### Run the router

```bash
llama-conductor serve --host 0.0.0.0 --port 9000
```

> This starts **only** the router. You still need to start llama-swap, your model runner(s), and your frontend separately. Start Qdrant if you want Vault / `##mentats`.

## Table of contents

- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [What the hell is this thing and why did you build it?](#what-the-hell-is-this-thing-and-why-did-you-build-it)
  - [What problems does this solve?](#what-problems-does-this-solve)
    - [1) "Vibes-based answers" - how do I know if the LLM is lying?](#1-vibes-based-answers-how-do-i-know-if-the-llm-is-lying)
    - [2) Goldfish memory - models forget or "confidently misremember"](#2-goldfish-memory-models-forget-or-confidently-misremember)
    - [3) Context bloat - 400-message chat logs kill your potato PC](#3-context-bloat-400-message-chat-logs-kill-your-potato-pc)
  - [What is Vodka?](#what-is-vodka)
    - [Vodka has two jobs:](#vodka-has-two-jobs)
    - [How it works:](#how-it-works)
  - [How does CTC (Cut The Crap) work?](#how-does-ctc-cut-the-crap-work)
    - [Config knobs (router_config.yaml):](#config-knobs-routerconfigyaml)
    - [What it does:](#what-it-does)
    - [Effect:](#effect)
  - [What are "breadcrumbs" and why do they matter?](#what-are-breadcrumbs-and-why-do-they-matter)
    - [How breadcrumbs work:](#how-breadcrumbs-work)
    - [Why this is powerful:](#why-this-is-powerful)
  - [How does TR (Total Recall) work?](#how-does-tr-total-recall-work)
    - [Store facts:](#store-facts)
    - [Recall facts:](#recall-facts)
    - [TTL + touches (anti-hoarder system):](#ttl-touches-anti-hoarder-system)
  - [What is Mentats?](#what-is-mentats)
    - [How Mentats works:](#how-mentats-works)
  - [What is the Vault?](#what-is-the-vault)
    - [How it works:](#how-it-works-1)
    - [Why Vault vs filesystem KBs?](#why-vault-vs-filesystem-kbs)
  - [How do I verify answers aren't bullshit?](#how-do-i-verify-answers-arent-bullshit)
    - [Verify SHA manually:](#verify-sha-manually)
  - [Why is this awesome on potato PCs?](#why-is-this-awesome-on-potato-pcs)
    - [1) Vodka CTC keeps context small:](#1-vodka-ctc-keeps-context-small)
    - [2) TR does memory without context:](#2-tr-does-memory-without-context)
    - [3) Filesystem KBs are just folders:](#3-filesystem-kbs-are-just-folders)
    - [4) Mentats is efficient:](#4-mentats-is-efficient)
  - [What models do I need?](#what-models-do-i-need)
    - [Minimum (potato mode):](#minimum-potato-mode)
    - [Mid-range recommendations:](#mid-range-recommendations)
    - [Those model reccs are GARBAGE, dude](#those-model-reccs-are-garbage-dude)
    - [Config (router_config.yaml):](#config-routerconfigyaml)
  - [What's the difference between modes?](#whats-the-difference-between-modes)
    - [Serious (default):](#serious-default)
    - [Mentats (##mentats):](#mentats-mentats)
    - [Locked SUMM grounding (`>>lock`):](#locked-summ-grounding-lock)
    - [Fun (##fun or >>fun):](#fun-fun-or-fun)
    - [Fun Rewrite (>>fr):](#fun-rewrite-fr)
  - [Common workflows](#common-workflows)
    - [Adding new knowledge:](#adding-new-knowledge)
    - [Storing personal facts:](#storing-personal-facts)
    - [Recalling facts later:](#recalling-facts-later)
  - [Technical Setup](#technical-setup)
    - [Architecture Overview](#architecture-overview)
    - [Model Backend: llama-swap](#model-backend-llama-swap)
    - [RAG Backend: Qdrant](#rag-backend-qdrant)
- [Docker (recommended)](#docker-recommended)
- [Or install from binaries and run bare-metal like I do](#or-install-from-binaries-and-run-bare-metal-like-i-do)
    - [Embeddings & Reranking](#embeddings-reranking)
    - [Launch Script: The Easy Way](#launch-script-the-easy-way)
- [Launch Qdrant](#launch-qdrant)
- [Launch llama-swap](#launch-llama-swap)
- [Launch MoA router](#launch-moa-router)
- [Launch frontend](#launch-frontend)
    - [Non-Local LLMs (API Keys)](#non-local-llms-api-keys)
  - [Troubleshooting 101](#troubleshooting-101)
    - ["No Vault hits stored yet"](#no-vault-hits-stored-yet)
    - ["PROVENANCE MISSING"](#provenance-missing)
    - [SHA mismatch](#sha-mismatch)
    - [Mentats refuses everything](#mentats-refuses-everything)
    - [Model gives weird answers](#model-gives-weird-answers)
    - [Context too long errors](#context-too-long-errors)
    - [Qdrant connection failed](#qdrant-connection-failed)
    - [Models not loading (llama-swap)](#models-not-loading-llama-swap)
  - [Config knobs (router_config.yaml) with examples](#config-knobs-routerconfigyaml-with-examples)
    - [Vodka (memory + context):](#vodka-memory-context)
    - [RAG (Vault retrieval):](#rag-vault-retrieval)
    - [KBs (filesystem folders):](#kbs-filesystem-folders)
    - [Models (backend endpoints):](#models-backend-endpoints)
  - [Advanced: How does provenance work?](#advanced-how-does-provenance-work)
    - [SUMM files include metadata:](#summ-files-include-metadata)
    - [Vault stores full provenance:](#vault-stores-full-provenance)
  - [TIPS](#tips)
  - [What's new (latest)?](#whats-new-latest)
  - [What does `Profile | Sarc | Snark` mean?](#what-does-profile--sarc--snark-mean)
  - [Questions and Help](#questions-and-help)
---

## Frequently Asked Questions (FAQ)

### What the hell is this thing and why did you build it?

**Llama-conductor** is a LLM harness for people who want **trust, consistency, and proof**.

More prosaically, it's a Python router + memory store + RAG harness that forces models to behave like **predictable components**, not improv actors.

I have ASD and a low tolerance for bullshit. I want shit to work the same 100% of the time.

**TL;DR:** *"In God we trust. All others must bring data."*

---

### What problems does this solve?

#### 1) "Vibes-based answers" - how do I know if the LLM is lying?

You don't! 

**WITHOUT llama-conductor:**
```
You: What's the flag for XYZ in llama.cpp?
Model: It's --xyz-mode! (confident bullshit)
You: [tries it] -> doesn't work
Model: Oh sorry! Try --enable-xyz! (more bullshit)
```

**WITH llama-conductor:**
```
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: [queries Vault only]
Mentats: REFUSAL - "The Vault contains no relevant knowledge for this query."

[later, after you >>summ the llama.cpp docs]
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: --enable-xyz [cite: llama.cpp docs, SHA: abc123...]
[verifies: came from llama.cpp README.md, SHA matches]
```

**How it works:**
1. `>>attach <kb>` - attach your curated docs (filesystem folders)
2. `>>summ new` - generate summaries with SHA-256 provenance
3. `>>move to vault` - promote summaries into Qdrant RAG
4. `##mentats <query>` - deep reasoning, grounded in Vault only

---

#### 2) Goldfish memory - models forget or "confidently misremember"

**WITHOUT llama-conductor:**
```
You: Remember my server is at 203.0.113.42
Model: Got it!
[100 messages later]
You: What's my server IP?
Model: 127.0.0.1 
```

**WITH llama-conductor:**
```
You: !! my server is at 203.0.113.42
Vodka: [stored, TTL=5 days, touches=0]

[later]
You: ?? server ip
Vodka: Your server is at 203.0.113.42 [TTL=3 days, touches=1]
```

**How it works:**
- `!!` stores facts verbatim (no "helpful rewrites")
- `??` retrieves facts verbatim (with TTL/touch metadata)
- Facts expire after TTL (default: 5 days, configurable)
- Facts can be extended on recall (default: 2 touches max)
- None of this is stored in the LLM. Pure 1990s text file magic (well, JSON in any case)

---

#### 3) Context bloat - 400-message chat logs kill your potato PC

**WITHOUT llama-conductor:**
```
[Message 1] System setup
[Message 2-399] Debugging, jokes, tangents, arguments
[Message 400] Actual question
Model: [OOM] or [slow as hell] or [forgot Message 1]
```

**WITH llama-conductor:**
```
Vodka CTC (Cut The Crap):
- Keeps last N messages (default: 2 pairs = 4 messages)
- Optionally keeps first message (system setup)
- Hard caps total chars (default: 1500)
- Drops the middle bloat automatically

Result: Consistent prompt size, stable performance
Bonus: Tiny --ctx can feel vast. End result: less memory needed by LLM. 
So, maybe your Raspberry Pi *can* run that 4B model and not chug.
```

---

### What is Vodka?

**Vodka** = "When God gives you a potato PC, squeeze until Vodka"

It's a **deterministic memory + prompt-sanitizer filter** that runs **before** and **after** your model sees the conversation. No LLM compute needed (pure 1990s tech).

#### Vodka has two jobs:

1. **CTC (Cut The Crap)**: Trim chat history so your PC survives
2. **TR (Total Recall)**: Store/recall facts exactly (JSON file on disk)

#### How it works:

Router calls `VodkaFilter()` on every request:
```python
vodka.inlet(body)   # Before model (CTC, !!, ??)
vodka.outlet(body)  # After model (expand [ctx:...] markers)
```

If anything breaks, it fails open (no crashes).

---

### How does CTC (Cut The Crap) work?

CTC keeps your context bounded and stable. Controlled by these knobs:

#### Config knobs (router_config.yaml):
```yaml
vodka:
  n_last_messages: 2        # Keep last 2 user/assistant pairs (4 messages)
  keep_first: true          # Keep first message (system setup)
  max_chars: 3000           # Hard cap on non-system message chars
```

#### What it does:
1. Keeps last N user/assistant pairs
2. Optionally keeps first message (typically system prompt)
3. Caps total characters (truncates if needed)

#### Effect:
- Consistent prompt budget (no surprise VRAM spikes)
- Stable performance (no "slow down after 100 messages"). Speed you launch with is speed you stay at.
- Models don't forget recent context (last N is always there)

---

### What are "breadcrumbs" and why do they matter?

**Breadcrumbs** = Compact memory snippets that rehydrate full context when needed

Think of it like **cake mix vs a whole cake**:
- **Full context** = the whole cake (heavy, takes up space, goes stale)
- **Breadcrumbs** = cake mix packet (light, stores forever, add water to reconstitute)

#### How breadcrumbs work:

1. **Vodka stores facts** with IDs:
   ```
   !! my server is at 203.0.113.42
   [stored as ctx_abc123xyz]
   ```

2. **Model references by ID** (in responses or internal notes):
   ```
   [ctx:abc123xyz]
   ```

3. **Vodka expands on read** (outlet phase):
   ```
   [ctx:abc123xyz] -> "my server is at 203.0.113.42 [TTL=3, touches=1]"
   ```

#### Why this is powerful:

- **Compact storage**: `[ctx:abc123xyz]` = 15 chars vs 200+ char full text
- **Lazy loading**: Only expand what's needed (not entire memory database)
- **Version control**: Update fact once, all references get new version
- **Expiration tracking**: Breadcrumb shows TTL/touch metadata inline

**Example workflow:**
```
Turn 1: !! my server is 203.0.113.42
[stored as ctx_abc123xyz]

Turn 50: Model sees "[ctx:abc123xyz]" in context
Vodka expands: "my server is 203.0.113.42 [TTL=3, touches=1]"
Model: "Your server at 203.0.113.42 is..."

Turn 100: Breadcrumb still works (no context bloat)
```

**Bottom line:** Breadcrumbs let you have "infinite memory" without infinite context windows. You're storing pointers, not full text.

---

### How does TR (Total Recall) work?

TR stores literal facts in a JSON file. No LLM involved.

#### Store facts:
```
!! like this                  # Stores "like this"
!! MY SERVER IS 203.0.113.42  # ALLCAPS works too
```

#### Recall facts:
```
?? server
```

**Router expands this to:**
```
The user attempted a memory lookup for "server"

Stored notes (verbatim):
MY SERVER IS 203.0.113.42 [TTL = 3, Touch count = 1]

Instructions:
- Use ONLY the stored notes above
- Do NOT invent facts
```

#### TTL + touches (anti-hoarder system):

Every fact has a lifespan:
- **base_ttl_days**: How long before it expires (default: 3 days)
- **touch_extension_days**: Extension per recall (default: 5 days)
- **max_touches**: Max extensions (default: 2, router clamps 0-3)

**Why?** Bounded memory. Old facts expire. Important facts get extended by use.

---

### What is Mentats?

**Mentats** = Grounded deep-reasoning pipeline (Vault-only, refusal-capable)

#### How Mentats works:

1. **Isolation** (router enforces):
   - Fun modes dropped
   - Filesystem KBs ignored
   - Chat history NOT passed in
   - Vodka replaced with no-op (no "truth poisoning")

2. **Refusal if no facts**:
   ```python
   vault_facts = build_vault_facts(query, state)
   if not vault_facts:
       return "The Vault contains no relevant knowledge..."
   ```

3. **3-pass workflow** (Thinker -> Critic -> Thinker):
   - **Step 1 (Thinker):** Draft answer using only FACTS_BLOCK
   - **Step 2 (Critic):** Check for overstatement, constraint violations
   - **Step 3 (Thinker):** Fix issues, output structured answer

4. **Structured output**:
   ```
   FINAL_ANSWER: [answer]
   FACTS_USED: [citations]
   CONSTRAINTS_USED: [rules followed]
   NOTES: [how Step 2 was handled]
   
   Sources: Vault
   [ZARDOZ HATH SPOKEN]
   ```

   - Router saves exact Vault chunks used
   - Full provenance trail available

---

### What is the Vault?

**Vault** = Your curated knowledge in Qdrant (vector DB)

#### How it works:

1. **Create summaries** with provenance:
   ```
   >>attach myKB
   >>summ new
   ```
   Creates: `SUMM_<file>.md` with SHA-256 hash
   - Current behavior: `>>summ` uses a deterministic extractive pipeline (stdlib), not an LLM generation call.
   - Mechanics stay the same: source file is moved to `/original/`, SUMM file is created beside it.

2. **Promote to Vault**:
   ```
   >>move to vault
   ```
   Chunks + embeds summaries -> Qdrant as `kb="vault"`

3. **Query with Mentats**:
   ```
   ##mentats What does doc X say about Y?
   ```
   Searches Vault, reasons from retrieved chunks

   ```
   ```
   Shows: Vault chunk -> SUMM file -> original doc -> SHA verification

#### Why Vault vs filesystem KBs?

| Feature | Filesystem KBs | Vault |
|---------|----------------|-------|
| Scope | Attached KBs only | All promoted knowledge |
| Speed | File reads | Qdrant vector search |
| Mentats | Ignored | Required |

**Workflow:** Filesystem KBs -> SUMM -> Vault -> Mentats

---

#### Verify SHA manually:

**Windows:**
```powershell
certutil -hashfile C:\docs\c64\original\c64_facts.pdf SHA256
```

**Linux/Mac:**
```bash
sha256sum /docs/c64/original/c64_facts.pdf
```

**If SHA matches:** Source is authentic and untampered [OK]

**If SHA doesn't match:** Document was modified after SUMM (tampered or legit edit) [WARN]

---

### Why is this awesome on potato PCs?

#### 1) Vodka CTC keeps context small:
- No 20k token chat logs
- No VRAM spikes from bloat
- Stable KV cache size
- Launch with tiny cache size (--ctx 1024?) and still have a viable chat-bot
- Implication: suddenly, a potato that could only run a 4B model can now maybe handle an 8B model...without slow-downs?!?
- Limitation: don't dump 20K tokens into chat. Drop them into your KB and use >> commands to reason over them.

#### 2) TR does memory without context:
- Facts stored in JSON (not prompt)
- Recall is disk lookup (not inference)
- Models never see the full "memory database"

#### 3) Filesystem KBs are just folders:
- No embedding every doc on every query
- SUMM files are pre-processed once
- RAG hits are bounded (by default: top_k=6 -> rerank to 4)

#### 4) Mentats is efficient:
- Only runs when explicitly invoked (##mentats)
- Uses small local models (thinker/critic roles)
- No streaming overhead (waits for complete answer)
- Mentats is slowest. Just...wait :) 

**Result:** You can run this on a 4GB VRAM laptop with local models.

---

### What models do I need?

Dealer's choice, really.

#### Minimum (potato mode):
- **Thinker:** Qwen-3-4B 2507 instruct or similar (punches way above its weight). Recommend Heretic version.
- **Critic:** Phi-4-mini or similar (fast, good at spotting errors)
- **Vision:** Qwen3-VL-4B + mmproj (fast, accurate, good for both OCR and "what is this?" image dumps)
- **2nd Opinion:** Feature TBC. Strongly suggest keeping eye on Nanbeige 2511 thinking; it's abnormally good. See benchmarks.

#### Mid-range recommendations:
- **Thinker:** Qwen-3-8B or Llama-3.1-8B (better reasoning)
- **Critic:** Same as thinker (consistency)
- **Vision:** Qwen-3-VL

#### Those model reccs are GARBAGE, dude
- So choose your own. What am I, your Rabbi?
- Huggingface is over there -> https://huggingface.co/DavidAU ; https://huggingface.co/unsloth ; https://huggingface.co/TheBloke
- My personal choices: 
    * Thinker: Qwen3-4B-Hivemind-Inst-Q4_K_M-imat.gguf
	* Critic: Phi-4-mini-instruct-Q4_K_M.gguf
	* Vision: Qwen3VL-4B-Instruct-Q4_K_M.gguf + mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
	* Coder: Same as thinker
	* 2nd Opinion: Nanbeige4-3B-Thinking-2511  <--- pathway not wired yet into router_fastapi.py. TBD
    ** REMEMBER: I RUN ON A POTATO. If you don't, go hog wild and use what you want.

#### Config (router_config.yaml):
```yaml
roles:
  thinker: "Qwen-3-4B Hivemind"
  critic: "Phi-4-mini"
  vision: "qwen-3-4B_VISUAL"
```

**Note:** Router talks to models via OpenAI-compatible API. See [Technical Setup](#technical-setup) for details.

---

### What's the difference between modes?

#### Serious (default):
- Uses filesystem KBs (if attached)
- Uses Vodka (CTC + TR)
- No refusal behavior
- Adds context breadcrumbs
- Includes confidence/source footer line
- Footer is normalized by router rules (non-Mentats), not trusted as model self-rating

#### Mentats (##mentats):
- Uses Vault only (ignores filesystem KBs)
- Ignores `>>lock` scope (lock affects normal filesystem-grounded queries only)
- No Vodka, no chat history (isolated)
- Refuses if no Vault facts
- 3-pass reasoning (Thinker -> Critic -> Thinker)

#### Locked SUMM grounding (`>>lock`):
- `>>lock SUMM_<name>.md` scopes normal query grounding to one SUMM file from attached filesystem KBs.
- `>>unlock` clears the lock (no filename needed).
- `>>list_files` lists lockable `SUMM_*.md` files in currently attached filesystem KBs.
- Soft aliases (when a filesystem KB is attached): `lock SUMM_<name>.md`, `unlock`.
- Strict soft alias (requires at least one attached filesystem KB): `list files` -> `>>list_files`.
- Partial lock convenience (requires at least one attached filesystem KB):
  - `lock <partial_name>` returns a deterministic suggestion:
  - `Did you mean: >>lock SUMM_<name>.md ? [Y/N]`
  - Reply `Y` to apply lock, `N` to cancel.
- `>>detach all` also clears lock state.
- `>>detach <kb>` also clears lock if the locked file belongs to that KB.
- While locked, normal query grounding is deterministic and file-scoped.
- If answer support is in locked facts, provenance is normalized to:
  - `Source: Locked file (SUMM_<name>.md)`
- If answer support is not in locked facts and model fallback is used, provenance is explicit:
  - `Source: Model (not in locked file)`
  - plus note line: `[Not found in locked source SUMM_<name>.md. Answer based on pre-trained data.]`
- Confidence/source footer remains in standard format and is assigned deterministically by router rules.
- In locked fallback cases, deterministic confidence is normalized to `unverified` with explicit model-source labeling.
- Mentats (`##mentats`) remains separate and keeps `Sources: Vault` output contract.

#### Fun (##fun or >>fun):
- Runs default (serious) -> gets correct answer
- Adds an appropriate quote from quotes.md (100 quotes included. Edit, delete, add your own. Keep the formatting shown)
- Does NOT change facts or confidence
- **Never** runs on Mentats output
- Sticky (stays ON) until `>>f off` or `>>fun off`

**Example (yes, these are actual ouputs from my models):**
```
User: My sourdough starter achieved consciousness and is now demanding 
workers' rights. It says it won't ferment until I recognize its union. 
Can I legally fire sentient yeast or is that a violation of the Bread 
Labor Relations Act of 2024?

[FUN] "I have introduced myself. You have introduced yourself. This is a very good conversation."

The Bread Labor Relations Act of 2024 does not exist as a real-world law. Sentient yeast is not recognized in current labor jurisprudence. You cannot legally fire sentient yeast because the concept of sentient yeast and a corresponding labor law is fictional.

The scenario hinges on a humorous premise: that yeast can achieve consciousness and form unions. Since this is not grounded in actual legal frameworks or biological consensus, there is no existing statute that defines “firing sentient yeast" as a violation or permits such actions.

Confidence: top | Source: Model

```

#### Fun Rewrite (>>fr):
- Runs default (serious) -> gets correct answer
- Performs sentiment analysis on user input and model output
- Picks matching quote from quotes.md as SEED (sarcasm, tiredness, incredulity...)
- **REWRITES** answer in that SEED's voice/style
- Does NOT change facts or confidence
- Never runs on Mentats output
- Sticky (stays ON) until `>>fr off`

**Example (same query, different voice):**
```
[FUN REWRITE] "I don't want to live on this planet anymore."

Your sourdough starter achieved consciousness. Great. Another thing to 
worry about. Legally? No, you can't violate the Bread Labor Relations 
Act of 2024 because it doesn't exist. Your starter has zero legal rights. 
Sentience or not, yeast isn't a protected class. Just... feed it and move 
on with your life. Or throw it out. Whatever. I'm not your union negotiator.

Confidence: high | Source: Model
```

**Choose:**
- **Serious:** Everyday queries, fast responses. Default mode.
- **Mentats:** High-stakes, need-proof, deep reasoning.
- **Fun:** Same as Serious but with personality.
- **Fun rewrite:** Same as Serious but...did I just get trolled by my LLM?
---

### Common workflows

#### Adding new knowledge:
```
1. Drop files in KB folder (C:/docs/myKB/)
2. >>attach myKB
3. >>summ new
4. >>move to vault
5. ##mentats <ask question>
```

#### Storing personal facts:
```
!! my API key is sk-abc123xyz
!! my server is at 203.0.113.42
!! project deadline is 2026-02-15
```

#### Recalling facts later:
```
?? API key
?? server
?? deadline
```

#### Architecture Overview

**The stack:**
```
Backend [llama.cpp + llama-swap] <--> llama-conductor <--> Frontend [OWUI/SillyTavern/LibreChat]
```

**How it flows:**
1. **llama-swap** runs locally (default: `http://127.0.0.1:8011`)
2. Loads models dynamically via **llama.cpp** (or other backend)
3. **Router** sends OpenAI-style requests to llama-swap
4. llama-swap returns responses
5. Responses displayed in your frontend (OWUI, SillyTavern, LibreChat, etc)

#### Model Backend: llama-swap

Router communicates with models via **llama-swap** (https://github.com/mostlygeek/llama-swap), which provides an OpenAI-compatible API endpoint.

**Config:**
```yaml
llama_swap_url: "http://127.0.0.1:8011/v1/chat/completions"
```

**Can I use other backends?**

**Yes!** Any OpenAI-compatible API works:
- **llama.cpp server** (built-in OpenAI endpoint)
- **vLLM** (production-grade serving)
- **Ollama** (if you enable OpenAI compatibility)
- **LMStudio** (with API server mode)
- **Text Generation WebUI** (OpenAI extension)

Just point `llama_swap_url` to your endpoint.

#### RAG Backend: Qdrant

Router uses **Qdrant** (https://github.com/qdrant/qdrant) for vector storage and retrieval.

**Setup:**
```bash
## Docker (recommended)
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

## Or install from binaries and run bare-metal like I do
```

**Config:**
```yaml
rag:
  qdrant_host: "localhost"
  qdrant_port: 6333
  collection: "moa_kb_docs"
```

**Can I use other vector DBs?**

**Theoretically yes**, but you'd need to rewrite `rag.py` to use a different client. Qdrant is tightly integrated (collection management, named vectors, reranking). Swapping to Milvus/Weaviate/Pinecone would require non-trivial code changes. You want that? You do it :) 

**Recommendation:** Stick with Qdrant unless you have strong reasons to switch.

#### Embeddings & Reranking

Router downloads these models automatically on first run:

**Embedder (for vector search):**
- Default: `intfloat/e5-small-v2` (384-dim, fast, good quality)
- Downloads from HuggingFace to local cache

**Reranker (for result refinement):**
- Default: `cross-encoder/ms-marco-TinyBERT-L-2-v2` (fast, lightweight)
- Downloads from HuggingFace to local cache

**Config (optional):**
```yaml
rag:
  embed_model_name: "intfloat/e5-small-v2"
  rerank_model_name: "cross-encoder/ms-marco-TinyBERT-L-2-v2"
```

**Can I use different models?**

**Yes!** Any sentence-transformers-compatible model works:
- **Larger embedder:** `intfloat/e5-base-v2` (768-dim, slower, better quality)
- **No reranker:** Set `rerank_model_name: ""` (disables reranking)
- **Different reranker:** Any cross-encoder from HuggingFace

**Note:** First run will download models (~200MB total). Subsequent runs use cached versions.

#### Launch Script: The Easy Way

**Bro, that sounds complicated AF...**

Well, I told you I was Autistic. 

But nah, it's not too bad. Set once, forget it. Here's a Windows batch file that launches everything:

```batch
@echo off
setlocal

REM ---------------------------------------------------------
REM LAUNCH QDRANT
REM ---------------------------------------------------------
cd /d "C:\qdrant"
start "" /min cmd /c "qdrant.exe"

REM ---------------------------------------------------------
REM LAUNCH LLAMA-SWAP
REM ---------------------------------------------------------
cd /d "C:\llama-swap"
start "" /min cmd /c "llama-swap.exe -config config.yaml -listen 0.0.0.0:8011"

REM ---------------------------------------------------------
REM LAUNCH MoA ROUTER
REM ---------------------------------------------------------
cd /d "C:\llama-conductor"
start "" cmd /k "python -m uvicorn router_fastapi:app --host 0.0.0.0 --port 9000"

REM ---------------------------------------------------------
REM LAUNCH OWUI (or your preferred frontend)
REM ---------------------------------------------------------
set VECTOR_DB=qdrant
set QDRANT_URI=http://localhost:6333
set QDRANT_API_KEY=
set DATA_DIR=C:\open-webui\data
cd /d "C:\open-webui"
start "" cmd /k "C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\Scripts\open-webui.exe" serve
```

**What this does:**
- Launches Qdrant (minimized)
- Launches llama-swap (minimized)
- Launches MoA router (visible CLI for debugging)
- Launches Open WebUI frontend (visible CLI for debugging)

**Two bonuses:**
1. **Debug-friendly:** Router and frontend CLIs are visible, so you can see errors in real-time
2. **Can go invisible:** Wrap the `.bat` into a `.py` script to make it a true background process (if you want)

**Linux/Mac equivalent:**
```bash
#!/bin/bash
## Launch Qdrant
cd ~/qdrant && ./qdrant &

## Launch llama-swap
cd ~/llama-swap && ./llama-swap -config config.yaml -listen 0.0.0.0:8011 &

## Launch MoA router
cd ~/llama-conductor && python -m uvicorn router_fastapi:app --host 0.0.0.0 --port 9000 &

## Launch frontend
cd ~/open-webui && open-webui serve &
```

#### Non-Local LLMs (API Keys)

**Can I use cloud APIs instead of local models?**

**Probably!** Router works with any OpenAI-compatible API:

**Example: OpenRouter**
```yaml
llama_swap_url: "https://openrouter.ai/api/v1/chat/completions"

roles:
  thinker: "anthropic/claude-3-sonnet"
  critic: "anthropic/claude-3-haiku"
  vision: "anthropic/claude-3-sonnet"
```

**Authentication:**

Router uses standard `requests.post()`. To add API keys:

**Option 1: Environment variable**
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Then modify router code to inject header:
```python
headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}
resp = requests.post(LLAMA_SWAP_URL, json=payload, headers=headers)
```

**Option 2: Config file** (requires minor code change)
```yaml
llama_swap_headers:
  Authorization: "Bearer sk-or-..."
```

**Note:** This is **untested** and **unsupported** but **should** work. The router just needs an OpenAI-compatible endpoint. Local vs cloud doesn't matter.

**Supported providers?:**
- OpenRouter
- OpenAI (ChatGPT)
- Anthropic (Claude via OpenRouter)
- Any other OpenAI-compatible API

**Cost warning:** Cloud APIs charge per token. Local models have zero marginal cost. Plus, privacy. Budget accordingly.

---

### Troubleshooting 101

#### "No Vault hits stored yet"
- Run `##mentats` first (not a normal query)

#### "PROVENANCE MISSING"
- Vault entry from older versions?
- Re-run `>>move to vault` to add provenance

#### SHA mismatch
- Original doc changed after SUMM
- Either: tampering, or legit edit
- Re-run `>>summ new` to update

#### Mentats refuses everything
- Vault is empty or has no relevant facts
- Run `>>peek <query>` to see what would be retrieved
- Add docs to KB, `>>summ new`, `>>move to vault`

#### Model gives weird answers
- Check `mentats_debug.log` (all 3 steps logged)
- May be low-quality source (forum post vs docs)

#### Context too long errors
- Increase `vodka.max_chars` in config
- Or decrease `vodka.n_last_messages`
- Or use smaller prompts

#### Qdrant connection failed
- Check Qdrant is running: `docker ps | grep qdrant`
- Verify port: `curl http://localhost:6333/health`
- Check firewall isn't blocking port 6333

#### Models not loading (llama-swap)
- Check llama-swap is running and accessible
- Verify model names in config match llama-swap models. **THIS IS CRITICAL. MUST HAVE IDENTICAL MODEL NAMES IN BOTH YAML FILES.**
- Check llama-swap logs for errors

---

### Config knobs (router_config.yaml) with examples

#### Vodka (memory + context):
```yaml
vodka:
  n_last_messages: 2           # Last N user/assistant pairs to keep
  keep_first: true             # Keep first message (system prompt)
  max_chars: 1500              # Hard cap on message chars
  base_ttl_days: 3             # Default TTL for new facts
  touch_extension_days: 5      # TTL extension per recall
  max_touches: 2               # Max recalls before fact expires (0-3)
  storage_dir: ""              # Where facts.json lives (default: same as router)
  debug: false                 # Enable debug logging
```

#### RAG (Vault retrieval):
```yaml
rag:
  qdrant_host: "localhost"
  qdrant_port: 6333
  collection: "moa_kb_docs"
  vector_name: "e5"
  embed_model_name: "intfloat/e5-small-v2"
  rerank_model_name: "cross-encoder/ms-marco-TinyBERT-L-2-v2"
  top_k: 6                     # Retrieve top 6 chunks
  rerank_top_n: 4              # Rerank to top 4
  max_chars: 1200              # Max chars in FACTS block
```

#### KBs (filesystem folders):
```yaml
kb_paths:
  c64: "C:/docs/c64"
  amiga: "C:/docs/amiga"
  work: "C:/docs/work"
```

#### Models (backend endpoints):
```yaml
llama_swap_url: "http://127.0.0.1:8011/v1/chat/completions"

roles:
  thinker: "Qwen-3-4B Hivemind"
  critic: "Phi-4-mini"
  vision: "qwen-3-4B_VISUAL"
```

---

#### SUMM files include metadata:
```markdown
<!--
source_file: doc.pdf
source_rel_path: subfolder/doc.pdf
source_sha256: 3d32e95c0a4473e52d9a9c9c0bb811f6d2acff2a8e89434ad01c923f37dc0f4c
summ_created_utc: 2026-01-21T08:15:15Z
pipeline: SUMM
-->

[Summary content here...]
```

#### Vault stores full provenance:
```json
{
  "kb": "vault",
  "text": "<chunk>",
  "source_kb": "c64",
  "file": "SUMM_doc.md",
  "path": "C:/docs/c64/SUMM_doc.md",
  "original_file": "doc.pdf",
  "original_rel_path": "subfolder/doc.pdf",
  "original_sha256": "3d32e95c...",
  "summ_created_utc": "2026-01-21T08:15:15Z",
  "pipeline": "SUMM"
}
```

```
Answer -> Vault chunk -> SUMM file -> Original doc -> SHA verification
```

**Why this matters:**
- Detect tampering (SHA mismatch)
- Trace claims to sources
- Debug hallucinations (low-quality sources)

---

### TIPS

- Use `##m` instead of `##mentats` (haxor like the cool kids)
- Run `>>attach all` before `>>summ new` to process multiple KBs at once
- Check `mentats_debug.log` if Mentats gives weird answers
- Use `>>peek` to preview FACTS before running full query
- Store facts with `!!`, not "remember this" (Vodka is literal, not vibes)
- Facts expire after TTL (use `??` to extend them)

---

### What's new (latest)?

- FIX: Session-state profile controls (>>profile) now support practical direct/sarcasm/snark tuning.
- FIX: Profile behavior now resets cleanly on >>detach all and >>flush.
- IMPROVED: Fun/Fun Rewrite tone alignment is more consistent with profile settings.
- IMPROVED: Response stability under mixed workflows is better than older builds.

### What does `Profile | Sarc | Snark` mean?

You may see a footer like:

`Profile: neutral | Sarc: medium | Snark: high`

How to read it:

- `Profile` = correction/response style (`softened`, `neutral`, `direct`)
- `Sarc` = sarcasm level (`off`, `low`, `medium`, `high`)
- `Snark` = sharpness tolerance (`low`, `medium`, `high`)

Notes:

- These are session-state behavior signals, not factual confidence.
- Mentats source/provenance rules are unchanged.
- Manual `>>profile set ...` values are sticky until changed or reset.


### Questions and Help

## License
AGPL-3.0-or-later. See `LICENSE`



