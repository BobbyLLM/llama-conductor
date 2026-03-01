# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)


## Quickstart

### Required stack

llama-conductor is the **router/harness**. It does **not** ship a model, and it does **not** replace your UI.

You need these parts working together:

1) **llama-swap**  https://github.com/mostlygeek/llama-swap
2) **llama.cpp (or other runner)**  https://github.com/ggml-org/llama.cpp
3) **Frontend UI (example: Open WebUI / OWUI)**  https://github.com/open-webui/open-webui
4) **Qdrant (Vault / RAG)**  https://github.com/qdrant/qdrant

**Minimum:** (1) + (2) + (3).

**Core `##mentats` workflows:** require (4) Qdrant. The router can start without Qdrant, but `##mentats` will generally fail closed/refuse due missing Vault facts.

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
    - [1) Context bloat on small hardware](#1-context-bloat-on-small-hardware)
    - [2) Goldfish memory and confident misremembering](#2-goldfish-memory-and-confident-misremembering)
    - [3) Lies, damned lies, and statistics](#3-lies-damned-lies-and-statistics)
    - [4) Grounding drift in normal chat (fixed with >>scratch / >>lock)](#4-grounding-drift-in-normal-chat-fixed-with-scratch--lock)
    - [5) Modes (get the stick out of your LLMs butt)](#5-modes-get-the-stick-out-of-your-llms-butt)
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
  - [How do I verify answers are not bullshit?](#how-do-i-verify-answers-are-not-bullshit)
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
  - [What do these commands *actually* do?](#what-do-these-commands-actually-do)
  - [Deep Example](#deep-example)
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
    - [Mentats refuses to answer your question](#mentats-refuses-to-answer-your-question)
    - [Mentats gives weird answers](#mentats-gives-weird-answers)
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
    - [Vault provenance chain (reader version):](#vault-provenance-chain-reader-version)
  - [TIPS](#tips)
  - [What's new (latest)?](#whats-new-latest)
  - [What does `Profile | Sarc | Snark` mean?](#what-does-profile-sarc-snark-mean)
  - [What do Confidence and Source mean?](#what-do-confidence-and-source-mean)
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

#### 1) Context bloat on small hardware

**WITHOUT llama-conductor:**
```
[400-message chat history]
-> slow generation, degraded recall, dropped setup context, OOM
```

**WITH llama-conductor:**
```
Vodka CTC trims context automatically:
- keeps the recent turns that matter
- hard-caps prompt growth
- drops mid-chat bloat
- keeps memory available through deterministic recall paths
- user definable presets
```

Result:
- Consistent prompt size, stable performance, and optional rolling deterministic summary (stdlib extractive).
- Tok/s you started with, is what you keep (more or less).
- Bonus: Tweak your `--ctx` and maybe your Raspberry Pi can run that 4B model without chug.

---

#### 2) Goldfish memory and confident misremembering

**WITHOUT llama-conductor:**
```
You: Remember my server is 203.0.113.42
[later]
You: What's my server IP?
Model: 127.0.0.1 :P
```

**WITH llama-conductor:**
```
You: !! my server is 203.0.113.42
[later]
You: ?? server ip
Router: 203.0.113.42 [TTL=7 days, Touch=1]
```

Result:
- Router stores and recalls what you said deterministically.
- TTL/touch lifecycle keeps memory useful without silent junkyard bloat.

---

#### 3) Lies, damned lies, and statistics

**WITHOUT llama-conductor:**
```
Start chatting
Ask questions
Model sounds certain
No provenance signal
You guess if grounded vs made up
```

**WITH llama-conductor:**
```
You see:
Confidence: <tier> | Source: <path>
```

Result:
- Router-assigned provenance metadata, not model self-confidence.
- `Source: Model` means fallback route, explicitly labeled.
- Grounded paths are visibly marked as docs/scratch/locked/vault.

---

#### 4) Grounding drift in normal chat (fixed with `>>scratch` / `>>lock`)

**WITHOUT llama-conductor:**
```
You: Summarise this article and tell me what claim got retracted.
Model: blends prior chat junk + generic priors
You: That's not what I asked.
Model: doubles down anyway
```

**WITH llama-conductor:**
```
You: >>scratch
You: [paste article]
You: What claim was retracted? Keep it tight.
Router: [answers from scratch-grounded facts]
Footer: Confidence: high | Source: Scratchpad

# or with curated docs:
You: >>attach <your kb name>
You: >>list_files
You: >>lock SUMM_<name>.md
You: Ask question normally
Router: [answers from locked source or fails loud if missing]
```

Result:
- You can force argument-from-source instead of vibes.
- Missing evidence is surfaced explicitly via source/fallback behavior.

---

#### 5) Modes (get the stick out of your LLMs butt)

**WITHOUT llama-conductor:**
```
One generic answer style
Tone/behavior drift by prompt luck
```

**WITH llama-conductor:**
```
Serious (default): strongest factual discipline
Fun (>>fun / ##fun): quote-anchored style path
Fun Rewrite (>>fr): rewrite-style path over deterministic selector core
Raw (>>raw): pass-through model behavior
```

Result:
- Style controls are explicit and switchable.
- Profile (`Profile | Sarc | Snark`) nudges tone, not grounding contracts.

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
  n_last_messages: 12       # Shipped default: keep last 12 user/assistant pairs
  keep_first: false         # Shipped default: do NOT pin first message
  max_chars: 6000           # Shipped default: hard cap on non-system message chars
```

**Note:** If you run without YAML overrides, internal fallback defaults are lower (`2`, `true`, `1500`).

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
   [ctx:abc123xyz]  "my server is at 203.0.113.42 [TTL=3, touches=1]"
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

Short operational truth:
- It usually runs **three passes** (`thinker -> critic -> thinker rewrite`).
- That is typically **two different model roles** (Thinker model + Critic model), even though Thinker is called twice.
- You *can* map both roles to one model, but separate critic model is usually better for error-catching because you get role diversity instead of self-agreement loops.

Research direction for this design pattern (verifier/critic pass improving reliability):
- `Let's Verify Step by Step` (OpenAI, 2023): https://arxiv.org/abs/2305.20050
- Additional verifier/scaffold reference: https://arxiv.org/abs/2510.26854

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

3. **3-pass workflow** (Thinker  Critic  Thinker):
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
   - Provenance is traceable to retrieved Vault chunks used in the run

**Important:** Mentats/Vault is deep storage, not auto-magic memory.
- You must manually distill content first (`>>summ new`), then promote (`>>move to vault`).
- If you skip that step, Mentats has nothing authoritative to reason over.

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
   Chunks + embeds summaries  Qdrant as `kb="vault"`

3. **Query with Mentats**:
   ```
   ##mentats What does doc X say about Y?
   ```
   Searches Vault, reasons from retrieved chunks

   ```
   ```
   Shows: Vault chunk > SUMM file > original doc > SHA verification

#### Why Vault vs filesystem KBs?

| Feature | Filesystem KBs | Vault |
|---------|----------------|-------|
| Scope | Attached KBs only | All promoted knowledge |
| Speed | File reads | Qdrant vector search |
| Mentats | Ignored | Required |

**Workflow:** Filesystem KBs > SUMM > Vault > Mentats

---

### How do I verify answers are not bullshit?

#### Verify SHA manually:

**Windows:**
```powershell
certutil -hashfile C:\docs\c64\original\c64_facts.pdf SHA256
```

**Linux/Mac:**
```bash
sha256sum /docs/c64/original/c64_facts.pdf
```

**If SHA matches:** Source is authentic and untampered 

**If SHA doesn't match:** Document was modified after SUMM (tampered or legit edit) 

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
- RAG hits are bounded (by default: top_k=6  rerank to 4)

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
- Huggingface is over there  https://huggingface.co/DavidAU ; https://huggingface.co/unsloth ; https://huggingface.co/TheBloke
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
- Default answer mode for day-to-day use: practical, grounded when sources are attached, and explicit when evidence is missing.
- Uses filesystem KBs (if attached)
- Uses Vodka (CTC + TR)
- Includes confidence/source footer line
- Footer is normalized by router rules, not trusted as model self-rating

#### Mentats (##mentats):
- Uses Vault only (ignores filesystem KBs)
- Ignores `>>lock` scope (lock affects normal filesystem-grounded queries only)
- No Vodka, no chat history (isolated)
- Refuses if no Vault facts
- 3-pass reasoning (Thinker > Critic > Thinker)

#### Locked SUMM grounding (`>>lock`):
- `>>lock SUMM_<name>.md` scopes normal grounding to one SUMM file from attached filesystem KBs.
- `>>unlock` clears the lock.
- `>>list_files` shows lockable `SUMM_*.md` files.
- Partial helper works: `lock <partial_name>` -> `Did you mean: >>lock SUMM_<name>.md ? [Y/N]`.
- `>>detach all` clears lock state; `>>detach <kb>` clears lock if that KB owns the locked file.
- Locked supported answer -> `Source: Locked file (SUMM_<name>.md)`.
- Locked unsupported answer -> `Source: Model (not in locked file)` plus explicit not-found note.
- `##mentats` is separate and remains `Sources: Vault` only.
- Full operator detail lives in `llama_conductor/command_cheat_sheet.md`.

#### Fun (##fun or >>fun):
- Runs default (serious) > gets correct answer
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

The scenario hinges on a humorous premise: that yeast can achieve consciousness and form unions. Since this is not grounded in actual legal frameworks or biological consensus, there is no existing statute that defines "firing sentient yeast" as a violation or permits such actions.

Confidence: top | Source: Model

```

#### Fun Rewrite (>>fr):
- Runs default (serious) > gets correct answer
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

### What do these commands *actually* do?

#### `>>attach <kb>` - what is a KB, anyway?

A KB is just a folder path from `kb_paths` in `router_config.yaml`.

`>>attach c64` means: "for normal grounded queries, include retrievable context from that filesystem KB."

It does **not** mean:
- Vault is attached
- everything is now guaranteed true
- Mentats is active

Use `>>list_kb` to see what exists, then `>>attach <your kb name>`.

#### `>>trust <query>` - what does it do?

`>>trust` is a recommendation router, not an auto-execution path.

What it does:
- classify your query deterministically
- suggest ranked command routes (A/B/C/...)
- suggest the best command route (including sidecars)
- keep execution in your control

#### Sidecars (`>>wiki`, `>>define`, `>>exchange`, `>>weather`) - what are they for?

These are deterministic utility tools for common retrieval/conversion tasks:

- `>>wiki <topic>`: quick Wikipedia summary (about 500 characters)
- `>>define <word>`: quick word-origin/etymology lookup (Etymonline)
- `>>exchange <query>`: live currency conversion (Frankfurter API)
- `>>weather <location>`: live weather lookup (Open-Meteo API)

They reduce token spend and avoid unnecessary model improvisation for utility queries.

#### `>>summ` - LLM summary or deterministic extract?

Current behavior is deterministic extractive generation (stdlib path), not free-form LLM summarization.

That means:
- lower hallucination risk than generative summary
- more literal carry-over from source docs
- still lossy (it is a distillation step), so critical claims should be verified from source when stakes are high
- As proof of concept: compare [Guardian source](https://tinyurl.com/GuardianMuskArticle) vs [SUMM_musk-article.md](SUMM_musk-article.md)


### But why even use `>>summ`? Why not reason against raw text?

- Less noise = more signal = better answers
- Speed

In testing, >>SUMM managed to distil key concepts from raw text, to an *approximate* 85-90% coverage (across multiple test samples). Exact % depends on complexity and how noisy the raw text is. Meaning, >>SUMM  is not flawless, but good enough for government work. 

See "proof of concept" above as an example of raw source and generated >>summ thereon, using QWEN3-4B HIVEMIND.

Remember, this is deterministic summary (not stochastic LLM magic). There will always be rough edges...but they are consistent and known.

But hey, if you really want to argue against raw, you can!
- generate the >>summ
- keep the SHA
- Delete the actually keypoints and copy back the raw text

This sort of defeats the purpose of SUMM as a distilled artifact. But if that's what you want, go for it

Else...may I suggest looking closely at the >>scratch section? You may like what you see :) 



#### `>>lock SUMM_<name>.md` - what does it prevent?

`>>lock` forces normal grounding to one SUMM file. 

It prevents:
- accidental retrieval blending across unrelated files
- hidden context drift from other attached KB docs
- "it kinda answered from somewhere" ambiguity

If support is missing in the locked file, router should fail loud and label fallback source. Put up or shut up.

#### `>>scratch` / scratchpad - what does it prevent?

Scratchpad gives session-local grounded notes for multi-step reasoning.

It prevents:
- losing critical interim facts across turns
- long-chat memory drift on working variables
- repeated copy/paste of the same context block

Think of it as temporary working memory, not permanent truth storage. However, as the user copy pastes content into scratchpad (or invokes it via `>>wiki` or other pipelines) it is a 1:1 lossless representation. This allows you to do some *very* interesting tricks, outlined below.

---

### Deep Example

Source article:
- Original: https://tinyurl.com/GuardianMuskArticle
- >>scratch then >>add (and copy paste the text in)

`Example prompts this stack can handle when grounded correctly`

```text
User:
List the article's concrete claims about permit status and turbine operations, each with support.

Assistant:
Claims grounded in the loaded source:
- EPA position: these turbines require permits under the Clean Air Act.
- Mississippi regulators: portable/mobile classification means temporary permit exemption.
- Floodlight thermal imagery + public records indicate active turbine usage before/while permit disputes continued.

Support excerpts:
- "On 15 January, the EPA reiterated its decades-old policy that such machines need a permit."
- "Floodlight's thermal imagery ... shows 15 unpermitted turbines in operation at the Southaven facility."
- "Public records obtained by Floodlight show 18 of the 27 turbines have been used since November, at least."

Confidence: high | Source: Scratchpad
Profile: neutral | Sarc: off | Snark: low
```

```text
User:
Given only this article, what is the strongest and weakest inference you can make?

Assistant:
Strongest inference:
- There is an active regulatory interpretation conflict (federal-vs-state) while turbines are operating at scale.

Weakest inference (flagged as weak):
- Any definitive claim that operations are fully lawful or fully unlawful in final adjudicated terms.

Why:
- The first is directly supported by the source's side-by-side EPA and state positions.
- The second requires formal enforcement/court outcomes not present in this file.

Confidence: medium | Source: Scratchpad
Profile: neutral | Sarc: off | Snark: low
```

```text
User:
What motives are implied, and which are explicitly unsupported?

Assistant:
Implied motives (bounded):
- Fast infrastructure scaling for AI datacenter demand.
- Preference for operational continuity while permitting questions are unresolved.

Explicitly unsupported in this source:
- Any definitive internal intent memo from xAI leadership.
- Any claim of malicious intent beyond reported operational/regulatory behavior.

I can infer plausible intent signals, but I will not promote those to fact without direct evidence.

Confidence: medium | Source: Scratchpad
Profile: direct | Sarc: off | Snark: low
```

```text
User:
Which claims were later revised or contradicted? Split confirmed vs disputed.

Assistant:
Confirmed in this source:
- EPA reiterated permit-required policy on 15 January.
- Source reports thermal evidence of unpermitted turbines operating in Southaven.
- Source reports state regulator position that units are permit-exempt under portable/mobile classification.

Disputed / revised / unresolved:
- Final enforcement outcome and legal resolution.
- Whether state interpretation remains durable under federal enforcement pressure.

Confidence: high | Source: Scratchpad
Profile: neutral | Sarc: low | Snark: low
```

NB:
- Your mileage varies by model quality. Qwen-3-4B HIVEMIND (and larger) are comically good for this use case. See benchmarks on HF.
- For this style of work, prioritize models that perform well on reasoning + factuality benchmarks (for example `GPQA`, `MMLU-Pro`, `TruthfulQA`) and then test them in your own grounded matrix.
- These examples match expected router output shape and grounded behavior. Pick a good GGUF! 
---

### Common workflows

#### Adding new knowledge:
```
1. Drop files in KB folder (C:/docs/myKB/)
2. >>attach myKB
3. >>summ new
4. >>lock SUMM_<name>.md. Query to your heart's content.
4. Happy? >>move to vault
5. Later, ##mentats <ask question> 
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

### Technical Setup

#### Architecture Overview

**The stack:**
```
Backend [llama.cpp + llama-swap] + llama-conductor + Frontend [OWUI/SillyTavern/LibreChat]
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


**This is *a lot* of moving parts. What's the ABSOLUTE MINIMUM STACK to use as proof of concept?**

Fine, you're lazy / overwhelmed. I get it :) 

Here's the MVP. 

- Something to launch your LLM / GGUF (eg: llama.cpp)
- A front end (sadly, llama-server alone is model backend, not a chat front end for router output, so you need *something* minimal. GPTChat? LibreChat? Your choice. You could even get it to work via CMD or Powershell, but if you're reading this section, I don't imagine that's your vibe)
- Configure the router-config.yaml so that EVERYTHING runs against ONE declared llm name (eg: thinker, critic etc are ALL the same and point to whatever you've called your GGUF).
- Launch everything per the .bat file example below

This will give you about...60% of the experience. You'll be missing a LOT (mentats, OCR etc) but for playing around with...ok, try.

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
nb: adjust filepaths to your actual.


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

#### Mentats refuses to answer your question 
- Did you >>add to vault?
- Run `>>peek <query>` to see what would be retrieved
- Rememner: the pathways is Add docs to KB, `>>summ new`, `>>move to vault`
- ##Mentats has STRICT recall schema. Check the mentats.log for failure reason

#### Mentats gives  weird answers
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

These are examples / what I use. Feel free to edit YOURS to your use case 


#### Vodka (memory + context):
```yaml
vodka:
  preset: "balanced"           # fast|balanced|max-recall
  n_last_messages: 12          # Last N user/assistant pairs to keep (shipped default)
  keep_first: false            # Keep first message (system prompt)
  max_chars: 6000              # Hard cap on message chars
  enable_summary: true         # Session-memory units pipeline (not old noisy blob reinjection)
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
  vector_name: "e5_small_v2"
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

### Advanced: How does provenance work?

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

#### Vault provenance chain (reader version):

When a chunk lands in Vault, it carries traceable source pointers:

```json
{
  "kb": "vault",
  "text": "<chunk>",
  "source_kb": "c64",
  "source_rel_path": "SUMM_doc.md",
  "file": "SUMM_doc.md",
  "path": "C:/docs/c64/SUMM_doc.md"
}
```

`SUMM_*.md` files keep the deeper source metadata (including SHA) in their headers, so the practical audit trail is:

```
Answer -> Vault chunk -> SUMM file -> source metadata header
```

**Why this matters:**
- You can trace where claims came from.
- You can verify source lineage from SUMM headers.
- You can debug bad answers against concrete chunks.

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

- Always check `NEW.md` for most recent update 

- NEW: Core router internals were decomposed into focused modules (`chat_*`, `state_*`, `router_*`) to reduce monolith risk and make behavior easier to maintain.
- IMPROVED: Correction/follow-up handling is more stable on tricky "I meant X not Y" turns, with cleaner pass-through when a query falls outside deterministic scope.
- IMPROVED: Consistency guardrail now catches obvious state/constraint contradictions before final output.
- IMPROVED: Public release staging is now standardized, with stricter promotion checks before shipping.
- STABLE: Existing grounding contracts (`>>lock`, `>>scratch`, Mentats/Vault, confidence/source footer) are preserved.

### What does `Profile | Sarc | Snark` mean?

You may see a footer like:

`Profile: neutral | Sarc: medium | Snark: high`

This is style metadata for the current session. It is not a truth score.

Default display policy:
- By default, profile footer is shown only when style is non-default.
- Default baseline (`neutral | off | low`) keeps only the main `Confidence | Source` footer line.
- If you want always-on profile footer for diagnostics, use `footer.profile.mode: always` in config.

#### `Profile` (how direct the assistant sounds)

- `softened`: gentler tone, more cushioning language, less blunt correction
- `neutral`: plain professional tone, balanced wording
- `direct`: concise, blunt, minimal padding, faster correction/reframe

#### `Sarc` (how ironic/wry the wording is)

- `off`: no sarcasm
- `low`: light dry humor
- `medium`: noticeable irony
- `high`: strong sarcastic framing (still bounded by safety/context rules)

#### `Snark` (how sharp/combative the edge is)

- `low`: polite and restrained
- `medium`: firmer, more pointed
- `high`: sharp, spicy phrasing allowed (especially when user style supports it)

### How state changes over time

- The profile updates from your turn-by-turn style signals.
- Strong explicit signals move it faster than weak implicit ones.
- Repeated signals increase confidence.
- Lack of reinforcement causes decay toward baseline.
- `>>detach all` and `>>flush` reset profile/style runtime state.

### Output impact

Higher `direct` / `sarc` / `snark` usually means:

- shorter acknowledgements
- faster correction + reframe
- less hedging
- more edge in wording

Lower settings usually means:

- softer transitions
- less irony
- more neutral/professional phrasing

Mode leverage:

- `serious`: lowest style impact; factual/grounding constraints dominate
- `raw`: medium style impact; less formatting constraint than serious
- `fun`: higher style impact (quote + tone shaping)
- `fr` (fun rewrite): highest style impact (tone rewrite path)

Grounding/provenance rules are unchanged by these style states.

### Manual control and shortcuts

You can override inference at any time:

- `>>profile show`
- `>>profile set correction_style=direct`
- `>>profile set sarcasm_level=medium`
- `>>profile set snark_tolerance=high`
- `>>profile reset`
- `>>profile on` / `>>profile off`

Shortcut mappings (what you type as short cut vs what effect it has):

- `>>profile casual` = `correction_style=direct`, `verbosity=compact`, `snark_tolerance=high`, `sarcasm_level=medium`, `profanity_ok=false`
- `>>profile feral` = `correction_style=direct`, `verbosity=compact`, `snark_tolerance=high`, `sarcasm_level=high`, `profanity_ok=true`
- `>>profile turbo` = same as `>>profile feral`

nb: The effects of profile on different settings ("serious", "fun", "fun rewrite") vary.

### What do Confidence and Source mean?

Short answer: the router assigns these labels, not the model.
They show how grounded an answer is, not how confident the model sounds.

#### Source = where the answer came from

- `Source: Model`
  - No grounded evidence was found. This is pre-trained model knowledge.
- `Source: Docs`
  - Grounded to attached filesystem docs / SUMM facts.
- `Source: Scratchpad`
  - Grounded to attached scratchpad content.
- `Source: Locked file (SUMM_*.md)`
  - Grounded to the currently locked SUMM file.
- `Source: Model (not in locked file)`
  - You were in lock mode, but the locked file did not contain the answer, so model fallback was used.
- `Sources: Vault`
  - Mentats path: grounded through Vault retrieval/reasoning flow.

#### Confidence = strength of grounding signals

- `unverified`
  - Model fallback with no grounded support. Best-effort answer.
- `low` / `medium` / `high` / `top`
  - Router-assigned confidence tiers from grounding quality/signals (not model self-confidence).

#### How confidence is assigned (plain English)

The router checks evidence signals, then assigns a tier:

- Strong grounded evidence present (locked SUMM / scratchpad / docs with clear hits) -> `high` to `top`
- Some grounding signal but partial/weaker support -> `medium`
- Thin grounding, ambiguous support, or weak retrieval -> `low`
- No grounded support and fallback to model knowledge -> `unverified`

So confidence means:

- "how much verifiable support do we have for this answer in this path?"
- not "how confident did the model sound?"

#### Why this footer exists

The footer makes provenance visible at a glance, so you can quickly decide whether to:

- accept
- verify
- or push for stricter grounding (`>>lock`, `>>scratch`, `##mentats`)

### Questions and Help

What, this wasn't enough? May God have mercy on your soul.

Submit a ticket. I read and respond once a week (usually Sunday or Mondays) between 11-1pm UTC. 

Else, you can roll the dice: sanding-oink-grant(AT)duck(DOT)com or find me on Lemmy.

Sorry - real life commitments :( 

## License
AGPL-3.0-or-later. See `LICENSE`


