# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)


## Quickstart

### Required stack

`llama-conductor` is the router/harness. It does not ship models.

For a normal install, you need:

1. A backend **OpenAI-compatible endpoint**
   (`llama_cpp`, `vllm`, `ollama`, or `custom`)
2. A frontend
   (llama.cpp WebUI + shim, OWUI, SillyTavern, or direct API client)
3. Qdrant for Vault/RAG features
   https://github.com/qdrant/qdrant

**Recommended default:**

- llama.cpp - https://github.com/ggml-org/llama.cpp
- Qdrant - https://github.com/qdrant/qdrant

That's it.

For install details, see [README Quickstart](README.md#quickstart-first-time-recommended).
Need the fastest path? See [README 5-Minute Quickstart](README.md#5-minute-quickstart).

## Table of contents

- [README Quickstart](README.md#quickstart-first-time-recommended)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [What the hell is this thing and why did you build it?](#what-the-hell-is-this-thing-and-why-did-you-build-it)
  - [What problems does this solve?](#what-problems-does-this-solve)
  - [What is Vodka?](#what-is-vodka)
  - [How does CTC (Cut The Crap) work?](#how-does-ctc-cut-the-crap-work)
  - [What are "breadcrumbs" and why do they matter?](#what-are-breadcrumbs-and-why-do-they-matter)
  - [How does TR (Total Recall) work?](#how-does-tr-total-recall-work)
  - [What is Mentats?](#what-is-mentats)
  - [What is the Vault?](#what-is-the-vault)
  - [How do I verify answers are not bullshit?](#how-do-i-verify-answers-are-not-bullshit)
  - [Why is this awesome on potato PCs?](#why-is-this-awesome-on-potato-pcs)
  - [What models do I need?](#what-models-do-i-need)
  - [What's the difference between modes?](#whats-the-difference-between-modes)
  - [What do these commands *actually* do?](#what-do-these-commands-actually-do)
  - [>>web (web retrieval sidecar)](#web-web-retrieval-sidecar)
  - [Cheatsheets (JSONL grounding)](#cheatsheets-jsonl-grounding)
  - [Scratchpad Deep Example](#scratchpad-deep-example)
  - [Common workflows](#common-workflows)
  - [What the hell is a shim (and why should you care?)](#what-the-hell-is-a-shim-and-why-should-you-care)
  - [Technical Setup](#technical-setup)
  - [Troubleshooting 101](#troubleshooting-101)
  - [Config knobs (router_config.yaml) with examples](#config-knobs-router_configyaml-with-examples)
  - [Advanced: How does provenance work?](#advanced-how-does-provenance-work)
  - [TIPS](#tips)
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
- Consistent prompt size, stable performance, and optional rolling deterministic summary using built-in Python text processing (extractive, no LLM generation).
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
   - Current behavior: `>>summ` uses deterministic extractive text processing built with Python's standard library, not an LLM generation call.
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

#### Does Mentats combine older and newer Vault knowledge?

Short answer: yes, usually.

Human version:
- `##mentats` does not only read your latest promoted file.
- It queries the full Vault collection in Qdrant and pulls the best-matching chunks.
- So if knowledge pile A and knowledge pile B were both promoted, one Mentats answer can combine A + B.

Example:
1. Promote first batch:
   - `>>attach kb_a`
   - `>>summ new`
   - `>>move to vault`
2. Later promote second batch:
   - `>>attach kb_b`
   - `>>summ new`
   - `>>move to vault`
3. Ask a cross-topic question:
   - `##mentats How do findings in A change the interpretation of B?`

Expected behavior:
- Mentats can pull relevant chunks from both promoted sets and synthesize one answer.

Important caveat:
- Retrieval is still top-k + rerank. If the query wording is vague, you might only get one side.
- If that happens, ask again with explicit anchors from both domains (names, terms, dates, entities).

#### `>>move to vault`

- `>>move to vault`:
  - promotes SUMM knowledge into Qdrant (used by `##mentats`).

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
- **Thinker:** Qwen-3-4B 2507 Instruct (or similar). Recommend Heretic/Hivemind-style builds for strong small-model reasoning.
- **Critic:** Phi-4-mini (or similar), kept DIFFERENT from thinker for better error-checking.
- **Vision:** Qwen3-VL-4B + mmproj (fast, accurate, good for both OCR and "what is this?" image dumps)
- **2nd Opinion:** Feature TBC. Strongly suggest keeping eye on Nanbeige 2511 thinking; it's abnormally good. See benchmarks.

#### Mid-range recommendations:
- **Thinker:** Qwen-3-8B or Llama-3.1-8B (better reasoning)
- **Critic:** Keep it a different model family (for example Phi-4-mini) instead of reusing thinker.
- **Vision:** Qwen-3-VL

#### Those model reccs are GARBAGE, dude
- So choose your own. What am I, your Rabbi?
- Huggingface is over there  https://huggingface.co/DavidAU ; https://huggingface.co/unsloth ; https://huggingface.co/TheBloke
- My personal choices: 
    * Thinker: Qwen3-4B-Hivemind-Inst-Q4_K_M-imat.gguf
	* Critic: Phi-4-mini-instruct-Q4_K_M.gguf
	* Vision: Qwen3VL-4B-Instruct-Q4_K_M.gguf + mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
	* Coder: Same as thinker
    ** REMEMBER: I RUN ON A POTATO. If you don't, go hog wild and use what you want.

#### Config (router_config.yaml):
```yaml
roles:
  thinker: "Qwen-3-4B Hivemind"
  critic: "Phi-4-mini"
  vision: "qwen-3-4B_VISUAL"
```

**Note:** Router talks to models via OpenAI-compatible API. Full config reference in [Config knobs](#config-knobs-router_configyaml-with-examples).

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

`>>attach c64` means: "attach the `c64` folder (and its contents), then ground normal answers using retrievable context from that filesystem KB."

It does **not** mean:
- Vault is attached
- everything is now guaranteed true
- Mentats is active

Use `>>list_kb` to see what exists, then `>>attach <your kb name>`.

<a id="kaioken-router"></a>
#### `>>kaioken on|off|status` - what does it do?

KAIOKEN is a real-time turn classifier that runs on every human message before generation. It reads the turn, labels it, and injects behavioural instructions into the thinker context accordingly. Those instructions are actual constraints - not metadata, not hints.

KAIOKEN is step 1 of a longer project, called "Claude in a Can".  The goal is to replicate the behavioural sophistication of a frontier model - specifically the kind of turn-aware, context-sensitive response shaping that Claude.ai does implicitly - on hardware that can't run a frontier model. 

Eg: A Qwen3-4B shouldn't behave like a Qwen3-4B. It should behave like a small model with a working prefrontal cortex bolted on externally. 

KAIOKEN is the start of that cortex. It externalises what large models do internally via scale, and implements it as explicit infrastructure. No weight updates. No fine-tuning. Structural constraint and classification, applied at inference time. Bear in mind, this is step 1 of a multistage roadmap. See blog for current roadmap [Claude-in-a-can](https://bobbyllm.github.io/llama-conductor/blog/claude-in-a-can-1/) 


**What it classifies:**

- Macro: working / casual / personal
- Subsignals: playful / friction / distress_hint / vulnerable_under_humour / kb_lookup_candidate

What the labels do: Different labels activate different behavioural contracts. A distress_hint turn suppresses cheatsheet injection and unlocks a different response register. A working turn keeps the thinker in task mode. A vulnerable_under_humour turn means someone is joking about something that isn't funny - the system knows not to play along.

What it does not do:

- Does not replace retrieval or grounding lanes (>>scratch, >>lock, Vault, Cheatsheets, sidecars)
- Does not make unknown facts true
- Does not own every route — deterministic and sidecar paths have their own contracts

Good mental model:

- KAIOKEN = behavioural shaping based on turn intent
- Retrieval lanes = knowledge grounding
- These are separate systems. Do not conflate them.

<a id="trust-router"></a>
#### `>>trust <query>` - what does it do?

`>>trust` is a recommendation router, not an auto-execution path.

What it does:
- classify your query deterministically
- suggest ranked command routes (A/B/C/...)
- suggest sidecars when they are the best route
- keep execution in your control

<a id="judge-router"></a>
---

#### `>>judge <criterion> : item1, item2, item3 [--verbose]` - what does it do?

Short answer: `>>judge` is a deterministic ranking tool, not a truth oracle. It compares options in both directions, scores the outcome, and shows its work. If you run it ungrounded, you're getting model priors. 

Which... while entertaining...and useful...isn't always *accurate*.

How it works:

- takes `2-4` comma-separated options
- compares each pair twice (`A,B` then `B,A`)
- accepts only strict verdicts (`A|B|TIE`) — fails loud if parse breaks
- outputs ranked scores + `Winner`
- computes confidence from agreement margin (not model swagger)

Where `>>trust` and `>>scratch` come in:

This bit is cool. When `>>judge` runs with `>>scratch` content attached, it reasons from *that* content — not model priors.

Suddenly, your 1.7B judge model has an informed opinion. Weird how that works when you give it something to read.

How to: 

`>>scratch`, `>>add` your evidence, then `>>judge [criterion] : option1, option2`. 

Or let `>>trust` walk you through it:

`>>trust` which is healthier, apples or pears?

- `A) >>scratch --> >>judge [HIGH]` — paste evidence first, then rank from it
- `B) >>judge [LOW]` — ungrounded pass (warns you this is priors, not verified facts)

If scratch is used, judge uses that lock scope. If the locked scope can't support the question, judge fails closed. No fake ranking, no vibes verdict.

Role routing:

- set `roles.judge` in `router_config.yaml` (recommended)
- fallback chain if empty: `roles.critic` → `roles.thinker`

Examples:

- `>>judge speed for daily use : rust, go, python`
- `>>judge best 4B model for reasoning : qwen3, phi4-mini, gemma3`
- `>>judge which is a better boy : golden retriever, labrador, pomeranian --verbose`

Verbose mode (`--verbose`):

- writes audit rows to `total_recall/judge/judge_audit_<UTCSTAMP>.jsonl`
- output includes exact file path
- audit rows include:
  - pair (`item_a`, `item_b`, order)
  - verdict (`A|B|TIE`)
  - condensed reasoning snippet
  - raw output
  - evidence provenance (`evidence_source`, `evidence_locked_indices`, `evidence_chars`)

Confidence tiers:

- `high`: top-vs-second margin `>= 1.5`
- `medium`: margin `>= 0.5` and `< 1.5`
- `low`: margin `< 0.5` (including ties)
- this is stability/agreement, not correctness proof

---

<a id="sidecars"></a>
#### Sidecars (`>>wiki`, `>>define`, `>>exchange`, `>>weather`, `>>web`) - what are they for?

These are deterministic utility tools for common retrieval/conversion tasks:

- `>>wiki <topic>`: encyclopedia summary lookup (Wikipedia)
- `>>define <word>`: word-origin/etymology lookup (deterministic Etymonline sidecar)
- `>>exchange <query>`: live currency conversion (Frankfurter API)
- `>>weather <location>`: live weather lookup (Open-Meteo API)
- `>>web <query>`: live web search with deterministic relevance gating (provider-agnostic; ships with DuckDuckGo, supports Tavily/SearxNG/custom)

They reduce token spend and avoid unnecessary model improvisation for utility queries.

### `>>web` (web retrieval sidecar)

`>>web` is a deterministic web search sidecar with strict relevance scoring.

**Two modes of operation:**

1. **Manual command:** `>>web <query>` - search the web for anything. Returns ranked results with `Source: Web` provenance and a `See: <url>` link.

2. **Automatic cascade:** When cheatsheets and wiki can't answer a retrievable-fact query, `>>web` fires automatically before model fallback. No command needed. Retrieval order: `Cheatsheets → Wiki → Web → Model`.

**Relevance scoring (deterministic, no model involvement):**

Every web result is scored before the router will accept it as evidence:
- exact phrase match in snippet/title: big bonus
- token overlap between query and result: scaled contribution
- result from a trusted domain (Wikipedia, IMDB, etc.): small boost
- combined score must pass a hard threshold or the result is discarded

If nothing passes the gate, the system cascades to the next step or fails loud. No "close enough" vibes.

**Provider abstraction:**

Ships with `ddg_lite` (DuckDuckGo Lite HTML parse - no API key, no account, no cloud dependency). Also supports:
- `tavily` (API)
- `searxng` (self-hosted)
- `custom` (generic adapter)

`ddg_lite` has strict HTML shape validation. If DuckDuckGo changes their page structure, the parser fails loud instead of returning garbage. You'll know immediately.

**User trust domains:**

Add your own trusted domains in `router_config.yaml`:

```yaml
web_search:
  user_trust_domains:
    - bbc.co.uk
    - reuters.com
```

These get the same relevance scoring boost as built-in domains. Ships empty. Built-in defaults stay active regardless.

**Footer provenance:**
- `Source: Web` -> real web evidence passed the relevance gate.
- `See: <url>` -> the actual source URL, injected deterministically from retrieval metadata. Never model-generated.
- If web retrieval fails or all results are rejected: cascades to model with `Confidence: unverified | Source: Model`.

#### Other utility commands (the ones nobody told you about)

These get mentioned in tips and troubleshooting but never properly introduced. Here they are:

- `>>help` / `>>help advanced` / `>>help full`: show the command cheat sheet (compact or full). Start here if you forget everything else.
- `>>status` / `>>status full` / `>>status raw`: inspect session state — what's attached, what's locked, what mode you're in.
- `>>find <query>`: search attached KB files for matching content.
- `>>peek <query>`: preview what retrieval chunks would come back for a query, without running the full pipeline. Useful for debugging "why did Mentats refuse my question?"
- `>>calc <expression>`: deterministic calculator. Supports `+ - * / % **`, parentheses, `sqrt/log/sin/cos`. No LLM involved.
- `>>flush`: nuclear reset. Clears CTC history, session memory, scratchpad captures, judge audit files, and resets profile/style state. Does NOT detach KBs.
- `>>detach <kb>` / `>>detach all`: stop grounding against attached KBs. Also clears lock state if the detached KB owns the locked file.
- `>>memory status`: show session memory pipeline diagnostics (unit count, last update turn, last inject).
- `>>memory show`: preview current session memory units.
- `>>memory clear`: wipe session memory units (does not detach KBs).
- `>>preset fast|balanced|max-recall`: switch memory/context preset on the fly.
- `>>vision` / `>>vl` / `>>v`: force vision pathway on an attached image. You can also just attach an image and ask naturally — the router auto-detects.
- `>>ocr` / `>>read`: force OCR text extraction from an attached image.
- `>>faq` / `>>faq <n>`: browse FAQ sections by number, directly in chat.

Full operator reference: [command_cheat_sheet.md](llama_conductor/command_cheat_sheet.md)

#### `>>summ` - LLM summary or deterministic extract?

Current behavior is deterministic extractive generation using built-in Python text processing (no LLM generation), not free-form LLM summarization.

That means:
- lower hallucination risk than generative summary
- more literal carry-over from source docs
- still lossy (it is a distillation step), so critical claims should be verified from source when stakes are high
- As proof of concept: compare [Guardian source](https://tinyurl.com/GuardianMuskArticle) vs [SUMM_musk-article.md](SUMM_musk-article.md)


### But why even use `>>summ`? Why not reason against raw text?

Two reasons: less noise and more speed. Raw text is full of nav bars, boilerplate, formatting junk. `>>summ` distils it down to the bits that matter.

In testing across multiple document types (news articles, technical docs, PDFs), `>>summ` retained approximately 85-90% of key concepts from source text. Exact coverage depends on how noisy the raw input is. It's lossy — it's a distillation step — but the losses are consistent and known because this is deterministic extraction (pure Python stdlib), not stochastic LLM generation.

Proof of concept: compare [Guardian source](https://tinyurl.com/GuardianMuskArticle) vs [SUMM_musk-article.md](SUMM_musk-article.md). That SUMM was generated by Qwen3-4B Hivemind.

For the full evidence bundle (8,974 runs, five model families, hallucination floor of 0.00% post-policy), see [PAPER.md](prepub/PAPER.md).

**Don't trust the SUMM?** Two options:
1. Use `>>scratch` instead — paste raw text, reason against it losslessly. No distillation, no coverage questions.
2. Generate the `>>summ`, keep the SHA for provenance, then swap the keypoints back out for raw text. Defeats the purpose, but it's your stack.



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

Scratch lock/unlock quick controls:
- Canonical: `>>scratch lock <index|index,index,...>` and `>>scratch unlock`
- Soft aliases (when scratchpad is attached):
  - `lock <n>` / `lock [n,m]` -> scratch lock
  - `unlock <n>` / `unlock [n,m]` -> scratch unlock
- `unlock` (without index) remains the KB unlock command (`>>unlock`).

---

### Cheatsheets (JSONL grounding)

Cheatsheets are deterministic local fact records loaded from `llama_conductor/cheatsheets/*.jsonl`.

Each non-empty line in a `.jsonl` file is one JSON object.
Blank lines are allowed for readability.

Required keys per line:

```json
{
  "term": "Nibbler",
  "category": "futurama",
  "definition": "Futurama character. ...",
  "source": "static",
  "confidence": "high",
  "tags": ["futurama", "pop_culture", "gen_x"]
}
```

Field notes:

- `term`: the lookup key users will ask for.
- `category`: lowercase topic bucket (for indexing/discovery).
- `definition`: the grounded fact text to return.
- `source`: usually `"static"` for local curated entries.
- `confidence`: one of `low|medium|high|top|unverified`.
- `tags`: extra match signals and grouping labels.

Fast path to add entries:

1. Copy an existing `.jsonl` file in `llama_conductor/cheatsheets/`.
2. Duplicate a line/object that looks close.
3. Edit `term`, `category`, `definition`, and tags.
4. Keep one valid JSON object per line.

Behavior:

- Strict lookup queries (`who/what is ...`) can return deterministic `term: definition`.
- Broader/open-ended asks may synthesize; footer should downgrade to `Source: Mixed` when not fully grounded.
- Distress lanes do not inject cheatsheets unless the user is explicitly asking a question.

How to use it in practice:

1. Discover what is loaded:
```text
list index
```
Returns deterministic category inventory from local cheatsheets index.

2. List terms inside a category:
```text
list pop culture
list hardware
list glossary
```
Returns deterministic grouped term names.

3. Get deterministic term facts:
```text
list nibbler
list qwen3-4b-hivemind
```
Returns deterministic cheatsheet fact output when the term exists.

4. Get deterministic multi-entry summaries:
```text
summary pop culture
summary hardware
summary nibbler
```
Returns formatted local definitions from cheatsheets records.

5. Then chat normally:
- After using `list`/`summary`, you can continue with normal prompts.
- Model responses may synthesize beyond local records unless you stay in strict list/summary paths.
- Footer provenance tells you what happened.

Footer provenance quick read:
- `Source: Cheatsheets` -> output is deterministic local cheatsheet-grounded.
- `Source: Mixed` -> partial cheatsheet grounding + model synthesis.
- `Source: Model` -> no local cheatsheet grounding used.

---

<a id="deep-example"></a>
### Scratchpad Deep Example

Source article:
- Original: https://tinyurl.com/GuardianMuskArticle
- `>>add` (and copy paste the text in)

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
!! my sourdough starter is named Greg
!! project deadline is 2026-02-15
!! the wifi password at work is taped under Karen's desk
```

#### Recalling facts later:
```
?? sourdough
?? deadline
?? wifi password
```

### What the hell is a shim (and why should you care?)

Plain language version:

- A shim is a thin "middle layer" between llama.cpp WebUI and llama-conductor. 
- The lovely people at llama.cpp ship with inbuilt web-ui, so why not use it? It's light, fast and functional.
- It does not replace your model, and it does not replace the router.
- It forwards/re-wires the right calls so the WebUI can piggyback on router behavior.

What this lets you do:

- Keep using llama.cpp's built-in WebUI as your frontend.
- Route chat/model traffic through llama-conductor so role routing and command behavior still apply.
- Smooth out model metadata in the UI layer so picker/capability behavior is more consistent.

Why this is cool:

- Fewer moving parts for MVP: llama.cpp + shim + open the shim URL on your PC = now you have it everywhere on your LAN.
- Add Qdrant and you have the full stack (Vault/RAG and `##mentats`).

If you want full operator details (env vars, endpoints, render modes), see [SHIM-README.md](llama_conductor/shim/SHIM-README.md).

### Technical Setup

#### Architecture Overview

**The stack (in human):**
```
Your model backend (llama.cpp / vLLM / Ollama / whatever) + llama-conductor (the brains) + your frontend (llama.cpp WebUI + shim / OWUI / SillyTavern / raw API)
```

**How the plumbing works:**
1. You tell `router_config.yaml` what backend you're using (`backend.provider`).
2. If it's `llama_cpp`, the launcher fires up `llama-server` for you on the configured port.
3. If it's anything else (`vllm`, `ollama`, `custom`), the launcher assumes you've already got something running and talks to your upstream URL.
4. Router speaks OpenAI-compatible API to whatever's listening.
5. Frontend gets the result either directly or via the shim. You pick.

#### Model Backend: provider-driven

Router talks to your models via OpenAI-compatible API. Point it at a URL and go. `llama-swap` is legacy cruft — you don't need it anymore.

**Can I use other backends?**

If it speaks OpenAI-compatible API, it works. llama.cpp, vLLM, Ollama, LMStudio, Text Generation WebUI — all fair game. Just set `backend.upstream_chat_url` to your endpoint and get on with your life.

See [Config knobs](#config-knobs-router_configyaml-with-examples) for the full backend config block.

#### RAG Backend: Qdrant

Vault and Mentats need a vector DB. We use **Qdrant** (https://github.com/qdrant/qdrant). It's fast, it's light, it runs in Docker, and it doesn't make me want to throw my PC out the window.

**Quickest path:**

#### Docker (recommended)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

Done. That's your vector store running.

##### Docker Compose (single-host quickstart)

Want the whole stack in one go? Use the bundled compose files:

```bash
docker compose up -d
```

Want the optional WebUI too?

```bash
docker compose --profile webui up -d
```

What each piece does:
- `qdrant`: vector store for Vault/Mentats
- `llama-conductor`: the router and shim
- `open-webui` (optional): a frontend, if you don't want to use llama.cpp's WebUI

Useful commands:

```bash
docker compose logs -f     # watch what's happening
docker compose down        # shut it all down
```

Repo files for this flow: `docker-compose.yml`, `docker.env.example`, `docker/router_config.docker.yaml`

#### Or install from binaries and run bare-metal like I do

See [Config knobs](#config-knobs-router_configyaml-with-examples) for Qdrant host/port settings.

**Can I use a different vector DB?**

Technically? Sure. Practically? You'd be rewriting `rag.py` from scratch. Qdrant is wired in deep — collection management, named vectors, reranking, the lot. Swapping to Milvus/Weaviate/Pinecone is non-trivial. You want that headache? Knock yourself out. I recommend Qdrant.

#### Embeddings & Reranking

First time you run the router, it'll download two small models from HuggingFace (~200MB total). After that, cached.

**Embedder** (turns text into vectors): `intfloat/e5-small-v2` — 384-dim, fast, good enough for potato hardware.

**Reranker** (sorts the wheat from the chaff): `cross-encoder/ms-marco-TinyBERT-L-2-v2` — tiny, fast, does its job.

**Want to swap them?** Any sentence-transformers-compatible model works. Bigger embedder for better quality (`intfloat/e5-base-v2`), disable reranking entirely (`rerank_model_name: ""`), or bring your own cross-encoder. See [Config knobs](#config-knobs-router_configyaml-with-examples) for the settings.


**This is *a lot* of moving parts. What's the ABSOLUTE MINIMUM STACK to use as proof of concept?**

Totally fair. Keep it simple:

- Run *one* local model in llama.cpp
- Use llama.cpp WebUI + shim
- In `router_config.yaml`, for ULTRA bare-bones proof of concept, point `thinker`/`critic`/`coder`/`second_op` to one model ID (and keep `vision` as a real VL model ID if you want VL capabilities)
- Launch via your normal `.bat` flow (or however you do you)

This is *ULTRA* bare bones mode: core chat/routing works, but Vault/RAG paths (`##mentats`), OCR-linked flows, multiple model calling etc are NOT available until you apply (at the least) "kick-the-tires" set up per the [Quickstart](#quickstart).

#### Launch Script: The Easy Way

Use this as the beginner-friendly llama.cpp + Qdrant flow.

**Minimal Windows `.bat` example:**
```batch
@echo off
setlocal
cd /d "C:\path\to\your\launch\folder"

REM Start Qdrant (reuse existing container if present)
docker start qdrant >nul 2>&1 || docker run --name qdrant -p 6333:6333 -d qdrant/qdrant

REM Show computed launch plan (sanity check)
python -m llama_conductor.launch_stack doctor --config llama_conductor/router_config.yaml

REM Start llama.cpp + router + shim stack
python -m llama_conductor.launch_stack up --config llama_conductor/router_config.yaml
```

**Linux/macOS equivalent:**
```bash
cd /path/to/your/launch/folder

# Start Qdrant (reuse existing container if present)
docker start qdrant >/dev/null 2>&1 || docker run --name qdrant -p 6333:6333 -d qdrant/qdrant

# Show computed launch plan (sanity check)
python -m llama_conductor.launch_stack doctor --config llama_conductor/router_config.yaml

# Start llama.cpp + router + shim stack
python -m llama_conductor.launch_stack up --config llama_conductor/router_config.yaml
```

**Required config:** See [Config knobs](#config-knobs-router_configyaml-with-examples) for the full `router_config.yaml` reference. The key things to get right for launch:

- `backend.provider` must match your setup (`llama_cpp` if you want the launcher to manage llama-server for you)
- `backend.llama_cpp.exe_path` / `models_dir` / `models_preset_path` must point at real files
- `roles.*` values must match the model ID your backend exposes in `/v1/models`
- For best results, thinker and critic should be TWO DIFFERENT models (eg: Qwen3-4B and Phi-4-mini)

**VL gotcha (important):**
- Most vision models in llama.cpp need `mmproj` configured in `llama_server.models_preset.models`.
- If `vision` points to a VL model but `mmproj` is missing/wrong, image/OCR behavior will fail.
- Make sure you have both! 

**Alias note:**
- Router role mapping uses the actual model ID string, not a separate alias.
- UI aliases (for display/picker) can exist in shim/frontend layers, but `roles.*` should target a valid backend model ID.

#### Non-Local LLMs (API Keys)

**Can I use cloud APIs instead of local models?**

**Probably!** Router works with any OpenAI-compatible API:

**Example: OpenRouter**
```yaml
backend:
  provider: "custom"
  upstream_base_url: "https://openrouter.ai/api/v1"
  upstream_chat_url: "https://openrouter.ai/api/v1/chat/completions"

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
resp = requests.post(UPSTREAM_CHAT_URL, json=payload, headers=headers)
```

**Option 2: Config file** (requires minor code change)
```yaml
upstream_headers:
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
- Remember: the pathway is Add docs to KB, `>>summ new`, `>>move to vault`
- ##Mentats has STRICT recall schema. Check the mentats.log for failure reason

#### Mentats gives weird answers
- Check `mentats_debug.log` (all 3 steps logged)
- May be low-quality source (forum post vs docs)

#### Context too long errors
- Increase `vodka.max_chars` in config
- Or decrease `vodka.n_last_messages`
- Or use smaller prompts

#### Qdrant connection failed
- Check Qdrant is running:
  - Windows PowerShell: `docker ps --format "{{.Names}}" | Select-String qdrant`
  - Linux/macOS: `docker ps | grep qdrant`
- Verify port: `curl http://localhost:6333/health`
- Check firewall isn't blocking port 6333

#### Models not loading (backend provider)
- Check provider startup:
  - `llama_cpp`: confirm `llama-server` is listening on `backend.local_llama_port`
  - `vllm` / `ollama` / `custom`: confirm upstream endpoint is reachable
- Verify `backend.upstream_chat_url` is correct for your provider
- Verify `roles.*` model IDs match what `/v1/models` actually returns
- Check provider logs for model-load failures or OOM

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

#### Web search sidecar:
```yaml
web_search:
  provider: "ddg_lite"            # ddg_lite|tavily|searxng|custom
  max_results: 3                  # top N results after relevance gate
  relevance_threshold: 3.0        # minimum score to accept a result
  user_trust_domains: []          # your trusted domains (appended to built-ins)
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
backend:
  provider: "llama_cpp"  # llama_cpp|vllm|ollama|custom
  local_llama_port: 8010
  upstream_base_url: "http://127.0.0.1:8010"
  upstream_chat_url: "http://127.0.0.1:8010/v1/chat/completions"
  llama_cpp:
    exe_path: "C:/path/to/llama-server.exe"
    models_dir: "C:/path/to/LLMs"
    models_preset_path: "C:/path/to/LLMs/llama-router-models.ini"
    host: "0.0.0.0"
    port: 8010

# Legacy fallback only (deprecated)
llama_swap_url: "http://127.0.0.1:8010/v1/chat/completions"

frontend:
  shim:
    enabled: true
    inject_ui: true
    force_vision: true
    ui_model_alias: "MOA"
    router_model_id: "moa-router"

launcher:
  router_host: "0.0.0.0"
  ready_timeout_s: 22

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

Check `NEW.md`. That's where the fresh stuff lives.

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
- `>>detach all` and `>>flush` reset profile/style runtime state (`>>flush` also purges session-memory and judge-audit JSONL artifacts).

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
- `Source: Web`
  - Grounded to live web retrieval that passed deterministic relevance scoring.
  - Includes `See: <url>` link to source.
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

The router checks where support came from and how direct that support is, then assigns a tier:

- Clear, direct support in the active source scope (for example locked SUMM, scratchpad, or strong doc hits) -> `high` to `top`
- Real source support exists, but only partly answers the question -> `medium`
- Source was touched, but support is weak/noisy/unclear -> `low`
- No grounded support; answer fell back to model knowledge -> `unverified`

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

## License
AGPL-3.0-or-later. See `LICENSE`
