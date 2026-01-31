# Frequently Asked Questions (FAQ)

## What the hell is this thing — and why did you build it?

**Llama-conductor** is a LLM harness for people who want **trust, consistency, and proof**.

More prosaically, it's a Python router + memory store + RAG harness that forces models to behave like **predictable components**, not improv actors.

I have ASD and a low tolerance for bullshit. I want shit to work the same 100% of the time.

**TL;DR:** *"In God we trust. All others must bring data."*

---

## What problems does this solve?

### 1) "Vibes-based answers" — how do I know if the LLM is lying?

You don't! 

**WITHOUT llama-conductor:**
```
You: What's the flag for XYZ in llama.cpp?
Model: It's --xyz-mode! (confident bullshit)
You: [tries it] → doesn't work
Model: Oh sorry! Try --enable-xyz! (more bullshit)
```

**WITH llama-conductor:**
```
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: [queries Vault only]
Mentats: REFUSAL — "The Vault contains no relevant knowledge for this query."

[later, after you >>summ the llama.cpp docs]
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: --enable-xyz [cite: llama.cpp docs, SHA: abc123...]
[verifies: came from llama.cpp README.md, SHA matches]
```

**How it works:**
1. `>>attach <kb>` — attach your curated docs (filesystem folders)
2. `>>summ new` — generate summaries with SHA-256 provenance
3. `>>move to vault` — promote summaries into Qdrant RAG
4. `##mentats <query>` — deep reasoning, grounded in Vault only

---

### 2) Goldfish memory — models forget or "confidently misremember"

**WITHOUT llama-conductor:**
```
You: Remember my server is at 203.0.113.42
Model: Got it!
[100 messages later]
You: What's my server IP?
Model: 127.0.0.1 🥰
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

### 3) Context bloat — 400-message chat logs kill your potato PC

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

## What is Vodka?

**Vodka** = "When God gives you a potato PC, squeeze until Vodka"

It's a **deterministic memory + prompt-sanitizer filter** that runs **before** and **after** your model sees the conversation. No LLM compute needed (pure 1990s tech).

### Vodka has two jobs:

1. **CTC (Cut The Crap)**: Trim chat history so your PC survives
2. **TR (Total Recall)**: Store/recall facts exactly (JSON file on disk)

### How it works:

Router calls `VodkaFilter()` on every request:
```python
# inlet() — runs BEFORE model sees messages
messages = vodka.inlet({"messages": raw_messages})
# ↓ trims history, handles !! / ?? / control commands

# outlet() — runs AFTER model responds
messages = vodka.outlet({"messages": messages})
# ↓ expands [ctx:...] breadcrumbs into stored facts
```

---

## How does CTC (Cut The Crap) work?

CTC is a **deterministic message trimmer** that keeps conversations lightweight.

### Config knobs (router_config.yaml):

```yaml
vodka:
  n_last_messages: 4      # Keep last N user/assistant pairs
  keep_first: true        # Keep first message (system setup)
  max_chars: 3000         # Hard cap on message chars
```

### What it does:

1. Extract last N user/assistant message pairs (usually 2 = 4 messages)
2. Optionally keep the first message
3. Hard-cap total character count
4. Model sees only this trimmed version

### Effect:

- **Before CTC:** 400 messages, 250KB context
- **After CTC:** 4-6 messages, ~5KB context
- **Result:** Same effective prompt size on tiny context windows

---

## What are "breadcrumbs" and why do they matter?

**Breadcrumbs** are `[ctx:ID]` markers that point to stored Vodka facts.

### How breadcrumbs work:

```
You: !! my server is 203.0.113.42
Vodka: [stores as ctx_abc123]

Later:
You: ?? server ip
Vodka message: "Your server is [ctx_abc123]"
↓
outlet() expands: "Your server is 203.0.113.42 [TTL=3d, Touch=1]"
↓
Model sees: exact fact + metadata
```

### Why this is powerful:

1. **Exact recall** — no paraphrasing
2. **Traceability** — you can verify the stored text
3. **Metadata** — TTL and touch counts visible
4. **Fail-open** — if expansion fails, model still sees the ID

---

## How does TR (Total Recall) work?

TR stores facts in `facts.json` (JSON file on disk) with automatic expiration.

### Store facts:

```bash
!! this is a fact to remember
# Stored as: {"ctx_abc123": {"value": "...", "created_at": "...", "expires_at": "...", ...}}
```

### Recall facts:

```bash
?? search query
# Finds matching facts, returns with TTL + touch metadata
```

### TTL + touches (anti-hoarder system):

- **TTL (Time To Live):** How many days before fact auto-expires (default: 3)
- **Touches:** How many times fact can be recalled before expiring (default: 2)
  - Each recall extends TTL by `touch_extension_days` (default: 5)
  - After `max_touches` recalls, fact is deleted

Example:
```
Day 1: !! fact (TTL=3 days, touches=0)
Day 2: ?? fact (touches=1, TTL extended to 7 days)
Day 3: ?? fact (touches=2, TTL extended to 8 days)
Day 4: ?? fact (DELETED — max_touches=2 reached)
```

---

## What is Mentats?

**Mentats** = "In-house expert mode" (named after the human computers from Dune)

It's a **3-pass deep reasoning harness** that:
1. **Step 1 (Thinker):** Drafts answer from Vault facts only
2. **Step 2 (Critic):** Hunts for hallucinations (temperature=0.1)
3. **Step 3 (Thinker):** Produces final answer, removing violations

### How Mentats works:

```bash
##mentats What's the Amiga's retail price?
```

**Step 1 (Draft):**
- Query: `build_rag_block("What's the Amiga's retail price?")`
- RAG returns facts from Vault
- Model drafts answer using ONLY those facts
- If no facts: **REFUSES** ("The provided facts do not contain...")

**Step 2 (Critic):**
- Checks for hallucinations: entity hallucination, number mismatches, invented comparisons, inference
- If violations found: `CHECK_OK: NO`
- If clean: `CHECK_OK: YES`

**Step 3 (Final):**
- If Step 2 marked violations: **remove them** (don't replace)
- Output: `FINAL_ANSWER: [clean answer]\nSources: Vault`

**Invariants:**
- Mentats MUST use Vault facts only (no chat history, no Vodka memory)
- Mentats MUST refuse cleanly if facts are insufficient
- Mentats MUST NOT hallucinate

---

## What is the Vault?

**Vault** is a Qdrant-based semantic search system for curated documents.

### How it works:

```bash
>>attach c64            # Attach KB folder
>>summ new              # Summarize raw docs
>>move to vault         # Chunk + embed + store in Qdrant

Later:
##mentats [query]       # Search Vault, ground reasoning in results
```

**Vault stores:**
- Chunk text
- Embedding vector (e5-small-v2)
- Full provenance (source file, SHA, creation date)
- KB metadata

### Why Vault vs filesystem KBs?

| Aspect | Vault (Qdrant) | Filesystem KBs |
|--------|-----------|------------|
| **Query type** | Semantic ("cultural impact on music") | Exact text ("unit price") |
| **Speed** | Fast (vector similarity) | Fast (substring search) |
| **Precision** | Good (but embedding-dependent) | Excellent (exact match) |
| **Use case** | Exploratory, synthesis | Fact lookup, verification |
| **Mentats** | ✅ Vault only | ❌ Not used |
| **Serious** | ✅ Optional | ✅ Primary |

**TL;DR:** Use `##mentats` for semantic reasoning. Use `>>find` for exact facts. Use `##fun` / `/serious` for creative answers.

---

## How do I verify answers aren't bullshit?

Mentats outputs include full provenance:

```
FINAL_ANSWER: The Amiga retail price was $1295.

Sources: Vault

FACTS_USED:
- Amiga was priced at launch ($1295) and quickly reduced to $700...
```

### Verify SHA manually:

```bash
# Get original SHA from Vault entry
sha256sum original_doc.pdf
# Compare to stored SHA in SUMM file

# If they match: document hasn't changed
# If they differ: document was edited since summarization
```

---

## Why is this awesome on potato PCs?

### 1) Vodka CTC keeps context small:
- Default: last 4 messages + ~1500 chars
- vs. typical: 100+ messages, 50KB+ context
- Result: 4B models run smoothly on 8GB RAM

### 2) TR does memory without context:
- Facts stored in JSON, not chat history
- `??` expands them at inference time
- You get "long-term memory" with tiny context windows

### 3) Filesystem KBs are just folders:
- No database needed
- `>>summ new` creates SUMM_*.md files
- `>>find` searches plain text
- Works on microSD cards

### 4) Mentats is efficient:
- Only 3 model calls per query
- Temperature=0.1 for critic (deterministic)
- Stops early if Vault returns no facts

---

## What models do I need?

### Minimum (potato mode):

- **Thinker:** 4B model (Qwen-3-4B, Phi-4-mini)
- **Critic:** 2B model (Phi-2, Phi-mini)
- **Vision:** Optional (if using `##ocr`)

### Mid-range recommendations:

- **Thinker:** 7B model (Qwen-7B, Mistral-7B)
- **Critic:** 3-4B model (Phi-4, Qwen-3)
- **Vision:** 7B multimodal (qwen-vl-max)

### Those model reccs are GARBAGE, dude

You're right. **Model choice matters.** What works on YOUR hardware depends on:
- GPU VRAM available
- Quantization (Q4, Q5, Q8)
- Batch size
- Context size

**Test on your hardware.** Start small, scale up.

### Config (router_config.yaml):

```yaml
roles:
  thinker: "Qwen-3-4B Hivemind"
  critic: "Phi-4-mini"
  vision: "qwen-3-4B_VISUAL"
```

**Must match exactly** what llama-swap has loaded.

---

## What's the difference between modes?

### Serious (default):

- Standard Q&A
- Uses optional KB grounding (if attached)
- Includes confidence/source line
- No styling

### Mentats (##mentats):

- 3-pass deep reasoning
- Vault facts only (no KB, no history)
- Refuses if facts insufficient
- Always includes sources + provenance

### Fun (##fun or >>fun):

- Serious answer + style rewrite
- Uses seed quote from quotes.md
- Sticky mode: `>>fun` keeps it on until `>>fun off`

### Fun Rewrite (>>fr):

- Rewrite an existing Serious answer in a character voice
- Sticky mode: `>>fr` keeps it on until `>>fr off`

---

## Common workflows

### Adding new knowledge:

```bash
# 1. Create folder: C:/docs/myknowledge/
# 2. Add docs: README.txt, guide.pdf, notes.md
>>attach myknowledge
>>summ new                    # Generates SUMM_*.md with SHA
>>move to vault               # Embeds into Qdrant
##mentats What does the guide say?
```

### Storing personal facts:

```bash
!! my vaccine lot is ABC-123
!! my child's birthday is 2020-03-15
?? vaccine lot                # Exact recall
?? birthday                   # Exact recall
```

### Recalling facts later:

```bash
?? [search query]
# Searches facts.json for matching entries
# Returns with TTL + touch metadata
# Extends TTL by default
```

---

## Technical Setup

### Architecture Overview

```
Frontend (Open WebUI)
    ↓
Router (FastAPI) — port 9000
    ├→ llama-swap (port 8011) ← Models (llama.cpp)
    ├→ Vodka Filter (memory + CTC)
    ├→ RAG (rag.py)
    └→ Qdrant (port 6333, Vault)
```

### Model Backend: llama-swap

Router is **model-agnostic.** It sends OpenAI-compatible requests to llama-swap.

llama-swap then dispatches to:
- llama.cpp
- vLLM
- Ollama
- Any OpenAI-compatible API

### RAG Backend: Qdrant

- Vector database for Vault
- Stores embeddings (e5-small-v2, 384-dim)
- Supports named vectors
- Default collection: `moa_kb_docs`

### Embeddings & Reranking

- **Embed model:** intfloat/e5-small-v2 (384-dim)
- **Rerank model:** cross-encoder/ms-marco-TinyBERT-L-2-v2

Both downloaded on first use via HuggingFace.

---

## Non-Local LLMs (API Keys)

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

## Troubleshooting 101

### "No Vault hits stored yet"
- Run `##mentats` first (not a normal query)

### "PROVENANCE MISSING"
- Vault entry from before v1.0.9?
- Re-run `>>move to vault` to add provenance

### SHA mismatch
- Original doc changed after SUMM
- Either: tampering, or legit edit
- Re-run `>>summ new` to update

### Mentats refuses everything
- Vault is empty or has no relevant facts
- Run `>>peek <query>` to see what would be retrieved
- Add docs to KB, `>>summ new`, `>>move to vault`

### Model gives weird answers
- Check `mentats_debug.log` (all 3 steps logged)
- May be low-quality source (forum post vs docs)

### Context too long errors
- Increase `vodka.max_chars` in config
- Or decrease `vodka.n_last_messages`
- Or use smaller prompts

### Qdrant connection failed
- Check Qdrant is running: `docker ps | grep qdrant`
- Verify port: `curl http://localhost:6333/health`
- Check firewall isn't blocking port 6333

### Models not loading (llama-swap)
- Check llama-swap is running and accessible
- Verify model names in config match llama-swap models. **THIS IS CRITICAL. MUST HAVE IDENTICAL MODEL NAMES IN BOTH YAML FILES.**
- Check llama-swap logs for errors

---

## Config knobs (router_config.yaml) with examples

### Vodka (memory + context):
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

### RAG (Vault retrieval):
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

vault:
  chunk_words: 250             # Smaller chunks = better semantic matching
  chunk_overlap_words: 50      # Overlap prevents context loss
```

### KBs (filesystem folders):
```yaml
kb_paths:
  c64: "C:/docs/c64"
  amiga: "C:/docs/amiga"
  work: "C:/docs/work"
```

### Models (backend endpoints):
```yaml
llama_swap_url: "http://127.0.0.1:8011/v1/chat/completions"

roles:
  thinker: "Qwen-3-4B Hivemind"
  critic: "Phi-4-mini"
  vision: "qwen-3-4B_VISUAL"
```

---

## Advanced: How does provenance work?

### SUMM files include metadata:
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

### Vault stores full provenance:
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
Answer → Vault chunk → SUMM file → Original doc → SHA verification
```

**Why this matters:**
- Detect tampering (SHA mismatch)
- Trace claims to sources
- Debug hallucinations (low-quality sources)

---

## TIPS

- Use `##m` instead of `##mentats` (haxor like the cool kids)
- Run `>>attach all` before `>>summ new` to process multiple KBs at once
- Check `mentats_debug.log` if Mentats gives weird answers
- Use `>>peek` to preview FACTS before running full query
- Store facts with `!!`, not "remember this" (Vodka is literal, not vibes)
- Facts expire after TTL (use `??` to extend them)
