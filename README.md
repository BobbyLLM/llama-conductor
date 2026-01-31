# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)

**LLM harness for people who want trust, consistency, and proof.**

Llama-conductor is a Python router + memory store + RAG harness that forces models to behave like **predictable components**, not improv actors.

I have ASD and a low tolerance for bullshit. I want shit to work the same 100% of the time.

**TL;DR:** *"In God we trust. All others must bring data."*

---

## Quick Links

📖 **[What's New](NEW.md)** — Latest fixes & what we broke along the way  
❓ **[FAQ](FAQ.md)** — Deep dives on how this actually works  
⌨️ **[Command Cheat Sheet](llama_conductor/command_cheat_sheet.md)** — All the incantations  
🚀 **[Quickstart](#quickstart)** — Get it running in 5 minutes  

---

## Three Problems This Actually Solves

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
1. `>>attach <kb>` — attach your curated docs
2. `>>summ new` — generate summaries with SHA-256 provenance
3. `>>move to vault` — promote into Qdrant RAG
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
- None of this is stored in the LLM. Pure 1990s JSON magic.

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

Result: Consistent prompt size, stable performance.
Bonus: Your Raspberry Pi can actually run that 4B model and not chug.
```

---

## Quickstart

### Required stack

You need these parts. No exceptions.

1. **llama-swap** → https://github.com/mostlygeek/llama-swap (the glue)
2. **llama.cpp** → https://github.com/ggml-org/llama.cpp (the engine)
3. **Frontend UI** → https://github.com/open-webui/open-webui (the interface)
4. **Qdrant** → https://github.com/qdrant/qdrant (the brain, optional but you want it)

**Absolute minimum:** (1) + (2) + (3). Everything else uses defaults.

**For `##mentats`:** You need (4). Without it, Mentats runs degraded (no Vault grounding).

### Install

```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

### Run it

```bash
llama-conductor serve --host 0.0.0.0 --port 9000
```

> This starts **only** the router. You still need llama-swap, your model runner(s), Qdrant, and your frontend running separately. Start them in this order: Qdrant → llama-swap → router → frontend.

---

## What This Actually Does

### 🔒 Grounded Reasoning (`##mentats`)
Ask questions and get answers grounded **only** in your curated docs. If the answer isn't in your docs, Mentats refuses. No hallucinations. No "well actually" from the model's training data.

### 💾 Perfect Memory (`!!` / `??`)
Store facts exactly as stated. Recall them verbatim. No "helpful rewrites." No forgetting. TTL-based auto-expiration so you don't hoard garbage.

### 📦 Context Trimming (Vodka CTC)
Automatically keeps only the last N messages. Hard caps context size. 400-message chat logs become 4 messages. Your potato PC will thank you.

### 🎯 Deterministic Sidecars
`>>calc`, `>>find`, `>>list`, `>>flush` — operations that **never hallucinate** because they don't use the LLM at all.

### 🎭 Multiple Modes
- **Serious** (default) — straightforward answers
- **Fun** — answers wrapped in random seed quotes (from your quotes.md)
- **Mentats** — deep 3-pass reasoning from Vault only

---

## Common Workflows

### Add knowledge
```bash
>>attach c64           # Point to your KB folder
>>summ new             # Summarize raw docs (generates SUMM_*.md with SHA)
>>move to vault        # Chunk + embed + store in Qdrant
```

### Store personal facts
```bash
!! my server is at 203.0.113.42    # Store exactly
?? server ip                        # Recall exactly (no "AI rewrites")
```

### Deep reasoning
```bash
##mentats What does the Amiga's chipset do?
# (Only uses facts from Vault; refuses if missing)
```

---

## Why This Works on Potato PCs

1. **Vodka CTC** keeps context tiny (last 4 messages instead of 400)
2. **Vodka TR** does "memory" via JSON, not context (facts live on disk, not in chat history)
3. **Filesystem KBs** are just folders (no database bloat)
4. **Mentats** only makes 3 LLM calls per query
5. **Sidecars** bypass LLM entirely for deterministic operations

Your Raspberry Pi can actually run this.

---

## Architecture

```
Frontend (Open WebUI)
    ↓
Router (FastAPI) — port 9000
    ├→ llama-swap (port 8011) ← Models (llama.cpp)
    ├→ Vodka Filter (memory + CTC)
    ├→ RAG (rag.py) 
    └→ Qdrant (port 6333, Vault)
```

See [FAQ](FAQ.md) for deep technical dives.

---

## Configuration

Edit `router_config.yaml`:
```yaml
roles:
  thinker: "Qwen-3-4B Hivemind"    # Main reasoning
  critic: "Phi-4-mini"              # Fact-checking in Mentats
  vision: "qwen-3-4B_VISUAL"        # Image understanding

vodka:
  n_last_messages: 2                # Keep last 2 user/assistant pairs
  max_chars: 1500                   # Hard cap context size
  base_ttl_days: 3                  # How long facts live

vault:
  chunk_words: 250                  # Smaller = better semantic matching
  chunk_overlap_words: 50           # Prevent context loss
```

Full reference in [FAQ](FAQ.md) **Config knobs** section.

---

## Quick Troubleshooting

- `##mentats` refuses everything? → `>>peek <query>` to see what Vault found
- Weird Mentats answers? → Check `mentats_debug.log` for all 3 reasoning steps
- Qdrant won't connect? → `curl http://localhost:6333/health`
- Models not loading? → **VERIFY MODEL NAMES MATCH BETWEEN router_config.yaml AND llama-swap config.yaml** (this trips everyone up)

Full troubleshooting in [FAQ](FAQ.md).

---

## License

AGPL-3.0-or-later. See `LICENSE`
