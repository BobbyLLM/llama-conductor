# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)

**LLM harness for people who want trust, consistency, and proof.**

Llama-conductor is a Python router + memory store + RAG harness that forces models to behave like **predictable components**, not improv actors.

**TL;DR:** *"In God we trust. All others must bring data."*

---

## Quick Links

📖 **[What's New](NEW.md)** — Recent updates & bug fixes  
❓ **[FAQ](FAQ.md)** — How this thing actually works  
⌨️ **[Command Cheat Sheet](command_cheat_sheet.md)** — All router commands at a glance  

---

## Quickstart

### Required stack

You need these parts working together:

1. **llama-swap** → https://github.com/mostlygeek/llama-swap
2. **llama.cpp (or other runner)** → https://github.com/ggml-org/llama.cpp
3. **Frontend UI (example: Open WebUI)** → https://github.com/open-webui/open-webui
4. **Qdrant (Vault / RAG)** → https://github.com/qdrant/qdrant (optional, required for `##mentats`)

**Minimum:** (1) + (2) + (3).

### Install

```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

### Run the router

```bash
llama-conductor serve --host 0.0.0.0 --port 9000
```

> This starts **only** the router. You still need to start llama-swap, your model runner(s), Qdrant, and your frontend separately.

---

## Core Features

### 🔒 Grounded Reasoning (`##mentats`)
Deep 3-pass reasoning grounded in your curated docs only. Refuses to hallucinate when facts are missing.

### 💾 Perfect Memory (`!!` / `??`)
Store facts exactly as stated. Recall them verbatim with TTL and touch tracking. No "helpful rewrites."

### 📦 Context Trimming (Vodka CTC)
Automatically keeps only the last N messages + first message. Hard caps total context size. Perfect for potato PCs.

### 🎯 Deterministic Sidecars
`>>calc`, `>>find`, `>>list`, `>>flush` — deterministic operations that bypass LLM flakiness entirely.

### 🎭 Multiple Modes
- **Serious** (default) — straightforward answers
- **Fun** — answers styled with random seed quotes
- **Mentats** — deep reasoning from Vault only

---

## Common Workflows

### Adding knowledge
```bash
>>attach c64           # Attach KB folder
>>summ new             # Summarize raw docs with SHA provenance
>>move to vault        # Promote summaries into Qdrant
```

### Storing personal facts
```bash
!! my server is at 203.0.113.42    # Store exactly
?? server ip                        # Recall exactly
```

### Deep reasoning
```bash
##mentats What does the Amiga's chipset do?
# (Mentats only uses facts from Vault, refuses if missing)
```

---

## Architecture

**Router** (FastAPI) → **LLM backend** (llama-swap) ← **Models** (llama.cpp)  
↓  
**Vodka filter** (memory + CTC) + **RAG** (Qdrant) + **Filesystem KBs**

See [FAQ](FAQ.md) for deep dives.

---

## Config

Edit `router_config.yaml` to customize:
- Model role mappings
- KB paths
- Vodka memory settings (TTL, context size)
- Qdrant connection

See [FAQ](FAQ.md) **Config knobs** section for examples.

---

## Troubleshooting

- `##mentats` refuses everything? → Run `>>peek <query>` to see what's in Vault
- Weird answers? → Check `mentats_debug.log` for the full 3-step reasoning chain
- Qdrant connection failed? → `curl http://localhost:6333/health`

See [FAQ](FAQ.md) **Troubleshooting** section for more.

---

## License

AGPL-3.0-or-later. See `LICENSE`
