# What's New

## Latest (v1.1.1) 

### 🐛 Critical Fixes

**Fun mode quote pool** — Fixed fun_pool to include ALL quotes from all tags instead of just one tone. Fun mode now relieves actual quotes from quotes.md instead of hallucinating.

**Mentats critic temperature** — Added wrapper forcing temperature=0.1 for critic role (Step 2 fact-checking). Eliminates hallucinations where critic accepts made-up facts.

**Vodka warm-up** — Eager-load facts.json on Filter initialization instead of lazy-loading. Fixed quirk where `?? list` showed only KBs on first call, then showed full memory store after any write operation.

**Vision mode persistence** — Fixed auto-vision detection checking entire history instead of just current message. After `##ocr` or image processing, router now correctly reverts to serious mode on next message.

**Vision/Fun sticky reversion** — Fixed sticky modes (`>>fun`, `>>fr`, `>>raw`) now correctly revert when switching modes or completing pipelines.

### ⚙️ Configuration

**Vault chunking optimization** — Added configurable chunk size for Qdrant ingestion:
```yaml
vault:
  chunk_words: 250        # Down from 600 (better semantic matching)
  chunk_overlap_words: 50  # Prevents context loss at boundaries
```

### 📚 Documentation

**Command cheat sheet** — Added sidecar utilities section:
- `>>calc` — Safe math expressions
- `>>list` — List Vodka memories
- `>>find` — Search filesystem KBs
- `>>flush` — Reset CTC cache

**Vodka commands documented** — Added `!!` (store), `!! nuke` (delete all), `!! forget` (delete matching), `??` (query).

**Clear distinction** — Documented difference: `!! nuke` (delete memory) vs `>>flush` (reset CTC cache).

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
- Additionally, use a large-en-v1 model

### Chunking Trade-offs
- Smaller chunks (250 words) improve semantic matching but increase Qdrant store size
- Larger chunks (600+ words) reduce storage but hurt retrieval precision
- Current setting (250) optimized for medical/knowledge domains

---

## Roadmap (Ideas)

- [ ] Better embedding model (nomic-embed-text or bge-large-en-v1.5)
- [ ] User-provided reference documents without summarization
- [ ] Session-level fact deduplication
- [ ] Multi-language support (Vodka memory)
- [ ] Web UI integration (hooks for `>>` and `??` commands)

---

## Installation / Upgrading

```bash
# Fresh install
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git

# Upgrade existing
pip install --upgrade git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

**New in this version:** Update `router_config.yaml` with `vault.chunk_words` and `vault.chunk_overlap_words` for better Qdrant performance. Old configs will use defaults (600/175).

---

## Questions?

- Run `>>help` in-chat for command reference
- Check `mentats_debug.log` for deep reasoning traces
- See [FAQ](FAQ.md) for architecture & troubleshooting
