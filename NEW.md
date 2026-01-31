# What's New

## Latest (v1.0.11)

### 🎯 Core Improvements

**RAW mode context fix** — `>>raw` mode now includes CONTEXT block (conversation history) so queries like "what have we discussed?" work correctly instead of hallucinating. Previously `>>raw` was outputting non-Serious answers but without conversation context, causing it to fabricate topics from training data.

### 🔍 New Sidecars (Deterministic API Tools)

**Wikipedia summaries** — `>>wiki <topic>` fetches article openings via free Wikipedia JSON API
- Full paragraph summaries (~500 chars)
- No hallucination, requires valid article name
- Example: `>>wiki Albert Einstein` → full paragraph on relativity, Nobel Prize, etc.

**Currency conversion** — `>>exchange <query>` fetches real-time rates via Frankfurter API
- Supports natural language: `>>exchange 1 USD to EUR`, `>>exchange GBP to JPY`
- Common currency aliases (USD, EUR, GBP, JPY, AUD, CAD, etc.)
- Example: `>>exchange 1 AUD to USD` → `0.70 USD`

**Weather** — `>>weather <location>` fetches current conditions via Open-Meteo API
- Uses geocoding to find location, then fetches current weather
- Returns: temperature, condition (Clear/Cloudy/Rainy), humidity, wind speed
- No rate limiting (Open-Meteo free tier is generous)
- Works best with city names or "City Country" format
- Example: `>>weather Perth` → `Perth, Australia: 22°C, Clear sky, 64% humidity`

### 📚 Documentation

**Updated command cheat sheet** — Added full section for knowledge lookup sidecars with examples and usage notes
- Wiki, Exchange, Weather all documented with examples
- Clear notes on limitations (wiki needs valid article names, weather works best with city names)

### 🐛 Bugfixes

**Wiki User-Agent header** — Wikipedia blocks requests without explicit User-Agent; added proper header to fix 403 errors

**Weather API reliability** — Switched from wttr.in (aggressive rate limiting, 1 call per 5 mins) to Open-Meteo (10,000 calls/day free, reliable)

---

## v1.0.10

**Fun mode quote pool** — Fixed fun_pool to include ALL quotes from all tags instead of just one tone. Fun mode now retrieves actual quotes from quotes.md instead of hallucinating.

**Mentats critic temperature** — Added wrapper forcing temperature=0.1 for critic role (Step 2 fact-checking). Eliminates hallucinations where critic accepts made-up facts.

**Vodka warm-up** — Eager-load facts.json on Filter initialization instead of lazy-loading. Fixed quirk where `?? list` showed only KBs on first call, then showed full memory store after any write operation.

**Vision mode persistence** — Fixed auto-vision detection checking entire history instead of just current message. After `##ocr` or image processing, router now correctly reverts to serious mode on next message.

**Vision/Fun sticky reversion** — Fixed sticky modes (`>>fun`, `>>fr`, `>>raw`) now correctly revert when switching modes or completing pipelines.

**Vault chunking optimization** — Added configurable chunk size for Qdrant ingestion:
```yaml
vault:
  chunk_words: 250        # Down from 600 (better semantic matching)
  chunk_overlap_words: 50  # Prevents context loss at boundaries
```

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

### Chunking Trade-offs
- Smaller chunks (250 words) improve semantic matching but increase Qdrant store size
- Larger chunks (600+ words) reduce storage but hurt retrieval precision
- Current setting (250) optimized for medical/knowledge domains

### Weather Locations
- Open-Meteo geocoding works best with city names or "City Country" format
- Long/complex location strings may not resolve (e.g. "Perth Western Australia" → use "Perth")

---

## Roadmap (Ideas)

- [ ] Better embedding model (nomic-embed-text or bge-large-en-v1.5)
- [ ] User-provided reference documents without summarization
- [ ] Session-level fact deduplication
- [ ] Multi-language support (Vodka memory)
- [ ] Web UI integration (hooks for `>>` and `??` commands)
- [ ] More sidecars (>>define, >>convert units, etc.)

---

## Installation / Upgrading

```bash
# Fresh install
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git

# Upgrade existing
pip install --upgrade git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

**New in this version:** 
- Three new API sidecars: `>>wiki`, `>>exchange`, `>>weather`
- RAW mode now includes conversation context for better summary/disambiguation questions
- Update `router_config.yaml` with `vault.chunk_words` and `vault.chunk_overlap_words` for better Qdrant performance. Old configs will use defaults (600/175).

---

## Questions?

- Run `>>help` in-chat for full command reference
- Check `mentats_debug.log` for deep reasoning traces
- See [FAQ](FAQ.md) for architecture & troubleshooting
- Try `>>wiki`, `>>exchange`, `>>weather` for fact-based lookups (no hallucination)
