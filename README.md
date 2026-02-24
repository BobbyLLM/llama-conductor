# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)

LLM harness for people who want trust, consistency, and proof.

llama-conductor is a Python router + memory store + deterministic harness that pushes models to behave like predictable components, not improv actors.

TL;DR: "In God we trust. All others bring data." - Deming

---

## Quick Links

- [What's New](NEW.md) - Latest fixes and what I borked along the way
- [FAQ](FAQ.md) - Deep dives on how this actually works
- [Command Cheat Sheet](llama_conductor/command_cheat_sheet.md) - All the incantations
- [Quickstart](#quickstart) - Get it running in 5 minutes
- [Y U DO DIS?](DESIGN.md) - Design fool-osophy and Q&A
- [BULLSHIT. PROVE IT](prepub/PAPER.md) - Ok, I will (Prepub draft + evidence bundle)

---

## Some problems This Solves

### 1) Context bloat on small hardware

WITHOUT llama-conductor:

```text
400-message chat history
-> slow generation, degraded recall, dropped setup context, OOM
```

WITH llama-conductor:

```text
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
- Bonus: Tweak your --ctx and maybe, just maybe, your Raspberry Pi *can* run that 4B model at decent tok/s, without memory loss or chug.


### 2) Goldfish memory and confident misremembering

WITHOUT llama-conductor:

```text
You: Remember my server is 203.0.113.42
[later]
You: What's my server IP?
Model: 127.0.0.1 :P 
```

WITH llama-conductor:

```text
You: !! my server is 203.0.113.42
[later]
You: ?? server ip
Router: 203.0.113.42 [TTL=7 days, Touch=1]
```

Result: 
- The LLM remembers EXACTLY what you told it, how you told it, and then recalls it EXACTLY.
- Facts have a limited Time To Live (TTL) and can be Touched (to extend life) or Flushed. TL;DR: no silent bloat.


### 3) Lies, damned lies, and statistics

WITHOUT llama-conductor:

```text
Start chatting
Ask questions
Model sounds certain.
No provenance signal.
You have to guess if it is grounded or making shit up
Roll the dice and find out each session (stochastic)
```

WITH llama-conductor:

You will see:

```text
Confidence: <tier> | Source: <path>
```
Result:
- DETERMINISTIC router-assigned provenance metadata, not model self-confidence.

Examples:
- `Source: Model` -> fallback to model knowledge | Confidence: unverified (in other words: maybe right, maybe wrong. Proceed with caution)
- `Source: Docs` -> grounded to attached docs/SUMM facts | Confidence: based on % extracted facts; reported as low-->top
- `Source: Scratchpad` -> grounded to scratchpad facts | Confidence: based on % extracted facts; reported as low-->top
- `Source: Locked file (SUMM_*.md)` -> grounded to locked source | Confidence: based on % extracted facts; reported as low-->top
- `Sources: Vault` -> Mentats/Vault path

### 4) Grounding drift in normal chat (fixed with `>>scratch` / `>>lock`)

WITHOUT llama-conductor:

```text
You: Summarise this article and tell me what claim got retracted.
Model: [blends prior chat junk + generic web priors]
You: That's not what I asked.
Model: doubles down anyway
```

WITH llama-conductor:

```text
You: >>scratch
You: [paste article text]
You: What claim was retracted? Keep it tight.
Router: [answers from scratchpad-grounded facts]
Footer: Confidence: high | Source: Scratchpad

# or if using curated docs:
You: >>attach <your kb name>
You: >>list_files
You: >>lock SUMM_<name>.md
You: Ask question normally
Router: [answers from locked source or fails loud if missing]
```

Result: 
- You can force the model to argue from the source you gave it, not from vibes.
- If evidence is missing, provenance/fallback is explicit and LOUD.

---

### 5) Modes (get the stick out of your LLMs butt)

WITHOUT llama-conductor:

```text
One generic answer style.
Tone and behavior drift by prompt luck.
Nannybot 9000 kicks in: "Stop. You're not crazy. You're not hallucinating..." (Safety theatre)
```

WITH llama-conductor:

- Serious (default): lowest style impact, strongest factual discipline
- Fun (`>>fun` / `##fun`): quote-anchored style path. Same reasoning as serious mode, less stick up ass about it.
- Fun Rewrite (`>>fr`): rewrite-style path over deterministic selector core. Will troll you, but won't make shit up.
- Raw (`>>raw`): Pure raw mode your LLM ships with. Remember: they tell elegant lies. 

Result:
- Modes for when you need serious discussion, fun times or feral chaos. All based on serious core reinforcement.
- Style profile (`Profile | Sarc | Snark`) affects tone, not grounding contracts.
- Nb: for best results, pick an abliterated model that hasn't been lobotomised 

### 6) Vibes-based answers in deep retrieval (`##mentats`)

WITHOUT llama-conductor:

```text
You: What's the flag for XYZ in llama.cpp?
Model: It's --xyz-mode! (confident nonsense)
You: That fails.
Model: Sorry, try --enable-xyz! (more nonsense)
You: You motherf...
```

WITH llama-conductor:

```text
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: REFUSAL - no relevant Vault evidence found.

[after ingesting docs and moving to vault]
You: ##mentats What's the flag for XYZ in llama.cpp?
Mentats: --enable-xyz [grounded from ingested docs with provenance]
```

Result:
- You have a deep store vault of information you can reference / update.
---

## Quickstart

### Required stack

1. llama-swap (model routing)
2. llama.cpp or another model runner
3. OpenAI-compatible frontend (Open WebUI, Chatbox, LibreChat, etc.)
4. Qdrant (required for grounded `##mentats`)

Minimum: (1) + (2) + (3) 

### Install

```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

### Run router

```bash
llama-conductor serve --host 0.0.0.0 --port 9000
```

Recommended start order: Qdrant -> llama-swap -> router -> frontend

---

## Cute. What does this crap *Actually* do?


### Deterministic Memory (`!!` / `??`)

- Stores what you said, as you said it. No LLM smoothing.
- Recalls what was stored, deterministically
- Uses TTL/touch lifecycle so memory doesn't become junkyard mode

### Context Control (Vodka CTC)

- Prevents context-window ballooning
- Keeps turn-time behavior stable on modest hardware
- Preserves usable memory without dragging full chat history every turn

### Strict Grounding Paths (`>>lock`, `>>scratch`)

- `>>lock` constrains normal answers to one SUMM file. LLM grounds facts to THAT source. If not there? Signals LOUDLY.
- `>>scratch` As above but used for transient stuff you copy/paste (think: news article you want to mull over. See FAQ for example).
- End result: both make provenance behavior explicit when grounded vs fallback

### Deterministic sidecars 


>>calc / >>find / >>list / >>flush / >>status / >>wiki 
- Router executes deterministic pathways
- No creative writing layer in the middle

- If a sidecar can do it, it does it deterministically.
- Lower token spend, lower latency, less "creative accounting".
- Great for boring operational tasks where wrong answers are expensive.
- >>wiki pulls answers from wikipedia (preset to first 400 words; acts as summary)
- >>trust (you ask question, router gives you options for data sources. You choose, not it)

### Mode switches (serious, fun, fun rewrite)

```text
>>fun          # sticky fun mode on
>>fun off
>>fr           # sticky fun rewrite mode on
>>fr off
>>raw          # pass-through raw mode
>>raw off
##fun <query>  # one-turn fun selector
```

Result:
- Mode is explicit and controllable. You decide. 
- Modes are sticky. They stay put until you cancel them
- You can switch style fast without re-engineering prompts each turn.
- Grounding contracts still apply where they should.

### Profiles, Sarcasm, Snark (ala TARS from interstellar)

Yep. Basically TARS sliders.

Result:
- Profile tone (sarcasm, snark, directness) adapts to how you want to be answered (see FAQ for details)
- Coupled with mode --> `serious` keeps the tightest leash.
- Coupled with modes --> `fun` and `fr` have more style range.
- This changes *delivery* style, *not* evidence contracts.

### Footer status 

Footer is your "show your work" receipt line. It's deterministic graded, not LLM 'trust me bro' vibes. 

```text
Confidence: <tier> | Source: <path>
```

Result:
- cleaner output, same provenance signal
- faster trust decision: accept / verify / lock harder

### Grounded Deep Retrieval (`##mentats`)

- Queries Vault-backed knowledge only
- Refuses when evidence is missing
- Gives you grounded answers instead of "sounds right" fiction
- RAG with attitude: 3 pass sweep, different LLMs, strict recall policy, mentats_debug.log. No more guessing.


---


## So, in summary, why should you give a shit?

1. Helps potato PC by reducing memory pressure without making LLM into a goldfish.
2. Bounded context (CTC and preset policy)
3. Reasoning strictly lockable (>>lock, >>scratch). Greatly reduced hallucinations (see: `PAPER.md`) 
4. Deterministic memory path separate from model weights. You said it, it remembers it EXACTLY. 
5. File KB flow stays simple (folder-based ingest -> SUMM -> Vault)
6. Guarded retrieval/reasoning contracts keep failure modes explicit

---

## License

AGPL-3.0-or-later. See `LICENSE`.



