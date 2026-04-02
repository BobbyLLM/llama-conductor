# llama-conductor
![llama-conductor banner](logo/zardoz.jpg)

LLM harness for people who want trust, consistency, and proof.

LLMs hallucinate, forget, and sound confident either way. llama-conductor forces them to bring receipts: grounded answers, deterministic memory, and explicit refusal when evidence is missing

TL;DR: "In God we trust. All others bring data." - Deming

---

## Quick Links

### Start Here
- 🚀 [5-Minute Quickstart](#5-minute-quickstart) - Get this running now, not next week.
- 🤨 [Why This Exists](DESIGN.md) - The "why" behind this whole thing.
- 📖 [What's New](NEW.md) - Latest fixes, updates, and "what I borked along the way".

### Using It
- ⌨️ [Command Cheat Sheet](llama_conductor/command_cheat_sheet.md) - All the incantations.
- ❓ [FAQ](FAQ.md) - Everything you wanted to know on how it works...and then some. No secrets. Glass-box, not black box.
- 🛠️ [Setup + Config Details](FAQ.md#technical-setup) - Full wiring and troubleshooting.

### Evidence and Research
- 📊 [Prepub Paper](prepub/PAPER.md) - "Bullshit. Prove it". Ok, here then.
- 🚗 [Meme Test](https://bobbyllm.github.io/llama-conductor/blog/meme-test/) - Mom, can we get ChatGPT? Mom: we have ChatGPT at home.
- 🧠 [Blog & Updates](https://bobbyllm.github.io/llama-conductor/) - Blog posts, roadmaps, and Deep Thoughts (tm). Also, me swearing at Python code.

---

## 🚀 5-Minute Quickstart

Pick one path and go.

1. Install llama-conductor:
```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

2. Start the stack (pick one):
- Bare metal:
```bash
python -m llama_conductor.launch_stack up --config llama_conductor/router_config.yaml
```
- Docker Compose:
```bash
docker compose up -d
```

3. Open the UI:
- `http://127.0.0.1:8088`

4. Optional (but nice): Firefox bridge
- Load from `extras/firefox-extension/` (`moa chat bridge.xpi`)

That’s it. You’re live.

Want the *full* wiring (models, config, Qdrant, troubleshooting)?  
Jump to [Quickstart (First-Time, Recommended)](#quickstart-first-time-recommended).

---

## 🛠️ Some problems This Solves

### 1) 🧠 Context bloat on small hardware

WITHOUT llama-conductor:

```text
400-message chat history
-> VRAM spikes, tok/s craters, model forgets its own name, OOM
-> you restart the chat and lose everything
-> rinse, repeat, contemplate career change
```

WITH llama-conductor:

```text
Vodka CTC trims context automatically:
- keeps the recent turns that matter
- hard-caps prompt growth so VRAM stays predictable
- drops mid-chat bloat before it kills your GPU
- keeps memory available through deterministic recall (not stuffed into prompt)
- user definable presets (fast / balanced / max-recall)
```

Result: 
- Consistent prompt size, stable performance, and optional rolling deterministic summary (stdlib extractive, no LLM compute).
- Tok/s you started with is tok/s you keep. No "why is it slow now" twenty minutes in.
- Bonus: Tweak your `--ctx` and maybe, just maybe, your Raspberry Pi *can* run that 4B model without chug. Your electricity bill called. It says you're welcome.


### 2) 🐠 Goldfish memory and confident misremembering

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

---

### 3) 🧾 Lies, damned lies, and LLM's

WITHOUT llama-conductor:
```text
You: Is paracetamol safe to take with ibuprofen?
Model: Yes, they can be safely combined as they work through different mechanisms.
You: Is that actually true or are you guessing?
Model: That is generally accepted medical guidance. (no source, no provenance, vibes all the way down)
```

WITH llama-conductor:
```text
Confidence: <tier> | Source: <path>
```

Every response. Every time. Assigned by the router, not the model.

Result:
- The model doesn't grade its own homework. The router does.
- You know immediately whether to trust it, verify it, or lock it harder.

Sources:
- `Source: Model` → fallback to model weights | Confidence: unverified. Maybe right. Maybe not. Proceed accordingly.
- `Source: Docs` → grounded to attached docs/SUMM facts | Confidence: based on % facts extracted
- `Source: Scratchpad` → grounded to what you pasted | Confidence: based on % facts extracted
- `Source: Locked file (SUMM_*.md)` → grounded to locked source | Confidence: based on % facts extracted
- `Source: Vault` → ##mentats path, grounded or refused
- `Source: Cheatsheets` → grounded to your definitions | Confidence: high (you told it, it parrots it back)
- `Source: Mixed` → touched multiple paths. Footer tells you which ones.
- `Source: Web` → grounded to live web retrieval | Confidence: based on relevance gate score. Comes with a `See: <url>` receipt.

### 4) 📌 Grounding drift in normal chat (fixed with >>scratch / >>lock)

WITHOUT llama-conductor:
```text
You: What does the article say the fine cap is?
Model: Under the AI Act, fines can reach €35 million or 7% of global turnover.
You: That's not in the article.
Model: You're right, the article specifies a different amount.
You: ...what amount?
Model: The article does not appear to state a specific figure.
You: FFS. It's *in* the second paragraph. Why do you exist?
```

WITH llama-conductor:
```text
You: >>add [paste article text]
You: >>scratch lock <n>
You: What does the article say the fine cap is?
Router: [answers from article only, not training data]
Footer: Confidence: high | Source: Scratchpad
```

Result:
- Model argues from what you gave it, not from what it half-remembers from training.
- If the answer isn't in the source, it says so. Explicitly. Not "I may have drawn on broader context."
- You can >> lock up to 12 items and force it to reason across all of them. Ask it to compare - it'll flag contradictions and gaps. 
- Still loud when evidence is missing.
- When you're done? Throw it away with >>detach all. If you forget, it auto cleans next boot. 

### 5) 🎛️ Modes (get the stick out of your LLMs butt)

WITHOUT llama-conductor:
```text
You: stop being so formal, just talk normally
Model: Of course! So basically the deal is... [casual, this turn only]
You: ok so what about X?
Model: Certainly! X is a multifaceted concept that encompasses— [formal again]
You: dude.
Model: My apologies! To put it more plainly... [casual, this turn only]
[repeat forever]
```

WITH llama-conductor:
```text
You: >>fun
You: Can you legally fire sentient yeast?
Assistant: No. The Bread Labor Relations Act of 2024 does not exist. Sentient yeast is not a protected class. Feed it or bin it. That's it.
Confidence: top | Source: Model

You: >>fr
You: Can you legally fire sentient yeast?
[FUN REWRITE] "I don't want to live on this planet anymore."
Your sourdough started a union. Cool. Legally? Zero rights. No statute covers sentient yeast. Just... feed it and move on. Or don't. I'm not your union rep.
Confidence: high | Source: Model
```

Result:
- Set it once. It stays set. Sick of it? Turn it off. Same turn, no need for new chat.
- Same grounding contracts underneath — style changes delivery, not evidence.
- TARS sliders (`Profile | Sarc | Snark`) tune the edge. Style adapts to how you talk to it.
- Nb: for best results, pick an abliterated model that hasn't been lobotomised.
---

### 6) 🧪 Vibes-based answers (your model is a stochastic idiot)

WITHOUT llama-conductor:
```text
You: What's vibe coding?
Model: It's a development methodology focused on aesthetic consistency and psychological flow states in team environments. Popularised by Extreme Programming. (confident, authoritative, completely wrong)
You: WTAF? Dude, no.
Model: You're right, apologies — it's actually a minimalist approach to code style, inspired by jazz improvisation. (still wrong, still confident)
You: ...
```

WITH llama-conductor:
```text
You: What's vibe coding?
Assistant: Contemporary term for AI-assisted coding where the developer describes intent in natural language and accepts AI-generated code without fully understanding it. Named for the workflow feel -- you vibe with the AI rather than write line by line. Distinct from pair programming: vibe coding often involves accepting output you can't fully verify.
Confidence: high | Source: Cheatsheets
```

Result:
- Known terms stop drifting to model priors.
- Local definitions are deterministic and editable in one file.
- You tell it once. It stays told. Priority: your definition → wiki → stochastic parrot.
- Broken or missing row? Fails loud. 
- Term in context window → deterministic re-grounding. Term gone → normal context rules. Don't like that? Pull it again.
- Footer provenance makes source path explicit instead of making you guess.

---

### 7) 🏛️ Grounded deep retrieval (##mentats) — grounds or refuses, nothing in between

WITHOUT llama-conductor:
```text
You: ##mentats What did the Kaltenborn study say about Grade III mobilisation?
Model: Kaltenborn recommends Grade III for acute inflammatory conditions. (fabricated, cites nothing)
You: That's the opposite of what it says.
Model: You raise a good point.
You: You useless, motherf...
```

WITH llama-conductor:
```text
You: ##mentats What did the Kaltenborn study say about Grade III mobilisation?
Mentats: FINAL_ANSWER: No Vault evidence found for this. Sources: Vault | FACTS_USED: NONE [ZARDOZ HATH SPOKEN]

[after ingesting docs and moving to vault]
You: ##mentats What did the Kaltenborn study say about Grade III mobilisation?
Mentats: Grade III mobilisation indicated for stiff joints, contraindicated in acute inflammation. [grounded from ingested docs with provenance]
```

Result:
- No Vault evidence? Explicit refusal. Not "I think..." — hard stop.
- Grounded answers from YOUR ingested docs, not model weights cosplaying as a textbook.
- 3-pass sweep, different LLMs on each pass, strict recall policy. `mentats_debug.log` if you want to see the work.
- Update your vault, answers update with it. Single source of truth you control.
- RAG with attitude. No more guessing whether it made that citation up.

---

### 8) 🎲 Vibes-based ranking ("trust me bro" but make it stochastic)

WITHOUT llama-conductor:

```text
You: Which is better for my use case, X or Y?
Model: X.
You: You sure?
Model: Hmmm....actually Y.
You: Why?
Model: Because reasons.
```

WITH llama-conductor:

```text
You: >>judge [criterion] : X, Y --verbose
[judge] ranking
1. X (score=2.00)
2. Y (score=0.00)
Judge confidence: high
audit_jsonl: total_recall/judge/judge_audit_<timestamp>.jsonl
```
Result:
What you actually get (instead of cosplay certainty):

- Pairwise both directions (A/B then B/A), so position bias gets caught, not hidden.
- Confidence is based on agreement pattern, not how swaggery the prose sounds.
- Want to look at why? --verbose gives you a JSONL audit trail with provenance.
- Attach scratchpad and judge evaluates from evidence, not vibes. Locked evidence unusable? Fail-closed.
- Read that again slowly. IT. JUDGES. FROM. EVIDENCE. NOT. VIBES. Even a 1.7B could do this. Fast.

---

### 9) 🌐 "I'm sure it's from that movie..." (your model is a confident liar about things Google knows)

WITHOUT llama-conductor:
```text
You: Where does the quote "I yam what I yam" come from?
Model: The quote "I yam what I yam" is from the 1939 musical The Wizard of Oz.
       It is sung by the character Glinda, the Good Witch of the South,
       in the song "I'm a Little Teapot."
You: ...
You: I need a drink.
```

WITH llama-conductor:
```text
You: Where does the quote "I yam what I yam" come from?
Assistant: I Yam What I Yam is the second Popeye theatrical cartoon short.
The source of the quote is the comic strip Thimble Theatre by E. C. Segar,
in which Popeye first appeared.
See: https://en.wikipedia.org/wiki/I_Yam_What_I_Yam
Confidence: medium | Source: Web
```

Result:
- When cheatsheets and wiki can't answer, the router now searches the web before letting the model guess.
- Retrieval cascade: `Cheatsheets → Wiki → Web → Model`. Each step fires only if the previous one missed.
- Deterministic relevance gate scores every result (phrase match + token overlap + domain trust). Garbage results get rejected, not served.
- `See: <url>` gives you the actual source link. One click to verify. Receipts, not pinky promises.
- Model is still last resort. If all retrieval fails, you get `Confidence: unverified | Source: Model` and you know exactly what you're dealing with.
- Want to search manually? `>>web <query>` works standalone for anything.
- Add your own trusted domains in config. BBC? Reuters? PubMed? Your call. Built-in defaults stay active either way.

---

## Quickstart (First-Time, Recommended)

### [Step 0] Prerequisites

- Python 3.10+ (recommended: 3.10-3.12 for first-time stability):
  - https://www.python.org/downloads/
- llama.cpp (`llama-server`) + at least one GGUF model (for example Qwen3-4B; TWO OR MORE is better - see [What is Mentats?](FAQ.md#what-is-mentats) for why):
  - https://github.com/ggml-org/llama.cpp
  - https://huggingface.co/unsloth
- Frontend:
  - llama.cpp WebUI + shim (recommended; WebUI ships with llama.cpp)
  - or any OpenAI-compatible client (OWUI, LibreChat, etc.)
- Optional for full stack:
  - Qdrant (REQUIRED for Vault/`##mentats` and full stack; OPTIONAL for kick-the-tires mode)
  - https://github.com/qdrant/qdrant

### [Step 1] Install llama-conductor

```bash
pip install git+https://codeberg.org/BobbyLLM/llama-conductor.git
```

### [Step 2] Configure `llama_conductor/router_config.yaml`

- Set `backend.provider: "llama_cpp"`
- Set:
  - `backend.llama_cpp.exe_path` = full filesystem path to your `llama-server` executable (where you installed llama.cpp)
  - `backend.llama_cpp.models_dir` = full filesystem path to your GGUF models folder
- Path examples:
  - Windows:
    - `backend.llama_cpp.exe_path: "C:/path/to/llama.cpp/llama-server.exe"` (edit to match your own)
    - `backend.llama_cpp.models_dir: "C:/path/to/LLMs"` (edit to match your own)
  - Linux/macOS:
    - `backend.llama_cpp.exe_path: "/path/to/llama.cpp/llama-server"` (edit to match your own)
    - `backend.llama_cpp.models_dir: "/path/to/models"` (edit to match your own)
- Set `roles.*` to model IDs from backend `/v1/models`.
  - Use different model IDs for `thinker`, `critic`, and `coder` for better results.
  - For minimal kick-the-tires mode only (not recommended), you can temporarily use the same model ID across roles.
- Vision/OCR setup (optional):
  - Set `roles.vision` to a real vision-language model ID (not a text-only model).
  - Add matching `mmproj` for that same model in `llama_server.models_preset.models`.
  - If you skip this, text chat still works, but image/OCR will not be reliable.
- Tip: open backend `/v1/models` in a browser and copy `id` values exactly.

More config detail:
- [FAQ: Launch Script: The Easy Way](FAQ.md#launch-script-the-easy-way)
- [FAQ: Config knobs (router_config.yaml) with examples](FAQ.md#config-knobs-router_configyaml-with-examples)

### [Step 3] Start stack

Launch core stack:

```bash
python -m llama_conductor.launch_stack up --config llama_conductor/router_config.yaml
```

For full stack (Vault/`##mentats`), start Qdrant (pick one):

- Docker (Windows):
```bat
docker start qdrant >nul 2>&1 || docker run --name qdrant -p 6333:6333 -d qdrant/qdrant
```
- Docker (Linux/macOS):
```bash
docker start qdrant >/dev/null 2>&1 || docker run --name qdrant -p 6333:6333 -d qdrant/qdrant
```
- Bare-metal:
  - start your local Qdrant service/binary (for example from `C:\Qdrant`)

Need more launch variants? See [FAQ: Launch Script: The Easy Way](FAQ.md#launch-script-the-easy-way).

### [Step 3A] First-time launcher setup (recommended)

The repo ships a stdlib-only stack supervisor at `tools/start_stack.py`. On first run, it writes a one-time local config for your machine paths.

```bash
python tools/start_stack.py --setup
```

Verify it resolved correctly before launching anything:

```bash
python tools/start_stack.py --doctor
```

Then launch as normal:

```bash
python tools/start_stack.py
```

Windows users: `START-ALL.bat` in the repo root calls this for you. Desktop shortcut optional.

- `stack.local.yaml` is your machine config - gitignored, never touches the repo.
- Leave overrides blank during `--setup` to inherit paths from `router_config.yaml`.
- Qdrant is optional. If absent or disabled, core stack still launches - you just lose `##mentats`/Vault paths.

---

### [Step 3B] Docker Compose (single-host)

If you prefer Docker Compose, this repo ships a ready baseline:

```bash
docker compose up -d
```

What it includes by default:
- `qdrant` (for Vault/mentats paths)
- `llama-conductor` router service
- optional `open-webui` profile (only if you enable it)

Useful commands:

```bash
# Start core stack
docker compose up -d

# Include optional Open WebUI
docker compose --profile webui up -d

# See logs
docker compose logs -f

# Stop stack
docker compose down
```

Config pointers:
- `docker-compose.yml` for service wiring
- `docker.env.example` for environment defaults
- `docker/router_config.docker.yaml` for container-friendly router config

### [Step 4] Open and enjoy

- http://127.0.0.1:8088

- Enjoy :)

### Kick-the-tires mode

- Skip Qdrant and run llama.cpp + web-ui (with shim) only.
- Nb: core chat/routing works; Vault/`##mentats` does not.

### NB (troubleshooting only)

- Router models: http://127.0.0.1:9000/v1/models
- Shim health: http://127.0.0.1:8088/shim/healthz

---

## ⚙️ Cute. What does this crap *Actually* do?


### 🧩 Deterministic Memory (!! / ??)

- Stores what you said, as you said it. No LLM smoothing.
- Recalls what was stored, deterministically
- Uses TTL/touch lifecycle so memory doesn't become junkyard mode

### ✂️ Context Control (Vodka CTC)

- Prevents context-window ballooning
- Keeps turn-time behavior stable on modest hardware
- Preserves usable memory without dragging full chat history every turn

### 🎯 Strict Grounding Paths (>>lock, >>scratch)

- `>>lock` constrains normal answers to one SUMM file. LLM grounds facts to THAT source. If not there? Signals LOUDLY.
- `>>scratch` As above but used for transient stuff you copy/paste (think: news article you want to mull over. [See FAQ for example](https://codeberg.org/BobbyLLM/llama-conductor/src/branch/main/FAQ.md#scratchpad-deep-example)).
- End result: both make provenance behavior explicit when grounded vs fallback

### 🧰 Deterministic sidecars


>>calc / >>find / >>list / >>flush / >>status / >>wiki / >>web
- Router executes deterministic pathways
- No creative writing layer in the middle

- If a sidecar can do it, it does it deterministically.
- Lower token spend, lower latency, less "creative accounting".
- Great for boring operational tasks where wrong answers are expensive.
- >>wiki pulls answers from wikipedia (preset to first 400 words; acts as summary)
- >>trust (you ask question, router gives you options for data sources. You choose, not it)
- >>web pulls answers from the live web with deterministic relevance scoring (DuckDuckGo, or bring your own provider)

### 🎚️ Mode switches (serious, fun, fun rewrite)

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

### 🤖 Profiles, Sarcasm, Snark (ala TARS from interstellar)

Yep. Basically TARS sliders.

Result:
- Profile tone (sarcasm, snark, directness) adapts to how you want to be answered (see [FAQ for details](https://codeberg.org/BobbyLLM/llama-conductor/src/branch/main/FAQ.md#what-does-profile-sarc-snark-mean))
- Coupled with mode --> `serious` keeps the tightest leash.
- Coupled with modes --> `fun` and `fr` have more style range.
- This changes *delivery* style, *not* evidence contracts.

### 🧷 Footer status

Footer is your "show your work" receipt line. It's deterministic graded, not LLM 'trust me bro' vibes. 

```text
Confidence: <tier> | Source: <path>
```

Result:
- cleaner output, same provenance signal
- faster trust decision: accept / verify / lock harder

### 🏛️ Grounded Deep Retrieval (##mentats)

- Queries Vault-backed knowledge only
- Refuses when evidence is missing
- Gives you grounded answers instead of "sounds right" fiction
- RAG with attitude: 3 pass sweep, different LLMs, strict recall policy, mentats_debug.log. No more guessing.


---


## ✅ So, in summary, why should you give a shit?

1. Helps potato PC by reducing memory pressure without making LLM into a goldfish.
2. Bounded context (CTC and preset policy)
3. Reasoning strictly lockable (>>lock, >>scratch). Greatly reduced hallucinations (see: [PAPER.md](prepub/PAPER.md))
4. Deterministic memory path separate from model weights. You said it, it remembers it EXACTLY. 
5. File KB flow stays simple (folder-based ingest -> SUMM -> Vault)
6. Guarded retrieval/reasoning contracts keep failure modes explicit
7. Web retrieval with relevance gating - when local knowledge runs out, it searches before guessing. And shows you the source URL.

---

## 📜 License

AGPL-3.0-or-later. See `LICENSE`.
