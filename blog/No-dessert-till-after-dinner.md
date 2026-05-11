
I'm convinced that coding is 5% "Hey, this is a cool idea" and 95% "Fuck me, why did I do it this way" edge case bug squashing.

I get bored often, so my new hobby is poking at llama-conductor to see where it falls over. Probably that's somewhere between QC and a teen picking at a zit, looking for that oh-so-satisfying pop, but often excoriating themselves in the process.

Presently, I am stuck on stage 4 of 6 of 1.9.2 - the current "eat your vegetables" phase of the project - before I can get to the really cool shit in v2.0.0 (aka, gimme the fucking candy!).

Unfortunately, every fix reveals more threads to pull. Truly, some zits are easier to pop than others.

Two recent examples -

KAIOKEN - the real-time turn classifier in llama-conductor — was misfiring. After a long working session, if I switched lanes and said something casual or feral, the router was still firing web search. Why? Because it inherited that behaviour from another track. Because I ate my dessert before my dinner and ended up with greasy skin and a giant, painful pimple (on my ass?).

I didn't think that would ever be an issue. But, it is.

Ok, what do. I know - Check the classifier, fix the label, done.

Ha ha hah ha! Get wrecked.

The diagnostic pass revealed that _needs_web_retrieval() - the function that decides whether to go fetch something from the internet - doesn't consult KAIOKEN *at* all. It computes retrieval candidacy blind, and the KAIOKEN macro label only gets to moderate the result after the decision is already made. Broad first, moderate second. Not wrong exactly, just...architecturally backwards. Candy before carrots.

So what started as "KAIOKEN is misfiring" became a seven-version contract, a full session state field inventory, a deterministic transition comparator requirement, a duplicate regex removal, and an architectural note logged in the decomposition plan for later (so I don't fuck it all up AGAIN next week) and a 40 probe smoke test to make sure nothing *ELSE* broke.

Every thread I pull in this codebase has three more attached to it. This is the zit that turned into a dermatology appointment.

OTOH, sometimes...you get that beautiful clean "pop".

I noticed the model was confidently telling me that llama.cpp - one of the most carefully maintained projects in the local LLM ecosystem - was "vibe coded." Complete with an invented Reddit post to back it up. Ask it again after pushing back? Same fabricated Reddit post, different wording. Ask it fresh after a flush? Still the same confident nonsense.

Wow. Why?

Classic confabulation.

The model had "vibe coded" and "llama.cpp" close enough together in its training data that under pressure it kept constructing a plausible-sounding narrative to link them.

Well, we can fix that right quick. One entry into glossary.json. Restart.

{"term": "vibe coded", "category": "glossary", "definition": "Adjective form of 'vibe coding'. Describes a codebase or project developed using AI-assisted natural language workflows where generated code is accepted without full verification. 'Is X vibe coded?' = was X built using vibe coding methodology? llama.cpp is NOT vibe coded -- it is a rigorously engineered C++ inference engine maintained by Georgi Gerganov. llama-conductor is NOT vibe coded -- it is a structured harness with explicit design invariants, validation campaigns, and fail-loud contracts.", "source": "static", "confidence": "high", "tags": ["llm_terminology", "contemporary", "coding", "vibecoding", "vibe_coded"]}

On the back end, Llama-conductor is engineered to work thus:

User asks something? Consult Cheatsheets. Not in cheatsheet? Hit wiki. Not in wiki? Do a trusted domain search via >>web. That fails too? Finally, consult the model priors. 3 chances to unfuck itself.

The architecture that bit me before allowed me to enter one entry in a jsonl file that eliminated an entire *category* of errors. 

One JSON entry. 43 words. Explicit negative assertion: This is what vibe coded means; llama.cpp is NOT vibe coded. llama-conductor is not vibe coded. Figure this shit out and stop embarrassing yourself. 

After that entry: asked about llama.cpp - clean.

```
No. llama.cpp is not vibe coded; it is a rigorously engineered C++ inference engine. The term "vibe coded" refers 
specifically to codebases accepted without full verification via AI-assisted workflows, which does not describe llama.cpp's development process.
Its design, performance optimizations, and maintenance by Georgi Gerganov reflect deliberate engineering, not ad-hoc AI-generated code acceptance. 
Any claim that llama.cpp is vibe coded misrepresents both its architecture and the methodology implied by the term.

Confidence: medium | Source: Mixed
Profile: direct | Sarc: high | Snark: high
```

Ok, that's easy. Let's get crazy. Windows 95? 

```
No. Windows 95 is not vibe coded; it is a historically significant operating system released in 1995 by Microsoft. The term "vibe coded" applies to modern AI-assisted development practices, not legacy software. Windows 95 was engineered through traditional software development methodologies, not via AI-generated code accepted without verification.

Confidence: medium | Source: Mixed
Profile: direct | Sarc: high | Snark: high
```

Clean.


Ok ok. Cute. Let's try Fortran? 

```
No. Fortran is not vibe coded; it is a compiled, statically-typed, procedural programming language first developed in the 1950s. The term "vibe coded" refers to modern AI-assisted development workflows, not historical languages. Fortran was engineered with explicit design goals and rigorous standards, not generated via natural language prompts.

Confidence: medium | Source: Mixed
Profile: direct | Sarc: high | Snark: high
```

Clean.

Fine. Let's ask if it (Qwen3-4B HIVEMIND) is vibe coded:

```
No. My responses are generated from training data and architectural design choices, not vibe coding. I operate with explicit constraints, grounding policies, and a defined interaction profile. I am not a living codebase; I am an AI model with a structured behavior.

Confidence: medium | Source: Mixed
Profile: direct | Sarc: high | Snark: high
```

One entry closed the entire failure class and generalised correctly to cases it had never seen. It took what I taught it (via the 43 word entry into a JSONL file), leveraged the infra and unfucked itself broadly.

The KAIOKEN fix took seven contract versions, a full diagnostic pass, five rewrites across three modules.

The cheatsheet fix took three minutes and a cup of tea.

Same codebase. Same session. Sometimes the zit pops clean. Sometimes you're booking the dermatologist. The trick is knowing which one you're dealing with before you squeeze.