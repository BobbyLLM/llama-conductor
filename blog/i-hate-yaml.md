# I hate YAML.


After getting sick of baked-in, pre-trained model priors (No Qwen, that's NOT what vibe-coding means. That's not what abliteration means. That's not what any of this means!) I decided to add in a "clever little feature" Cheatsheets - see: [New](https://codeberg.org/BobbyLLM/llama-conductor/src/branch/main/NEW.md) and [FAQ](https://github.com/BobbyLLM/llama-conductor/blob/main/FAQ.md#cheatsheets-jsonl-grounding)

that allows users to add in new, high yield information on the fly. Literally, add the info into the file, save it, boom, no more stupid.

Well....I underestimated my own stupid.

For the sake of end users, I started this with the perfectly parsable, sensible YAML format.

**Attempt 1: "This should work" (narrator: it did not, in fact, work)**

- `term: Vibe coding`
- `category: glossary`
- `definition: ... "Remember - friends don't let friends Vibe code. Just say no" ...`
- `source: static`
- `confidence: high`
- `tags: llm_terminology, contemporary, coding`

Astute readers will notice the issue immediately. The quotes in the definition will probably break the YAML parser. YAML hates inline quotes without escaping. Bad YAML files = YAML file not work = Qwen defaults to stupid.

**Fine, take 2**

- `term: Vibe coding`
- `category: glossary`
- `definition: " ... \"Remember - friends don't let friends Vibe code. Just say no\" ... "`
- `source: static`
- `confidence: high`
- `tags: llm_terminology, contemporary, coding`

Problem: Now you've got escape backslashes polluting your text. If you want to add or edit the quote later, you have to remember to escape it. Miss one? Parser error. AGAIN!

At this point, I decide; fuck YAML. It's cute but brittle. Turns out, YAML being ass is a [known feature](https://noyaml.com/)

So, what do?

**JSONL: It "just works" (ish).**

By all rights, JSONL just works. It's ugly, but it works. One line. Parse or die. No indent hell. Right? Right?

Valid JSONL shape (clean and parseable):

- `term = Vibe coding`
- `category = glossary`
- `definition = ... \"Remember - friends don't let friends Vibe code. Just say no\" ...`
- `source = static`
- `confidence = high`
- `tags = llm_terminology, contemporary, coding`

Until Boy Wonder over here did the following:

Broken JSONL shape (one unescaped quote nukes the row):

- `term = Vibe coding`
- `category = glossary`
- `definition = ... "Remember - friends don't let friends Vibe code. Just say no" ...`
- `source = static`
- `confidence = high`
- `tags = llm_terminology, contemporary, coding`

Result? The JSON parser hits `..."you can't fully verify. "Remember...` and assumes the string ended. Then it expects a comma or closing brace, but instead sees `Remember`, which is not valid JSON syntax.

**Parser says:**

> JSONDecodeError: Expecting ',' delimiter: line 1 column 287

The world stops. Cheatsheets stop working. Qwen (silently) reverts to stupid.

Example bad output after that parse fail:

> User: What is vibe coding?
>
> Assistant: Vibe coding is a lightweight, informal coding style that prioritizes readability and developer comfort over strict conventions...
>
> Confidence: unverified | Source: Model

FML.


**The footer, stupid. Look at the footer**

I assumed (correctly) that I would break things. So past me had to save present me from stupid.

In this case, the footer immediately shows the issue.

`Confidence: unverified | Source: Model`

Which is llama-conductor-speak for "yo, your shit's fucked and the JSONL didn't load".

Fixing the unescaped `"` gives us:

> Assistant: Vibe coding: Contemporary term for AI-assisted coding where the developer describes intent in natural language and accepts AI-generated code without fully understanding it. Named for the workflow feel -- you vibe with the AI rather than write line by line. Distinct from pair programming: vibe coding often involves accepting output you can't fully verify. "Remember - friends don't let friends Vibe code. Just say no" - BobbyLLM.
>
> Confidence: high | Source: Cheatsheets

**The moral of the story**

1) To quote Winter Soldier "Bucky: Don't do anything stupid. Cap: How can I - you're taking all the stupid with you".
2) Always assume that [PEBKAC](https://en.wikipedia.org/wiki/User_error) is the first fail point
3) Architect around it


- Router now pre-loads cheatsheets on startup (pre-flight load), and the loader records parse errors as structured warnings.
- If a JSONL line is malformed, app still runs, but the user gets a visible warning with exact `file:line:error` so they can fix it immediately.
- Warning is non-spammy (shown once per warning signature, not every turn forever).

**The way forward**
- Resolution order stays: **Cheatsheets (Track A) -> Wiki sidecar (once I get off my ass and cross-link it) -> Model priors**.
- If JSONL is borked, it fails loud instead of silently pretending everything is fine.
- So if local fact exists, it wins. If it doesn't, we hit wiki. If JSONL is borked, we hit wiki.

That's me doing my best to architect around my own stupid. Maybe it helps.
