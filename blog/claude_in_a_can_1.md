**I'd like to show you my...ass**

Marvel movies.

When they're good, they're great (Winter Soldier). When they're bad, they're...The Eternals.

From time to time, I find myself drawn to one of the "less good ones" that I personally quite enjoy (IronMan 2). Apart from the near incessant torturing of my wife with Ivan Vanko impressions ("Dats not my Boid") I find myself reflecting on the cameo scenes with Howard Stark (John Slattery) that make the heart of the movie.

In one pivotal scene, Howard goes from offering to "personally show you his ass" ([YouTube clip](https://www.youtube.com/watch?v=Rxb_iemjo4Y)) to lamenting the limits of his technology while expressing his love for Tony. He looks at the camera, dead on, and says -

"I am limited by the technology of my time".

Which in a torturous, saccharine way, brings us to Claude in a Can.

**I've got sunshine, in a bag**

Like you, I've used all manner of cloud based, state of the art (SOTA) LLM.

There are good ones, and bad ones (*cough* You're absolutely right!), but like many I always seem to come back to Claude.

Not because Anthropic are particularly virtuous but rather because Claude just ... doesn't suck.

It will frequently tell me to fuck off to bed. It will REFUSE to do things it knows I can (or should) do for myself. As AI go, it does a passable impression of something that has an opinion and a spine.

Yet...Claude is a rich man's AI. Undoubtedly. Daily time limits, weekly time limits, "buy extra usage now for just $9.99!". What's worse, from a privacy and "don't be evil" point of view, I find myself on shaky moral ground. Better then to Tony Stank (not a typo) myself something from a box of scraps.

So, that got me to thinking, what is it about Claude that I so enjoy....and how can I steal it?


**I am limited by the technology of my time**

The normal, sane, solution to this problem would be to drop $5-$10K, purchase a beastly rig (that no doubt chews through 2Kw/day of power), slap some LoRA set dressing on it to give it the Anthropic accent and call it a day.

But, I'm not normal (clearly). Instead, I decided to do something...different.

I decided to think.

What makes Claude...Claude?

For the average (non coder), I'd argue the entire experience comes down to behaviour. The tone, the refusals, timing, the call backs. The seemingly magical ability to sit with something instead of pivoting and then help create a novel connection that sparks a new idea. That is a thinking tool.

And that can be stolen.

At least, the nucleus of it - because like Cesar Millan showed us, behaviour is trainable.

The question then becomes -

* How does one approximate Claude-like behaviour?
* Without LoRA?
* On hardware that Anthropic wouldn't use to play Tetris on?

**Kaioken x 5**

I run on 4GB of VRAM, with 640 CUDA cores, on a 1L shoebox. Do not weep for me, for I am already dead.

However, being dead gives me an advantage. It cuts off a lot of "just add more grunt" options. So, I find myself returning to the tried and tested decision tree.

For the uninitiated, a decision tree is exactly what it sounds like - a flowchart that makes decisions based on rules. No neural network. No training data. Hell, no GPU required. Millisecond fast.

Just if this --> then that.

It's the oldest trick in the book. It's also, it turns out, exactly the right tool for this problem.

Because Claude-like behaviour isn't magic. It's a sequence of decisions. What is this person doing right now? What do they need from this response? What would make this worse? A decision tree can answer all three before the model touches the keyboard.

So that's what I did.

Before the model sees anything, something else runs first.

Every message you send gets classified. Not for content - for *intent*. What are you actually doing right now? Are you working? Venting? Taking the piss? Wrapping something painful in a joke because that's how you handle it?

That's KAIOKEN. A fast turn classifier that reads the human before the model does.

The taxonomy is simple. Macro label: `working`, `casual`, or `personal`. Subsignal: `playful`, `friction`, `distress_hint`, `vulnerable_under_humour`, or nothing. It's a work in progress, but as MVP, I started here.

"I'm 47, old, fat and broken. Why did I take up coding?" hits different than "prove you're not sentient, Aristotle." They should get different responses from the LLM. Not because the model is smart enough to figure that out reliably - it isn't, not at 4B - but because the harness already knows, and tells it what to do.

KAIOKEN doesn't touch the model. It touches the instructions the model receives. Classification fires, label maps to behavioural injection, 4B gets concrete direction before it generates a single token. All at millisecond speeds (yay, no latency)

And yes: the name is from Dragon Ball Z. Kaioken multiplies a fighter's power beyond their normal limits. Seemed accurate.

It's the steering wheel. The next post will describe the engine. Assuming the engine doesn't embarrass me first before I finish building it from a box of scraps.


Until then, Ivan Vanko says hello.

[![Ivan Vanko says hello](https://i.imgur.com/R7RGy3D.jpeg)](https://i.imgur.com/R7RGy3D.jpeg)
