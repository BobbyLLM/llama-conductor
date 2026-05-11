Well, if you clicked on this, I suppose you really *want* to know how this thing came to be. 

Let's start with the obvious:

I am not a coder. I am not a CS professional. 

This is not false modesty. 

I taught myself this much over the course of a year and the reactivation of some very old skills (30 years hence). When I decided to do this, it wasn’t from any school of thought or design principle. I don’t know how CS professionals build things. The last time I looked at an IDE was TurboPascal. (Yes, I’m that many years old. I think it probably shows, what with the >> ?? !! ## all over the place). 

What I do know is - what was the problem I was trying to solve? Let me try to pseudo-code my thinking process for you a bit :) 

IF the following are true;

- I have ASD. If you tell me a thing, I assume your telling me a thing. I don’t assume you’re telling me one thing but mean something else.
- A LLM could “lie” to me, and I would believe it, because I’m not a subject matter expert on the thing (usually). Also see point 1.
- I want to believe it, because why would a tool say X but mean Y? See point 1.
- A LLM could lie to me in a way that is undetectable, because I have no idea what it’s reasoning over, how it’s reasoning over it. It’s  
  literally a black box. I ask a Question—>MAGIC WIRES---->Answer.

AND

    “The first principle is that you must not fool yourself and you are the easiest person to fool”

THEN

**STOP.**

I’m fucked. This problem is unsolvable.

IF 
LLMs are inherently hallucinatory within bounds (AFAIK, the current iterations all are)

AND 

IF there’s even a 1% chance that it will fuck me over (it has)

THEN for my own sanity, I have to assume that such an outcome is a mathematical certainty. I cannot operate in this environment.

STOP. Fuck. Ok.

PROBLEM: How do I interact with a system that is dangerously mimetic and dangerously opaque? What levers can I pull? Or do I just need to walk away?

- Unchangeable. Eat shit, BobbyLLM. Ok.
- I can do something about that…or at least, I can verify what’s being said, if the process isn’t too mentally taxing. Hmm. How?
- Fine, I want to believe it…but, do I have to believe it blindly? How about a defensive position - “Trust but verify”?. Hmm. How?
- Why does it HAVE to be opaque? If I build it, why do I have to hide the workings? I want to know how it works, breaks, and what it can do.

Everything else flowed from those ideas. I actually came up with a design document (list of invariants). It’s about 1200 words or so, and unashamedly inspired by Asimov :)

#################

System Invariants

0. What an invariant is (binding)

An invariant is a rule that:

    Must always hold, regardless of refactor, feature, or model choice
    Must not be violated temporarily, even internally. The system must not fuck me over silently.
    Overrides convenience, performance, and cleverness.

If a feature conflicts with an invariant, the feature is wrong. Do not add.

1. Global system invariant rules:

1.1 Determinism over cleverness

    Given the same inputs and state, the system must behave predictably.

    No component may:
        infer hidden intent,
        rely on emergent LLM behavior
        or silently adapt across turns without explicit user action.

1.2 Explicit beats implicit

    Any influence on an answer must be inspectable and user-controllable.

    This includes:
        memory,
        retrieval,
        reasoning mode,
        style transformation.

If something affects the output, the user must be able to:

    enable it,
    disable it,
    and see that it ran.

Assume system is going to lie. Make its lies loud and obvious.

##############

On and on it drones LOL. I spent a good 4-5 months just revising a tighter and tighter series of constraints that a would make a Shibari master blush, such that the end product 1) would be less likely to break 2) if it did break, it do in a loud, obvious way.

What you see on the repo is the best I could do, with what I had. 

I hope it’s something and I didn’t GIGO myself into stupid. But no promises :)

For more (including me answering some pointed question about this router) please view  https://lemmy.world/post/41992574  

