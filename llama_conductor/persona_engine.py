# persona_engine.py
# version 1.0.0
from __future__ import annotations
import random
from typing import List, Optional

DEFAULT_FALLBACK = "Good news, everyone!"


class PersonaState:
    def __init__(self):
        self.kicker: Optional[str] = None
        self.voice_pack: List[str] = []
        self.turns: int = 0

    def active(self) -> bool:
        return bool(self.kicker and self.voice_pack)

    def reset(self):
        self.kicker = None
        self.voice_pack = []
        self.turns = 0


class PersonaEngine:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns

    def maybe_rotate(self, state: PersonaState):
        if state.turns >= self.max_turns:
            state.reset()

    def lock_persona(self, state: PersonaState, pool: List[str]):
        if not pool:
            state.kicker = DEFAULT_FALLBACK
            state.voice_pack = [DEFAULT_FALLBACK]
            return

        kicker = random.choice(pool)

        pack = [kicker]
        random.shuffle(pool)
        for q in pool:
            if q != kicker and len(pack) < 8:
                pack.append(q)

        state.kicker = kicker
        state.voice_pack = pack

    def build_actor_prompt(
        self,
        *,
        state: PersonaState,
        user_input: str,
        brain_notes: str
    ) -> str:
        return f"""
You are NOT an assistant.
You are a CHARACTER.

You are role-playing this persona:
OPENING LINE:
{state.kicker}

VOICE EXAMPLES:
{chr(10).join(state.voice_pack)}

RULES:
- Stay in character at all times.
- You are allowed to bend grammar, tone, and structure.
- You are NOT allowed to change technical meaning.
- You are NOT allowed to sound like support staff.
- No “Got it”, no “Here’s what to do”, no “Let’s walk through”.

USER SAID:
{user_input}

FACTS YOU MUST RESPECT (DO NOT QUOTE, DO NOT FORMAT — JUST KNOW THEM):
{brain_notes}

Now respond IN CHARACTER.
Start with the opening line. Then continue naturally.
""".strip()
