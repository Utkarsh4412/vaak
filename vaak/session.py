"""
vaak/session.py — per-connection Session with unique ID, language, history deque,
and cancel event.

Design note: Session is created inside the handler's copy() method (called by
FastRTC once per WebRTC connection), so each caller gets its own isolated
history. This fixes the shared-memory bug where a single history
deque was shared across all connections.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING
from uuid import uuid4

from vaak.memory import SummarizingMemory

if TYPE_CHECKING:
    from vaak.interfaces import BaseLLM

HINGLISH_INSTRUCTION: str = (
    "Please reply in Hinglish (a natural blend of Hindi and English written in Latin script). "
    "Keep replies conversational, concise, and easy to speak aloud."
)


class Session:
    """Encapsulates all per-connection state.

    Attributes:
        id: Unique UUID4 string identifying this connection.
        language: BCP-47-style language code for this session
                  ("en", "hi", "hi-en", "es", "fr", "en-gb").
        memory: SummarizingMemory instance managing history and periodic condensation.
        history: Rolling deque of {"role": ..., "content": ...} dicts.
        cancel: threading.Event set when a barge-in is requested.
                Cleared automatically at the start of each pipeline turn.
        is_speaking: True while the agent is synthesising speech.
                     Gate for echo / false-positive barge-in detection.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        memory_turns: int = 4,
        summarize_threshold: int = 6,
        memory: SummarizingMemory | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.id: str = str(uuid4())
        self.language: str = language
        self.memory: SummarizingMemory = (
            memory
            if memory is not None
            else SummarizingMemory(
                memory_turns=memory_turns,
                summarize_threshold=summarize_threshold,
                llm=llm,
            )
        )
        self.cancel: threading.Event = threading.Event()
        self.is_speaking: bool = False

    @property
    def history(self) -> deque[dict[str, str]]:
        """Active rolling history deque from SummarizingMemory."""
        return self.memory.history

    @history.setter
    def history(self, val: deque[dict[str, str]]) -> None:
        self.memory.history = val

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset_cancel(self) -> None:
        """Clear the cancel event at the start of a new pipeline turn."""
        self.cancel.clear()

    def request_cancel(self) -> None:
        """Request cancellation of the current pipeline turn (barge-in)."""
        self.cancel.set()

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a completed turn to memory."""
        self.memory.append_turn(user_text, assistant_text)

    def build_context(self, system_prompt: str) -> list[dict[str, str]]:
        """Return the full message list ready to pass to BaseLLM.generate()."""
        prompt = system_prompt
        if self.language == "hi-en":
            prompt = f"{prompt}\n\n{HINGLISH_INSTRUCTION}"
        return self.memory.build_context(prompt)

    def __repr__(self) -> str:
        return (
            f"Session(id={self.id!r}, lang={self.language!r}, "
            f"history_len={len(self.history)}, "
            f"cancelled={self.cancel.is_set()})"
        )
