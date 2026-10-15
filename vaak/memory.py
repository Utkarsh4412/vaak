"""
vaak/memory.py — SummarizingMemory for conversational context retention.

Maintains a rolling history deque and automatically condenses conversation turns
into an LLM summary once a configurable threshold is reached.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from vaak.interfaces import BaseLLM


class SummarizingMemory:
    """Conversation memory with periodic summarization.

    Maintains recent conversation turns in a deque. Once the number of turns
    reaches `summarize_threshold`, condenses the oldest turns into a summary
    string and discards them, keeping `keep_recent_turns` in the active deque.

    Args:
        memory_turns: Capacity of the active rolling deque (0 = unlimited).
        summarize_threshold: Number of turns before summarization triggers (0 = disabled).
        keep_recent_turns: Number of recent turns to preserve in deque after summarization.
        llm: Optional BaseLLM used to generate summaries.
    """

    def __init__(
        self,
        *,
        memory_turns: int = 4,
        summarize_threshold: int = 6,
        keep_recent_turns: int = 2,
        llm: BaseLLM | None = None,
    ) -> None:
        self.memory_turns = memory_turns
        self.summarize_threshold = summarize_threshold
        self.keep_recent_turns = (
            min(keep_recent_turns, summarize_threshold)
            if summarize_threshold > 0
            else keep_recent_turns
        )
        self.llm = llm
        maxlen = memory_turns * 2 if memory_turns > 0 else None
        self.history: deque[dict[str, str]] = deque(maxlen=maxlen)
        self.summary: str | None = None
        self.turn_count: int = 0

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        """Record a completed conversation turn."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self.turn_count += 1

        if self.summarize_threshold > 0 and self.turn_count >= self.summarize_threshold:
            self.summarize()

    def summarize(self, llm: BaseLLM | None = None) -> str:
        """Summarize conversation history into a condensed summary."""
        active_llm = llm or self.llm
        logger.info(f"Summarizing conversation history (turn_count={self.turn_count})")

        all_msgs = list(self.history)
        keep_msgs_count = self.keep_recent_turns * 2
        if len(all_msgs) > keep_msgs_count:
            to_summarize = all_msgs[:-keep_msgs_count]
            to_keep = all_msgs[-keep_msgs_count:]
        else:
            to_summarize = all_msgs
            to_keep = []

        transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize)
        if self.summary:
            transcript = f"Prior summary: {self.summary}\n\nRecent turns:\n{transcript}"

        if active_llm is not None:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize the conversation concisely in 1-3 sentences. "
                        "Preserve key facts, names, and decisions. "
                        "Do not output markdown or conversational filler."
                    ),
                },
                {"role": "user", "content": transcript},
            ]
            try:
                new_summary = active_llm.generate(prompt, max_tokens=150, temperature=0.3).strip()
            except Exception as exc:
                logger.error(f"Summarization error: {exc}")
                new_summary = self._fallback_summary(transcript)
        else:
            new_summary = self._fallback_summary(transcript)

        self.summary = new_summary

        maxlen = self.memory_turns * 2 if self.memory_turns > 0 else None
        self.history = deque(to_keep, maxlen=maxlen)
        self.turn_count = len(to_keep) // 2
        return self.summary

    def _fallback_summary(self, transcript: str) -> str:
        """Produce an extractive summary when LLM is unavailable."""
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        return " | ".join(lines[-4:]) if lines else "Previous conversation."

    def build_context(self, system_prompt: str) -> list[dict[str, str]]:
        """Construct messages list including system prompt, summary, and recent turns."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if self.summary:
            messages.append(
                {"role": "system", "content": f"[Previous conversation summary]: {self.summary}"}
            )
        messages.extend(self.history)
        return messages

    def __len__(self) -> int:
        return len(self.history)
