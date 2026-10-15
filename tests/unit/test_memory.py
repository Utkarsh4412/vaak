"""Unit tests for SummarizingMemory, multilingual voice selection, and Hinglish prompt injection."""

from __future__ import annotations

from vaak.config import AgentConfig
from vaak.interfaces import BaseLLM
from vaak.memory import SummarizingMemory
from vaak.session import HINGLISH_INSTRUCTION, Session
from vaak.tts.kokoro import _LANG_TO_DEFAULT_VOICE


class FakeSummaryLLM(BaseLLM):
    def __init__(self, summary: str = "User discussed weather and travel.") -> None:
        self.summary = summary
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append(messages)
        return self.summary


def test_summary_after_n_turns() -> None:
    fake_llm = FakeSummaryLLM("Summary of conversation.")
    mem = SummarizingMemory(
        memory_turns=4,
        summarize_threshold=3,
        keep_recent_turns=1,
        llm=fake_llm,
    )

    mem.append_turn("hello", "hi there")
    assert mem.summary is None
    assert len(mem.history) == 2

    mem.append_turn("how are you?", "I am doing well")
    assert mem.summary is None
    assert len(mem.history) == 4

    # 3rd turn triggers summarization
    mem.append_turn("what is the weather?", "It is sunny")
    assert mem.summary == "Summary of conversation."
    assert len(fake_llm.calls) == 1
    # keep_recent_turns=1 preserves the last turn (2 messages)
    assert len(mem.history) == 2
    assert mem.history[0]["content"] == "what is the weather?"

    # Context includes summary
    context = mem.build_context("System prompt.")
    assert any(
        "[Previous conversation summary]: Summary of conversation." in m["content"]
        for m in context
    )


def test_hindi_voice_selected() -> None:
    cfg = AgentConfig()
    # TTSConfig resolves Hindi and Hinglish to hf_alpha
    assert cfg.tts.resolve_voice("hi") == "hf_alpha"
    assert cfg.tts.resolve_voice("hi-en") == "hf_alpha"
    assert _LANG_TO_DEFAULT_VOICE["hi"] == "hf_alpha"
    assert _LANG_TO_DEFAULT_VOICE["hi-en"] == "hf_alpha"


def test_hinglish_prompt_injected() -> None:
    session_en = Session(language="en")
    ctx_en = session_en.build_context("Be helpful.")
    assert HINGLISH_INSTRUCTION not in ctx_en[0]["content"]

    session_hien = Session(language="hi-en")
    ctx_hien = session_hien.build_context("Be helpful.")
    assert HINGLISH_INSTRUCTION in ctx_hien[0]["content"]


def test_summarizing_memory_fallback_without_llm() -> None:
    mem = SummarizingMemory(
        memory_turns=4,
        summarize_threshold=2,
        keep_recent_turns=1,
        llm=None,
    )
    mem.append_turn("t1 user", "t1 assistant")
    mem.append_turn("t2 user", "t2 assistant")

    assert mem.summary is not None
    assert "t1" in mem.summary
    assert len(mem.history) == 2


def test_summarize_threshold_zero_disables_summarization() -> None:
    mem = SummarizingMemory(
        memory_turns=4,
        summarize_threshold=0,
    )
    for i in range(10):
        mem.append_turn(f"user {i}", f"assistant {i}")
    assert mem.summary is None
