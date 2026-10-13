"""Integration test for streaming pipeline: verifies first audio byte arrives before full LLM reply."""

from __future__ import annotations

import time
from collections.abc import Generator

import numpy as np

from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.pipeline import Pipeline
from vaak.session import Session


class DelayedStreamingLLM(BaseLLM):
    def __init__(self, tokens: list[str], delay_s: float = 0.05) -> None:
        self.tokens = tokens
        self.delay_s = delay_s
        self.all_tokens_yielded = False

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return "".join(self.tokens)

    def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> Generator[str, None, None]:
        for tok in self.tokens:
            time.sleep(self.delay_s)
            yield tok
        self.all_tokens_yielded = True


class MockSTT(BaseSTT):
    def transcribe(self, audio: tuple[int, object], *, language: str | None = None) -> str:
        return "test input"


class MockTTS(BaseTTS):
    def synthesize(
        self, text: str, *, language: str = "en"
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        # Yield one chunk per sentence
        yield (24000, np.zeros(240, dtype=np.float32))


def test_first_chunk_before_full_reply() -> None:
    # Two complete sentences with delay between tokens
    # Sentence 1: "First sentence here. "
    # Sentence 2: "Second sentence followed here."
    tokens = ["First", " sentence", " here. ", "Second", " sentence", " followed", " here."]
    llm = DelayedStreamingLLM(tokens, delay_s=0.04)
    tts = MockTTS()
    pipeline = Pipeline(
        stt=MockSTT(),
        llm=llm,
        tts=tts,
        system_prompt="system",
        streaming=True,
    )

    session = Session()
    audio = (16000, np.zeros(1600, dtype=np.float32))

    gen = pipeline.run(session, audio)
    # Receive the first PCM chunk (synthesized from sentence 1)
    first_chunk = next(gen)
    assert first_chunk is not None

    # Crucial assertion: first PCM chunk MUST arrive BEFORE all LLM tokens were yielded!
    assert not llm.all_tokens_yielded, (
        "First PCM chunk should arrive before the LLM finishes generating the full reply"
    )

    # Consume the rest of the stream
    rest = list(gen)
    assert len(rest) == 1  # Sentence 2 chunk
    assert llm.all_tokens_yielded
    assert len(session.history) == 2
