"""Integration tests for Hindi and Hinglish end-to-end pipeline flow."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from vaak.config import AgentConfig
from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.pipeline import Pipeline
from vaak.session import HINGLISH_INSTRUCTION, Session


class MockFasterWhisperSTT(BaseSTT):
    def __init__(self, transcript: str = "नमस्ते, आप कैसे हैं?") -> None:
        self.transcript = transcript
        self.recorded_lang: str | None = None

    def transcribe(self, audio: tuple[int, object], *, language: str | None = None) -> str:
        self.recorded_lang = language
        return self.transcript


class MockMultilingualLLM(BaseLLM):
    def __init__(self, reply: str = "मैं ठीक हूँ, धन्यवाद।") -> None:
        self.reply = reply
        self.last_messages: list[dict[str, str]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.last_messages = messages
        return self.reply

    def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> Generator[str, None, None]:
        self.last_messages = messages
        for word in self.reply.split():
            yield word + " "


class MockMultilingualTTS(BaseTTS):
    def __init__(self) -> None:
        self.synthesized: list[tuple[str, str]] = []

    def synthesize(
        self, text: str, *, language: str = "en"
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        self.synthesized.append((text, language))
        yield (24000, np.zeros(240, dtype=np.float32))


def test_hindi_pipeline_e2e() -> None:
    """Verify complete turn flow in Hindi with language routing."""
    stt = MockFasterWhisperSTT(transcript="नमस्ते, आप कैसे हैं?")
    llm = MockMultilingualLLM(reply="मैं बढ़िया हूँ। आप बताइए।")
    tts = MockMultilingualTTS()

    pipeline = Pipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        system_prompt="You are a helpful voice assistant.",
        streaming=True,
    )

    session = Session(language="hi", memory_turns=4)
    audio = (16000, np.zeros(1600, dtype=np.float32))

    chunks = list(pipeline.run(session, audio))
    assert len(chunks) > 0
    assert stt.recorded_lang == "hi"
    assert tts.synthesized[0][1] == "hi"
    assert len(session.history) == 2
    assert session.history[0]["content"] == "नमस्ते, आप कैसे हैं?"


def test_hinglish_pipeline_e2e_prompt_and_voice() -> None:
    """Verify Hinglish mode injects instruction and uses Hindi TTS voice."""
    cfg = AgentConfig(language="hi-en")
    assert cfg.tts.resolve_voice(cfg.language) == "hf_alpha"

    stt = MockFasterWhisperSTT(transcript="Kya scene hai bhai?")
    llm = MockMultilingualLLM(reply="Sab badhiya hai bhai, tum sunao!")
    tts = MockMultilingualTTS()

    pipeline = Pipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        system_prompt=cfg.system_prompt,
        streaming=True,
    )

    session = Session(language=cfg.language, memory_turns=4)
    audio = (16000, np.zeros(1600, dtype=np.float32))

    chunks = list(pipeline.run(session, audio))
    assert len(chunks) > 0
    # LLM messages must include Hinglish guidance
    system_msg = llm.last_messages[0]["content"]
    assert HINGLISH_INSTRUCTION in system_msg
    assert tts.synthesized[0][1] == "hi-en"
