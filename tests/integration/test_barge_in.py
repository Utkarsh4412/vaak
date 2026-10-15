"""Integration tests for barge-in and turn interruption."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator

import numpy as np

from vaak.agent import VoiceHandler
from vaak.barge_in import BargeInController
from vaak.config import AgentConfig
from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.pipeline import Pipeline
from vaak.session import Session


class DummySTT(BaseSTT):
    def transcribe(self, audio: tuple[int, object], *, language: str | None = None) -> str:
        return "user speech"


class SlowStreamingLLM(BaseLLM):
    def __init__(self, sentences: list[str], delay_s: float = 0.05) -> None:
        self.sentences = sentences
        self.delay_s = delay_s

    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return " ".join(self.sentences)

    def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> Generator[str, None, None]:
        for sentence in self.sentences:
            time.sleep(self.delay_s)
            yield sentence + " "


class SlowTTS(BaseTTS):
    def __init__(self, chunk_delay_s: float = 0.02) -> None:
        self.chunk_delay_s = chunk_delay_s
        self.synthesized_chunks: int = 0

    def synthesize(
        self, text: str, *, language: str = "en"
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        for _ in range(3):
            time.sleep(self.chunk_delay_s)
            self.synthesized_chunks += 1
            yield (24000, np.zeros(240, dtype=np.float32))


def test_barge_in_controller_lifecycle() -> None:
    controller = BargeInController()
    assert not controller.is_cancelled

    controller.request_cancel()
    assert controller.is_cancelled
    assert controller.event.is_set()

    controller.reset()
    assert not controller.is_cancelled
    assert not controller.event.is_set()

    # Test attaching to existing session
    session = Session()
    controller.attach_session(session)
    assert not controller.is_cancelled
    session.request_cancel()
    assert controller.is_cancelled


def test_voice_handler_handle_barge_in() -> None:
    cfg = AgentConfig()
    template = VoiceHandler(
        cfg, _stt=DummySTT(), _llm=SlowStreamingLLM(["Hello."]), _tts=SlowTTS()
    )
    handler = template.copy()

    assert handler.session is not None
    assert handler.barge_in is not None

    # When not speaking, barge-in is ignored (echo false-positive guard)
    assert handler.session.is_speaking is False
    assert handler.handle_barge_in() is False
    assert not handler.barge_in.is_cancelled

    # When speaking, barge-in is triggered
    handler.session.is_speaking = True
    assert handler.handle_barge_in() is True
    assert handler.barge_in.is_cancelled
    assert handler.session.cancel.is_set()


def test_barge_in_stops_streaming_pipeline_within_window() -> None:
    llm = SlowStreamingLLM(
        ["First sentence.", "Second sentence.", "Third sentence."], delay_s=0.03
    )
    tts = SlowTTS(chunk_delay_s=0.03)
    pipeline = Pipeline(
        stt=DummySTT(),
        llm=llm,
        tts=tts,
        system_prompt="sys",
        streaming=True,
    )
    session = Session()
    barge = BargeInController(session.cancel)
    audio = (16000, np.zeros(1600, dtype=np.float32))

    chunks_received: list[tuple[int, np.ndarray]] = []

    def consume() -> None:
        for chunk in pipeline.run(session, audio):
            chunks_received.append(chunk)

    t = threading.Thread(target=consume)
    t.start()

    # Wait until session begins speaking (first audio reached output)
    deadline = time.time() + 2.0
    while not session.is_speaking and time.time() < deadline:
        time.sleep(0.01)

    assert session.is_speaking

    # Simulate user barge-in mid-reply
    barge.request_cancel()
    t.join(timeout=2.0)

    # Pipeline stopped promptly without finishing all 3 sentences (3 sentences * 3 chunks = 9 chunks)
    assert len(chunks_received) < 9
    # History must NOT be committed on cancelled turn
    assert len(session.history) == 0


def test_build_stream_options() -> None:
    cfg = AgentConfig(vad_threshold=0.6, min_silence_ms=600)
    handler = VoiceHandler(cfg, _stt=DummySTT(), _llm=SlowStreamingLLM(["hi"]), _tts=SlowTTS())
    stream = handler.build_stream()
    assert stream is not None


def test_max_concurrent_sessions_busy_response() -> None:
    """Verify that exceeding max_concurrent_sessions returns busy PCM response within 200ms."""
    cfg = AgentConfig(max_concurrent_sessions=1)
    template = VoiceHandler(
        cfg,
        _stt=DummySTT(),
        _llm=SlowStreamingLLM(["Active session conversation."]),
        _tts=SlowTTS(chunk_delay_s=0.08),
    )

    handler_1 = template.copy()
    handler_2 = template.copy()
    dummy_audio = (16000, np.zeros(1600, dtype=np.float32))

    started = threading.Event()

    def run_first() -> None:
        started.set()
        list(handler_1(dummy_audio))

    t = threading.Thread(target=run_first)
    t.start()
    started.wait()
    time.sleep(0.02)  # ensure handler_1 is inside pipeline.run holding semaphore

    # Handler 2 attempts to run while Handler 1 is executing
    t0 = time.perf_counter()
    chunks_2 = list(handler_2(dummy_audio))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Must return immediately (< 200 ms) with busy tone
    assert latency_ms < 200.0
    assert len(chunks_2) == 1
    sr, pcm = chunks_2[0]
    assert sr == 24000
    assert isinstance(pcm, np.ndarray)

    t.join(timeout=3.0)
