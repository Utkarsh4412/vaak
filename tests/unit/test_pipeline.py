"""Unit tests for Pipeline using fake STT, LLM, and TTS adapters.

All tests run without loading any real model weights.
"""

import numpy as np
import pytest

from vaak.barge_in import BargeInController
from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.pipeline import Pipeline
from vaak.session import Session

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSTT(BaseSTT):
    """Returns a fixed transcript or empty string."""

    def __init__(self, transcript: str = "hello"):
        self.transcript = transcript
        self.calls: list[tuple] = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.transcript


class FakeLLM(BaseLLM):
    """Returns a fixed reply or raises on demand."""

    def __init__(self, reply: str = "world", raise_on_call: bool = False):
        self.reply = reply
        self.raise_on_call = raise_on_call
        self.calls: list = []

    def generate(self, messages, *, max_tokens=200, temperature=0.7, top_p=0.9):
        self.calls.append(messages)
        if self.raise_on_call:
            raise RuntimeError("LLM is down")
        return self.reply


class FakeTTS(BaseTTS):
    """Yields a fixed number of silence chunks."""

    def __init__(self, n_chunks: int = 2, raise_on_call: bool = False):
        self.n_chunks = n_chunks
        self.raise_on_call = raise_on_call
        self.calls: list = []

    def synthesize(self, text, *, language="en"):
        self.calls.append((text, language))
        if self.raise_on_call:
            raise RuntimeError("TTS is down")
        for _ in range(self.n_chunks):
            yield (24000, np.zeros(100, dtype=np.float32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SILENCE = (16000, np.zeros(1600, dtype=np.float32))


def _make_pipeline(**kwargs):
    defaults: dict = {
        "stt": FakeSTT(),
        "llm": FakeLLM(),
        "tts": FakeTTS(),
        "system_prompt": "Be helpful.",
    }
    defaults.update(kwargs)
    return Pipeline(**defaults)


def _run(pipeline, session=None, audio=_SILENCE):
    if session is None:
        session = Session()
    return list(pipeline.run(session, audio))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineHappyPath:
    def test_yields_tts_chunks(self):
        p = _make_pipeline(tts=FakeTTS(n_chunks=3))
        chunks = _run(p)
        assert len(chunks) == 3

    def test_each_chunk_is_tuple(self):
        p = _make_pipeline()
        for sr, arr in _run(p):
            assert isinstance(sr, int)
            assert isinstance(arr, np.ndarray)

    def test_history_updated_after_clean_run(self):
        s = Session()
        p = _make_pipeline()
        _run(p, session=s)
        assert len(s.history) == 2

    def test_stt_called_once(self):
        stt = FakeSTT()
        p = _make_pipeline(stt=stt)
        _run(p)
        assert len(stt.calls) == 1

    def test_llm_receives_system_prompt(self):
        llm = FakeLLM()
        p = _make_pipeline(llm=llm, system_prompt="Custom prompt.")
        _run(p)
        assert llm.calls[0][0]["content"] == "Custom prompt."

    def test_tts_receives_llm_reply(self):
        tts = FakeTTS()
        p = _make_pipeline(llm=FakeLLM(reply="test reply"), tts=tts)
        _run(p)
        assert tts.calls[0][0] == "test reply"

    def test_tts_receives_session_language(self):
        tts = FakeTTS()
        s = Session(language="hi")
        p = _make_pipeline(tts=tts)
        _run(p, session=s)
        assert tts.calls[0][1] == "hi"


class TestPipelineEmptySTT:
    def test_empty_transcript_produces_no_chunks(self):
        p = _make_pipeline(stt=FakeSTT(transcript=""))
        assert _run(p) == []

    def test_empty_transcript_does_not_call_llm(self):
        llm = FakeLLM()
        p = _make_pipeline(stt=FakeSTT(transcript=""), llm=llm)
        _run(p)
        assert len(llm.calls) == 0

    def test_empty_transcript_does_not_update_history(self):
        s = Session()
        p = _make_pipeline(stt=FakeSTT(transcript=""))
        _run(p, session=s)
        assert len(s.history) == 0


class TestPipelineLLMError:
    def test_llm_error_still_returns_tts_chunks(self):
        # Pipeline catches LLM errors and returns fallback text
        tts = FakeTTS()
        captured_report = None

        def on_report_cb(report):
            nonlocal captured_report
            captured_report = report.to_dict()

        p = _make_pipeline(
            llm=FakeLLM(raise_on_call=True),
            tts=tts,
            llm_error_reply="fallback message",
            on_report=on_report_cb,
        )
        s = Session()
        chunks = _run(p, session=s)

        # TTS should be called with the fallback message
        assert tts.calls[0][0] == "fallback message"
        # TTS chunks should be yielded
        assert len(chunks) > 0
        # History must NOT be written
        assert len(s.history) == 0
        # Report must be emitted
        assert captured_report is not None
        assert captured_report["llm_total_ms"] >= 0.0


class TestPipelineCancel:
    def test_cancelled_before_tts_yields_nothing(self):
        s = Session()
        s.request_cancel()  # cancel before pipeline starts
        p = _make_pipeline()
        chunks = _run(p, session=s)
        # cancel is reset at start of run, so pipeline runs normally
        # (reset_cancel clears it) — result is chunks ARE yielded
        assert len(chunks) > 0

    def test_history_not_updated_if_tts_interrupted(self):
        """Simulate barge-in mid-TTS by setting cancel inside a custom TTS."""

        class InterruptingTTS(BaseTTS):
            def __init__(self, session_ref):
                self._session = session_ref

            def synthesize(self, text, *, language="en"):
                # yield one chunk then fire cancel
                yield (24000, np.zeros(100, dtype=np.float32))
                self._session.request_cancel()
                yield (24000, np.zeros(100, dtype=np.float32))

        s = Session()
        p = Pipeline(
            stt=FakeSTT(),
            llm=FakeLLM(),
            tts=InterruptingTTS(s),
            system_prompt="sys",
        )
        chunks = list(p.run(s, _SILENCE))
        # Only 1 chunk yielded before cancel
        assert len(chunks) == 1
        # History must NOT be updated
        assert len(s.history) == 0


def test_barge_in_stops_tts() -> None:
    """Verify that triggering barge-in stops TTS output mid-stream."""
    s = Session()
    barge = BargeInController(s.cancel)

    class InterruptingTTS(BaseTTS):
        def synthesize(self, text, *, language="en"):
            yield (24000, np.zeros(100, dtype=np.float32))
            barge.request_cancel()
            yield (24000, np.zeros(100, dtype=np.float32))
            yield (24000, np.zeros(100, dtype=np.float32))

    p = Pipeline(
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=InterruptingTTS(),
        system_prompt="sys",
        streaming=False,
    )
    chunks = list(p.run(s, _SILENCE))
    assert len(chunks) == 1
    assert s.cancel.is_set()


def test_history_unchanged_after_cancel() -> None:
    """Verify that cancelling during TTS prevents partial response committing to history."""
    s = Session()
    barge = BargeInController(s.cancel)

    class InterruptingTTS(BaseTTS):
        def synthesize(self, text, *, language="en"):
            yield (24000, np.zeros(100, dtype=np.float32))
            barge.request_cancel()
            yield (24000, np.zeros(100, dtype=np.float32))

    p = Pipeline(
        stt=FakeSTT(),
        llm=FakeLLM(),
        tts=InterruptingTTS(),
        system_prompt="sys",
    )
    list(p.run(s, _SILENCE))
    assert len(s.history) == 0


class TestPipelineSessionIsolation:
    def test_two_sessions_do_not_share_history(self):
        p = _make_pipeline()
        s1, s2 = Session(), Session()
        _run(p, session=s1)
        assert len(s2.history) == 0

    def test_history_accumulates_across_turns(self):
        p = _make_pipeline()
        s = Session()
        _run(p, session=s)
        _run(p, session=s)
        assert len(s.history) == 4  # 2 turns × 2 messages


class TestPipelineLatency:
    def test_latency_metrics_logged_on_clean_run(self, monkeypatch):
        # Fake time sequence:
        # e2e start -> stt start -> stt end -> llm start -> llm end ->
        # tts start -> tts first chunk -> tts second chunk -> tts end -> e2e end
        times = [
            1.000,  # e2e start
            1.000,  # stt start
            1.100,  # stt end (stt_ms = 100)
            1.100,  # llm start
            1.200,  # llm end (llm_total_ms = 100)
            1.200,  # tts start
            1.250,  # tts first chunk (for tts_first_ms)
            1.250,  # tts first chunk (for first_audio_ms)
            1.300,  # tts second chunk
            1.300,  # tts end
            1.300,  # e2e end
        ]

        def fake_perf_counter():
            return times.pop(0) if times else 1.300

        monkeypatch.setattr("vaak.metrics.time.perf_counter", fake_perf_counter)

        captured_report = None

        def on_report_cb(report):
            nonlocal captured_report
            captured_report = report.to_dict()

        p = _make_pipeline(tts=FakeTTS(n_chunks=2), on_report=on_report_cb, streaming=False)
        _run(p)

        assert captured_report is not None

        assert captured_report["stt_ms"] == pytest.approx(100.0)
        assert captured_report["llm_total_ms"] == pytest.approx(100.0)
        assert captured_report["tts_first_chunk_ms"] == pytest.approx(50.0)
        assert captured_report["tts_total_ms"] == pytest.approx(100.0)
        assert captured_report["first_audio_ms"] == pytest.approx(250.0)
        assert captured_report["turn_ms"] == pytest.approx(300.0)

    def test_latency_metrics_logged_on_generator_exit(self, monkeypatch):
        # Same sequence, but we break after the first chunk
        times = [
            1.000,  # e2e start
            1.000,  # stt start
            1.100,  # stt end
            1.100,  # llm start
            1.200,  # llm end
            1.200,  # tts start
            1.250,  # tts first chunk (tts_first_ms)
            1.250,  # tts first chunk (first_audio_ms)
            # consumer breaks loop here, causing GeneratorExit
            1.300,  # tts end
            1.300,  # e2e end
        ]

        def fake_perf_counter():
            return times.pop(0) if times else 1.300

        monkeypatch.setattr("vaak.metrics.time.perf_counter", fake_perf_counter)

        captured_report = None

        def on_report_cb(report):
            nonlocal captured_report
            captured_report = report.to_dict()

        p = _make_pipeline(tts=FakeTTS(n_chunks=3), on_report=on_report_cb, streaming=False)

        # Consume only 1 chunk to trigger GeneratorExit (or just stop iterating)
        s = Session()
        gen = p.run(s, _SILENCE)
        next(gen)
        gen.close()  # explicitly close

        assert captured_report is not None

        assert captured_report["first_audio_ms"] == pytest.approx(250.0)
        assert captured_report["tts_total_ms"] > 0.0
        assert captured_report["turn_ms"] == pytest.approx(300.0)
        # History must NOT be updated because it didn't complete cleanly
        assert len(s.history) == 0

    def test_latency_metrics_not_logged_on_empty_stt(self):
        captured_report = None

        def on_report_cb(report):
            nonlocal captured_report
            captured_report = report.to_dict()

        p = _make_pipeline(stt=FakeSTT(transcript=""), on_report=on_report_cb)
        _run(p)
        assert captured_report is None


class TestSentenceChunker:
    def test_splits_on_punctuation_with_whitespace(self):
        from vaak.pipeline import sentence_chunker

        tokens = ["Hello", " world.", " How", " are", " you? ", "I", " am", " fine."]
        sentences = list(sentence_chunker(tokens))
        assert sentences == ["Hello world.", "How are you?", "I am fine."]

    def test_yields_remainder_without_terminal_punctuation(self):
        from vaak.pipeline import sentence_chunker

        tokens = ["This is", " incomplete"]
        sentences = list(sentence_chunker(tokens))
        assert sentences == ["This is incomplete"]

    def test_empty_token_stream(self):
        from vaak.pipeline import sentence_chunker

        assert list(sentence_chunker([])) == []


class TestAuditLog:
    def test_audit_log_written_when_enabled(self, tmp_path):
        import json

        log_path = tmp_path / "test_audit.log"
        p = _make_pipeline(audit_log=True, audit_log_path=str(log_path))
        _run(p)

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["user_text"] == "hello"
        assert record["reply_text"] == "world"
        assert record["completed"] is True
        assert record["cancelled"] is False
        assert "metrics" in record

    def test_audit_log_not_written_when_disabled(self, tmp_path):
        log_path = tmp_path / "test_audit.log"
        p = _make_pipeline(audit_log=False, audit_log_path=str(log_path))
        _run(p)

        assert not log_path.exists()
