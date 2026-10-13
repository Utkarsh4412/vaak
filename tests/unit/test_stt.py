"""Unit tests for STT adapters: FasterWhisperSTT and LangDetectRouter."""

from unittest.mock import MagicMock, patch

import numpy as np

from vaak.stt.faster_whisper import FasterWhisperSTT
from vaak.stt.lang_detect import LangDetectRouter


class DummySTT:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple] = []
        self.warmed_up = False

    def transcribe(self, audio, *, language=None):
        self.calls.append((audio, language))
        return f"{self.name}:{language}"

    def warmup(self):
        self.warmed_up = True


class TestLangDetectRouter:
    def test_routes_english_to_moonshine(self):
        moonshine = DummySTT("moonshine")
        faster_whisper = DummySTT("faster_whisper")
        router = LangDetectRouter(moonshine, faster_whisper)  # type: ignore[arg-type]

        audio = (16000, np.zeros(1600, dtype=np.float32))
        res = router.transcribe(audio, language="en")
        assert res.startswith("moonshine")
        assert len(moonshine.calls) == 1
        assert len(faster_whisper.calls) == 0

    def test_routes_english_variants_to_moonshine(self):
        moonshine = DummySTT("moonshine")
        faster_whisper = DummySTT("faster_whisper")
        router = LangDetectRouter(moonshine, faster_whisper)  # type: ignore[arg-type]

        audio = (16000, np.zeros(1600, dtype=np.float32))
        router.transcribe(audio, language="en-US")
        assert len(moonshine.calls) == 1

        router.transcribe(audio, language="en-gb")
        assert len(moonshine.calls) == 2
        assert len(faster_whisper.calls) == 0

    def test_routes_hindi_to_faster_whisper(self):
        moonshine = DummySTT("moonshine")
        faster_whisper = DummySTT("faster_whisper")
        router = LangDetectRouter(moonshine, faster_whisper)  # type: ignore[arg-type]

        audio = (16000, np.zeros(1600, dtype=np.float32))
        res = router.transcribe(audio, language="hi")
        assert res.startswith("faster_whisper")
        assert len(faster_whisper.calls) == 1
        assert len(moonshine.calls) == 0

    def test_routes_marathi_to_faster_whisper(self):
        moonshine = DummySTT("moonshine")
        faster_whisper = DummySTT("faster_whisper")
        router = LangDetectRouter(moonshine, faster_whisper)  # type: ignore[arg-type]

        audio = (16000, np.zeros(1600, dtype=np.float32))
        router.transcribe(audio, language="mr")
        assert len(faster_whisper.calls) == 1

    def test_warmup_calls_both(self):
        moonshine = DummySTT("moonshine")
        faster_whisper = DummySTT("faster_whisper")
        router = LangDetectRouter(moonshine, faster_whisper)  # type: ignore[arg-type]

        router.warmup()
        assert moonshine.warmed_up
        assert faster_whisper.warmed_up


class TestFasterWhisperSTT:
    @patch("vaak.stt.faster_whisper.WhisperModel")
    def test_transcribe_normalises_audio(self, mock_model_cls):
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = " नमस्ते"
        mock_model.transcribe.return_value = ([seg], None)
        mock_model_cls.return_value = mock_model

        stt = FasterWhisperSTT(default_language="hi")
        # Pass int16 stereo audio
        int16_audio = (16000, np.zeros((1600, 2), dtype=np.int16))
        text = stt.transcribe(int16_audio)

        assert text == "नमस्ते"
        assert mock_model.transcribe.called
        call_args, call_kwargs = mock_model.transcribe.call_args
        passed_audio = call_args[0]
        assert passed_audio.dtype == np.float32
        assert passed_audio.ndim == 1
        assert call_kwargs["language"] == "hi"

    @patch("vaak.stt.faster_whisper.WhisperModel")
    def test_transcribe_strips_subtags(self, mock_model_cls):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        mock_model_cls.return_value = mock_model

        stt = FasterWhisperSTT()
        audio = (16000, np.zeros(1600, dtype=np.float32))
        stt.transcribe(audio, language="hi-en")

        call_args, call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs["language"] == "hi"
