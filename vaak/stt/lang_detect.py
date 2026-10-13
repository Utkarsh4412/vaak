"""
vaak/stt/lang_detect.py — routes STT requests based on session language.

Uses MoonshineSTT for the English fast path and FasterWhisperSTT for multilingual.
"""

from __future__ import annotations

from vaak.interfaces import BaseSTT


class LangDetectRouter(BaseSTT):
    """Routes audio transcription to the optimal STT backend based on session language.

    Uses MoonshineSTT for English fast path and FasterWhisperSTT for Hindi,
    Marathi, and other multilingual requests.
    """

    def __init__(
        self,
        moonshine: BaseSTT,
        faster_whisper: BaseSTT,
    ) -> None:
        self._moonshine = moonshine
        self._faster_whisper = faster_whisper

    def transcribe(
        self,
        audio: tuple[int, object],
        *,
        language: str | None = None,
    ) -> str:
        lang = (language or "en").lower()
        if lang in ("en", "en-us", "en-gb"):
            try:
                return self._moonshine.transcribe(audio, language=lang)
            except TypeError:
                return self._moonshine.transcribe(audio)
        return self._faster_whisper.transcribe(audio, language=lang)

    def warmup(self) -> None:
        self._moonshine.warmup()
        self._faster_whisper.warmup()
