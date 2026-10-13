"""
vaak/stt/faster_whisper.py — multilingual STT via faster-whisper (CTranslate2).

Supports 99 languages including Hindi, Marathi, and English.
Runs on CPU using int8 quantisation for optimal real-time latency.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from vaak.interfaces import BaseSTT


class FasterWhisperSTT(BaseSTT):
    """Multilingual STT adapter using faster-whisper with int8 CPU quantisation.

    Args:
        model_size_or_path: Whisper model size (e.g. "small", "base", "tiny") or local path.
        device: Device to run on ("cpu" or "cuda").
        compute_type: Quantisation type ("int8", "float16", "float32"). Default "int8" for CPU.
        default_language: Fallback language code (e.g. "hi", "mr", "en").
    """

    def __init__(
        self,
        model_size_or_path: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        default_language: str | None = None,
    ) -> None:
        logger.info(
            f"Loading FasterWhisper STT model {model_size_or_path!r} "
            f"(device={device}, compute_type={compute_type})…"
        )
        self._model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        self._default_language = default_language
        logger.info("FasterWhisper STT model loaded.")

    def transcribe(
        self,
        audio: tuple[int, object],
        *,
        language: str | None = None,
    ) -> str:
        """Transcribe *audio* to text.

        Args:
            audio: (sample_rate, samples) tuple. samples must be float32 values in [-1, 1].
            language: Optional language code (e.g. "hi", "en", "mr"). If omitted,
                      uses default_language or auto-detects.

        Returns:
            Transcript string.
        """
        sample_rate, samples = audio
        arr = np.asarray(samples)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if arr.ndim > 1:
            arr = arr.mean(axis=-1)

        # Resample to 16 kHz if needed
        if sample_rate != 16000 and len(arr) > 0:
            target_length = int(len(arr) * 16000 / sample_rate)
            indices = np.linspace(0, len(arr) - 1, target_length)
            arr = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)

        lang = language or self._default_language
        # Map composite language tags if needed, e.g. "hi-en" -> "hi"
        if lang and "-" in lang:
            lang = lang.split("-")[0]

        segments, _info = self._model.transcribe(
            arr,
            language=lang,
            beam_size=1,  # Greedy search for lowest real-time latency
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def warmup(self) -> None:
        """Run a silent audio pass to pre-load weights into memory."""
        logger.info("Warming up FasterWhisper STT…")
        silence = np.zeros(16000, dtype=np.float32)
        self.transcribe((16000, silence))
        logger.info("FasterWhisper STT warmed up.")
