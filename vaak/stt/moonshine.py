"""
vaak/stt/moonshine.py — English-only STT via fastrtc.get_stt_model().

Moonshine (UsefulSensors/moonshine, loaded by fastrtc-moonshine-onnx) is
confirmed English-only. It is used as the fast path for English and Hinglish
sessions where speech is English. For Hindi/Marathi use FasterWhisperSTT.

The model is loaded once at construction and optionally warmed up by calling
warmup(). fastrtc's MoonshineSTT.stt() expects a (sample_rate, samples) tuple.
"""

from __future__ import annotations

import numpy as np
from fastrtc import get_stt_model as _get_fastrtc_stt
from loguru import logger

from vaak.interfaces import BaseSTT


class MoonshineSTT(BaseSTT):
    """English STT via fastrtc-moonshine-onnx.

    Language support: English only (verified 2026-10-07 from installed package).
    The Moonshine model family (base/tiny) shipped via fastrtc does not include
    multilingual models; those belong to a different package not used here.
    """

    def __init__(self) -> None:
        logger.info("Loading Moonshine STT model…")
        self._model = _get_fastrtc_stt()
        logger.info("Moonshine STT model loaded.")

    # ------------------------------------------------------------------
    # BaseSTT implementation
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: tuple[int, object],
        *,
        language: str | None = None,
    ) -> str:
        """Transcribe *audio* to text.

        Args:
            audio: (sample_rate, samples) tuple. samples must be a numpy array
                   of float32 values. If it is int16, it is converted.
            language: Ignored (Moonshine is English-only).

        Returns:
            Transcript string (may be empty).
        """
        sample_rate, samples = audio

        # Normalise int16 → float32 if needed (fastrtc may pass either)
        arr = np.asarray(samples)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        # Ensure 1-D (mono)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)

        text: str = self._model.stt((sample_rate, arr))
        return text.strip()

    def warmup(self) -> None:
        """Run a silent audio pass to force ONNX runtime initialisation."""
        logger.info("Warming up Moonshine STT…")
        silence = np.zeros(16000, dtype=np.float32)  # 1 s of silence at 16 kHz
        self.transcribe((16000, silence))
        logger.info("Moonshine STT warmed up.")
