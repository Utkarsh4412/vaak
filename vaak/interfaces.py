"""
vaak/interfaces.py — abstract base classes for STT, LLM, and TTS components.

All adapters must implement these interfaces. The Pipeline only depends on
these abstractions, never on concrete implementations.
"""

from __future__ import annotations

import abc
from collections.abc import Generator


class BaseSTT(abc.ABC):
    """Speech-to-text interface.

    Implementors: MoonshineSTT (English fast path), FasterWhisperSTT (multilingual).
    """

    @abc.abstractmethod
    def transcribe(self, audio: tuple[int, object], *, language: str | None = None) -> str:
        """Convert an audio segment to a transcript string.

        Args:
            audio: (sample_rate, samples) tuple where samples is a numpy array
                   of shape (N,) with dtype float32, values in [-1.0, 1.0].
            language: Optional language code hint (e.g. "hi", "mr", "en").

        Returns:
            Transcript string. May be empty if no speech detected.
        """

    def warmup(self) -> None:  # noqa: B027
        """Optional: pre-load model weights. Called once at startup."""


class BaseLLM(abc.ABC):
    """Language model interface.

    Implementors: OllamaLLM.
    """

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a complete reply given a message list.

        Args:
            messages: OpenAI-style list of {"role": ..., "content": ...} dicts.
            max_tokens: Upper bound on tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Complete reply string.
        """

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Yield reply tokens one at a time as they arrive.

        Default implementation calls generate() and yields the full string
        in one shot. Subclasses should override for true streaming.

        Args:
            messages: OpenAI-style message list.
            max_tokens: Upper bound on tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Yields:
            Token strings (fragments of the complete reply).
        """
        yield self.generate(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def warmup(self) -> None:  # noqa: B027
        """Optional: ping the backend to load the model. Called once at startup."""


class BaseTTS(abc.ABC):
    """Text-to-speech interface.

    Implementors: KokoroTTS.
    """

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        language: str = "en",
    ) -> Generator[tuple[int, object], None, None]:
        """Synthesise *text* and yield PCM chunks.

        Each yielded value is a (sample_rate, samples) tuple where samples
        is a numpy array of shape (N,) with dtype float32.

        Args:
            text: Sentence or paragraph to synthesise.
            language: BCP-47-style language code ("en", "hi", "es", "fr", …).

        Yields:
            (sample_rate, samples) pairs; may be multiple chunks per call.
        """

    def warmup(self) -> None:  # noqa: B027
        """Optional: pre-load model weights. Called once at startup."""
