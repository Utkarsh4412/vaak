"""
vaak/llm/ollama.py — OllamaLLM adapter (blocking + streaming).

Wraps ollama.chat() with:
  - Configurable keep_alive to keep the model loaded between requests.
  - Startup warmup ping to avoid paying model-load latency on the first call.
  - One automatic retry on failure.
  - generate_stream() for token-by-token streaming via stream=True.
"""

from __future__ import annotations

import time
from collections.abc import Generator

import ollama
from loguru import logger

from vaak.interfaces import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama-backed LLM with blocking and streaming modes.

    Args:
        model: Ollama model tag (e.g. "llama3.2:1b").
        keep_alive: How long Ollama keeps the model in VRAM/RAM between
                    requests (e.g. "10m"). Passed directly to ollama.chat().
        ollama_host: Base URL of the Ollama REST server.
    """

    def __init__(
        self,
        *,
        model: str = "llama3.2:1b",
        keep_alive: str = "10m",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._keep_alive = keep_alive
        self._client = ollama.Client(host=ollama_host)

    # ------------------------------------------------------------------
    # BaseLLM implementation — blocking
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a complete reply (non-streaming).

        Retries once on transient failures. Returns a fallback error string
        rather than raising, so the pipeline can still yield a TTS response.
        """
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                t0 = time.perf_counter()
                response = self._client.chat(
                    model=self._model,
                    messages=messages,
                    options=options,
                    keep_alive=self._keep_alive,
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                text: str = response["message"]["content"]
                logger.debug(
                    f"OllamaLLM generated {len(text)} chars in {elapsed_ms} ms "
                    f"(model={self._model})"
                )
                return text
            except Exception as exc:
                last_exc = exc
                logger.warning(f"OllamaLLM attempt {attempt + 1}/2 failed: {exc}")

        logger.error(f"OllamaLLM failed after 2 attempts: {last_exc}")
        return "I am having trouble connecting to my local LLM. Please try again later."

    # ------------------------------------------------------------------
    # BaseLLM implementation — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Yield reply tokens one at a time via Ollama's streaming API.

        On failure, yields the full error fallback string as a single token
        so the pipeline can still produce TTS output.
        """
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            t0 = time.perf_counter()
            stream = self._client.chat(
                model=self._model,
                messages=messages,
                options=options,
                keep_alive=self._keep_alive,
                stream=True,
            )
            token_count = 0
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    token_count += 1
                    yield token
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.debug(
                f"OllamaLLM streamed {token_count} tokens in {elapsed_ms} ms (model={self._model})"
            )
        except Exception as exc:
            logger.error(f"OllamaLLM streaming failed: {exc}")
            yield "I am having trouble connecting to my local LLM. Please try again later."

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Ping Ollama with a minimal prompt to load the model into memory.

        This avoids the model-load latency hit on the first real user request.
        Failure is logged but does not prevent startup.
        """
        logger.info(f"Warming up OllamaLLM model '{self._model}'…")
        try:
            self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
                keep_alive=self._keep_alive,
            )
            logger.info(f"OllamaLLM '{self._model}' warm-up complete.")
        except Exception as exc:
            logger.warning(
                f"OllamaLLM warm-up failed (Ollama may not be running): {exc}. "
                "First request will pay model-load latency."
            )
