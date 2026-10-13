"""
vaak/pipeline.py — Pipeline.run(session, audio) stitching STT → LLM → TTS.

1. STT: transcribe the audio segment.
2. LLM: stream tokens from the language model (or generate complete reply if non-streaming).
3. Sentence chunker: split the token stream on sentence boundaries.
4. TTS: synthesise each sentence and yield PCM chunks immediately.

First audio reaches the speaker as soon as the LLM finishes its first
sentence — no need to wait for the complete reply.

History is appended only on successful completion (not on error or cancel).
The cancel event is checked between TTS chunks to support barge-in.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Generator, Iterable

from loguru import logger

from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.metrics import LatencyReport, StageTimer
from vaak.session import Session

# Sentence boundary: split after .!?; followed by whitespace (or end of stream).
# Avoids splitting abbreviations like "Dr." or "U.S." by requiring the char
# after the period to be whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


def sentence_chunker(token_stream: Iterable[str]) -> Generator[str, None, None]:
    """Accumulate tokens and yield complete sentences.

    Splits the token stream on sentence-ending punctuation (.!?;) followed
    by whitespace. Any remaining text is yielded when the stream ends.

    Args:
        token_stream: Iterable yielding token strings from an LLM.

    Yields:
        Sentence strings, each stripped of leading/trailing whitespace.
    """
    buffer = ""
    for token in token_stream:
        buffer += token
        # Split on sentence boundaries
        parts = _SENTENCE_RE.split(buffer)
        # Yield all complete sentences (everything except the last fragment)
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if sentence:
                yield sentence
        # Keep the last fragment as the buffer
        buffer = parts[-1]

    # Yield any remaining text
    buffer = buffer.strip()
    if buffer:
        yield buffer


class Pipeline:
    """Orchestrates a single STT → LLM → TTS turn with optional streaming.

    Args:
        stt: STT adapter implementing BaseSTT.
        llm: LLM adapter implementing BaseLLM.
        tts: TTS adapter implementing BaseTTS.
        system_prompt: System message injected at the front of every context.
        max_tokens: Upper bound on tokens to generate.
        temperature: LLM sampling temperature.
        top_p: LLM nucleus sampling threshold.
        llm_error_reply: Fallback text spoken via TTS when the LLM fails.
        on_report: Optional callback invoked with the LatencyReport after each turn.
        streaming: If True, stream tokens to sentence-chunked TTS. If False,
                   wait for complete LLM reply before synthesizing.
    """

    def __init__(
        self,
        *,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        system_prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        llm_error_reply: str = "I'm having trouble thinking right now. Please try again later.",
        on_report: Callable[[LatencyReport], None] | None = None,
        streaming: bool = True,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._llm_error_reply = llm_error_reply
        self._on_report = on_report
        self._streaming = streaming

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        session: Session,
        audio: tuple[int, object],
    ) -> Generator[tuple[int, object], None, None]:
        """Execute one turn: STT → LLM → TTS.

        Yields (sample_rate, samples) PCM chunks as they come from TTS.
        History is updated only on successful, uninterrupted completion.

        Args:
            session: Caller's Session instance (carries language, history, cancel).
            audio: (sample_rate, samples) audio segment from FastRTC VAD.

        Yields:
            PCM chunks suitable for passing to FastRTC's reply mechanism.
        """
        log = logger.bind(session_id=session.id, language=session.language)
        session.reset_cancel()
        session.is_speaking = False

        stt_ms = 0.0
        llm_first_ms = 0.0
        llm_total_ms = 0.0
        tts_first_ms = 0.0
        tts_total_ms = 0.0
        first_audio_ms = 0.0
        completed = False
        llm_failed = False
        cancelled = False
        user_text = ""
        reply_text = ""
        t_tts = None

        try:
            with StageTimer() as t_e2e:
                # ----------------------------------------------------------
                # Stage 1: STT
                # ----------------------------------------------------------
                with StageTimer() as t_stt:
                    try:
                        user_text = self._stt.transcribe(audio)
                    except Exception as exc:
                        log.error(f"STT error: {exc}")
                        user_text = ""
                stt_ms = t_stt.elapsed_ms
                log.debug(f"STT transcript: {user_text!r}")
                u_len = len(user_text)
                log.info(f"STT {stt_ms:.0f} ms | chars={u_len}")

                if not user_text.strip():
                    log.debug("Empty STT transcript; skipping turn.")
                    return

                context = session.build_context(self._system_prompt)
                context.append({"role": "user", "content": user_text})

                if not self._streaming:
                    # ------------------------------------------------------
                    # Non-streaming (blocking) pipeline: Stage 2 LLM
                    # ------------------------------------------------------
                    with StageTimer() as t_llm:
                        try:
                            reply_text = self._llm.generate(
                                context,
                                max_tokens=self._max_tokens,
                                temperature=self._temperature,
                                top_p=self._top_p,
                            )
                        except Exception as exc:
                            log.error(f"LLM error: {exc}")
                            reply_text = self._llm_error_reply
                            llm_failed = True
                    llm_total_ms = t_llm.elapsed_ms
                    llm_first_ms = llm_total_ms
                    r_len = len(reply_text)
                    if not llm_failed:
                        log.debug(f"LLM reply: {reply_text!r}")
                        log.info(f"LLM {llm_total_ms:.0f} ms | chars={r_len}")
                    else:
                        log.debug(f"LLM fallback reply: {reply_text!r}")
                        log.info(f"LLM {llm_total_ms:.0f} ms | chars={r_len} (fallback)")

                    if session.cancel.is_set():
                        log.info("Cancelled after LLM; discarding reply.")
                        cancelled = True
                        return

                    # ------------------------------------------------------
                    # Non-streaming: Stage 3 TTS
                    # ------------------------------------------------------
                    session.is_speaking = True
                    first_chunk = True

                    with StageTimer() as t_tts:
                        try:
                            for chunk in self._tts.synthesize(
                                reply_text, language=session.language
                            ):
                                if first_chunk:
                                    tts_first_ms = t_tts.elapsed_ms
                                    first_audio_ms = t_e2e.elapsed_ms
                                    first_chunk = False

                                if session.cancel.is_set():
                                    log.info("Barge-in detected mid-TTS; stopping synthesis.")
                                    cancelled = True
                                    break
                                yield chunk
                            else:
                                completed = True
                        except Exception as exc:
                            log.error(f"TTS error: {exc}")
                        finally:
                            session.is_speaking = False
                    log.info(f"TTS {t_tts.elapsed_ms:.0f} ms | completed={completed}")

                else:
                    # ------------------------------------------------------
                    # Streaming pipeline: Token streaming → Chunker → TTS
                    # ------------------------------------------------------
                    def safe_tokens() -> Generator[str, None, None]:
                        nonlocal llm_failed, llm_first_ms, llm_total_ms
                        t0 = time.perf_counter()
                        first_tok = True
                        try:
                            stream: Iterable[str]
                            if callable(getattr(self._llm, "generate_stream", None)):
                                stream = self._llm.generate_stream(
                                    context,
                                    max_tokens=self._max_tokens,
                                    temperature=self._temperature,
                                    top_p=self._top_p,
                                )
                            else:
                                stream = [
                                    self._llm.generate(
                                        context,
                                        max_tokens=self._max_tokens,
                                        temperature=self._temperature,
                                        top_p=self._top_p,
                                    )
                                ]

                            for token in stream:
                                if first_tok:
                                    llm_first_ms = (time.perf_counter() - t0) * 1000.0
                                    first_tok = False
                                yield token
                            llm_total_ms = (time.perf_counter() - t0) * 1000.0
                        except Exception as exc:
                            log.error(f"LLM streaming error: {exc}")
                            llm_failed = True
                            llm_total_ms = (time.perf_counter() - t0) * 1000.0
                            if first_tok:
                                yield self._llm_error_reply

                    session.is_speaking = True
                    first_tts_chunk = True
                    sentence_count = 0
                    sentences = sentence_chunker(safe_tokens())

                    with StageTimer() as t_tts:
                        try:
                            for sentence in sentences:
                                if not sentence.strip():
                                    continue

                                sentence_count += 1
                                reply_text += (" " if reply_text else "") + sentence

                                if session.cancel.is_set():
                                    log.info("Barge-in detected; stopping before sentence TTS.")
                                    cancelled = True
                                    break

                                for chunk in self._tts.synthesize(
                                    sentence, language=session.language
                                ):
                                    if first_tts_chunk:
                                        tts_first_ms = t_tts.elapsed_ms
                                        first_audio_ms = t_e2e.elapsed_ms
                                        first_tts_chunk = False

                                    if session.cancel.is_set():
                                        log.info("Barge-in detected mid-TTS; stopping synthesis.")
                                        cancelled = True
                                        break
                                    yield chunk

                                if cancelled:
                                    break

                            if not cancelled:
                                completed = True

                        except Exception as exc:
                            log.error(f"TTS error: {exc}")
                        finally:
                            session.is_speaking = False

                    r_len = len(reply_text)
                    if not llm_failed:
                        log.debug(f"LLM reply: {reply_text!r}")
                        log.info(
                            f"Streaming LLM+TTS | sentences={sentence_count} "
                            f"chars={r_len} | completed={completed}"
                        )
                    else:
                        log.debug(f"LLM fallback reply: {reply_text!r}")
                        log.info(
                            f"LLM failed, TTS fallback | chars={r_len} | completed={completed}"
                        )

                # ----------------------------------------------------------
                # Commit to history only on clean completion with no LLM error
                # ----------------------------------------------------------
                if completed and not llm_failed:
                    session.append_turn(user_text, reply_text)
                    log.debug(f"History updated ({len(session.history)} msgs).")
                else:
                    if llm_failed:
                        log.debug("LLM failed; history NOT updated.")
                    elif cancelled:
                        log.debug("Turn cancelled; history NOT updated.")

        finally:
            if t_tts is not None:
                tts_total_ms = t_tts.elapsed_ms

            # Emit report only if STT returned text (not an empty turn)
            if user_text.strip():
                report = LatencyReport(
                    session_id=session.id,
                    language=session.language,
                    stt_ms=stt_ms,
                    llm_first_token_ms=llm_first_ms,
                    llm_total_ms=llm_total_ms,
                    tts_first_chunk_ms=tts_first_ms,
                    tts_total_ms=tts_total_ms,
                    first_audio_ms=first_audio_ms,
                    turn_ms=t_e2e.elapsed_ms,
                )
                log.info(f"Turn metrics: {report.to_json()}")
                if self._on_report:
                    self._on_report(report)
