"""
vaak/agent.py — FastRTC adapter creating a per-connection Session.

FastRTC's ReplyOnPause calls handler.copy() for each new WebRTC connection,
then calls the copy's __call__ method for each audio segment. This module
wraps the Pipeline in a handler class that:

  1. Stores all constructor config.
  2. Creates a fresh Session in copy() so each caller gets isolated history.
  3. Delegates process_audio() to Pipeline.run().
  4. Exposes a build_stream() helper that creates the FastRTC Stream.

The shared-memory bug in the original implementation (a single
history deque shared across all WebRTC callers) is fixed here:
copy() creates a new Session(uuid4()) for every connection.
"""

from __future__ import annotations

from collections.abc import Generator

from fastrtc import ReplyOnPause, Stream
from loguru import logger

from vaak.barge_in import BargeInController
from vaak.config import AgentConfig
from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS
from vaak.llm.ollama import OllamaLLM
from vaak.pipeline import Pipeline
from vaak.session import Session
from vaak.stt.faster_whisper import FasterWhisperSTT
from vaak.stt.lang_detect import LangDetectRouter
from vaak.stt.moonshine import MoonshineSTT
from vaak.tts.kokoro import KokoroTTS


class VoiceHandler:
    """FastRTC-compatible audio handler.

    FastRTC calls handler.copy() per WebRTC connection and __call__ per segment.
    All heavy objects (STT model, LLM client, TTS model, Pipeline) are shared
    across copies — they are thread-safe read-only objects. Only the Session is
    per-connection.

    Args:
        config: Validated AgentConfig.
        _session: Internal; set by copy(). Do not pass externally.
        _pipeline: Internal; set by copy(). Do not pass externally.
        _barge_in: Internal; set by copy(). Do not pass externally.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        _stt: BaseSTT | None = None,
        _llm: BaseLLM | None = None,
        _tts: BaseTTS | None = None,
        _session: Session | None = None,
        _pipeline: Pipeline | None = None,
        _barge_in: BargeInController | None = None,
    ) -> None:
        self._config = config

        # Shared (expensive) objects — created once, referenced by all copies.
        self._stt = _stt
        self._llm = _llm
        self._tts = _tts

        # Per-connection state — None on the "template" instance.
        self._session = _session
        self._pipeline = _pipeline
        self._barge_in = _barge_in

    @property
    def session(self) -> Session | None:
        """The active Session for this handler connection, if any."""
        return self._session

    @property
    def barge_in(self) -> BargeInController | None:
        """The active BargeInController for this handler connection, if any."""
        return self._barge_in

    # ------------------------------------------------------------------
    # FastRTC protocol: copy() creates a per-connection instance
    # ------------------------------------------------------------------

    def copy(self) -> VoiceHandler:
        """Called by FastRTC once per new WebRTC connection.

        Creates a fresh Session (new uuid4) so each caller has isolated history.
        Heavy model objects are re-used from the template instance.
        """
        cfg = self._config
        session = Session(
            language=cfg.language,
            memory_turns=cfg.memory.memory_turns,
            summarize_threshold=cfg.memory.summarize_threshold,
        )
        barge_in = BargeInController(session.cancel)
        logger.info(f"New WebRTC connection — Session {session.id!r} lang={session.language!r}")
        assert self._stt is not None
        assert self._llm is not None
        assert self._tts is not None
        pipeline = Pipeline(
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            system_prompt=cfg.system_prompt,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            llm_error_reply=cfg.llm_error_reply,
        )
        return VoiceHandler(
            cfg,
            _stt=self._stt,
            _llm=self._llm,
            _tts=self._tts,
            _session=session,
            _pipeline=pipeline,
            _barge_in=barge_in,
        )

    # ------------------------------------------------------------------
    # FastRTC protocol: __call__ handles each audio segment
    # ------------------------------------------------------------------

    def handle_barge_in(self) -> bool:
        """Handle incoming speech during synthesis (barge-in).

        If the agent is currently speaking, requests pipeline cancellation and
        returns True. Returns False if agent was not speaking (not a barge-in).
        """
        if self._session is not None and self._session.is_speaking and self._barge_in is not None:
            logger.info(f"Barge-in triggered for session {self._session.id}")
            self._barge_in.request_cancel()
            return True
        return False

    def __call__(self, audio: tuple[int, object]) -> Generator[tuple[int, object], None, None]:
        """Process one audio segment — called by ReplyOnPause after VAD fires."""
        if self._session is None or self._pipeline is None:
            # Safety net: should never be called on the template instance.
            logger.error("VoiceHandler.__call__ on template (no session). Skipping.")
            return
        yield from self._pipeline.run(self._session, audio)

    # ------------------------------------------------------------------
    # FastRTC Stream builder
    # ------------------------------------------------------------------

    def build_stream(self) -> Stream:
        """Create and return the FastRTC Stream using ReplyOnPause."""
        model_options = None
        try:
            from fastrtc.pause_detection.silero import SileroVadOptions

            model_options = SileroVadOptions(
                threshold=self._config.vad_threshold,
                min_silence_duration_ms=self._config.min_silence_ms,
            )
        except Exception as exc:
            logger.warning(f"Could not initialize SileroVadOptions: {exc}")

        return Stream(
            ReplyOnPause(
                self,  # type: ignore[arg-type]
                model_options=model_options,
                can_interrupt=True,
            ),
            modality="audio",
            mode="send-receive",
        )


# ---------------------------------------------------------------------------
# Factory: build a ready-to-serve VoiceHandler from an AgentConfig
# ---------------------------------------------------------------------------


def build_agent(config: AgentConfig) -> VoiceHandler:
    """Construct and warm up all components, return the FastRTC handler.

    Args:
        config: Validated AgentConfig.

    Returns:
        A VoiceHandler template; FastRTC will call .copy() per connection.
    """
    logger.info("Building vaak agent…")

    # STT
    stt: BaseSTT
    if config.stt.backend == "faster_whisper":
        stt = FasterWhisperSTT(
            model_size_or_path=config.stt.faster_whisper_model,
            compute_type=config.stt.faster_whisper_compute_type,
            default_language=config.language,
        )
    elif config.stt.backend == "router":
        stt = LangDetectRouter(
            MoonshineSTT(),
            FasterWhisperSTT(
                model_size_or_path=config.stt.faster_whisper_model,
                compute_type=config.stt.faster_whisper_compute_type,
                default_language=config.language,
            ),
        )
    else:
        stt = MoonshineSTT()
    stt.warmup()

    # LLM
    llm = OllamaLLM(
        model=config.llm.model,
        keep_alive=config.llm.keep_alive,
        ollama_host=config.llm.ollama_host,
    )
    llm.warmup()

    # TTS
    tts = KokoroTTS(
        voice=config.tts.voice,
        speed=config.tts.speed,
        skip_voice_check=config.tts.skip_voice_check,
    )
    tts.warmup()

    handler = VoiceHandler(config, _stt=stt, _llm=llm, _tts=tts)
    logger.info("vaak agent ready.")
    return handler
