"""
vaak/barge_in.py — BargeInController for mid-reply cancellation.

The VAD fires when the user speaks mid-reply. The controller sets a
threading.Event that the Pipeline checks between TTS chunks. When set,
the pipeline stops synthesis and discards the partial reply.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from vaak.session import Session


class BargeInController:
    """Coordinates barge-in cancellation across the pipeline.

    Thread-safe: request_cancel() may be called from the VAD thread while
    the Pipeline reads the cancel event on the audio output thread.

    Usage::

        barge_in = BargeInController()
        session = Session()
        session.cancel = barge_in.event  # pipeline reads session.cancel

        # In VAD callback when new speech starts:
        barge_in.request_cancel()

        # Pipeline calls between chunks:
        if session.cancel.is_set():
            break
    """

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event if event is not None else threading.Event()

    def attach_session(self, session: Session) -> None:
        """Attach to a session's cancel event."""
        self._event = session.cancel

    @property
    def event(self) -> threading.Event:
        """The cancel event that the pipeline polls."""
        return self._event

    def request_cancel(self) -> None:
        """Signal the pipeline to stop TTS synthesis.

        Idempotent: safe to call multiple times. The event stays set
        until reset() is called.
        """
        logger.debug("BargeInController: cancel requested")
        self._event.set()

    def reset(self) -> None:
        """Clear the cancel signal at the start of a new turn."""
        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        """True if a cancel request is pending."""
        return self._event.is_set()
