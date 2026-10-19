"""local_voice_chat.py — Backwards-compatible CLI entrypoint.

This shim preserves backwards compatibility for legacy invocations by
forwarding arguments to the vaak CLI entrypoint.
"""

from __future__ import annotations

import sys

from vaak.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
