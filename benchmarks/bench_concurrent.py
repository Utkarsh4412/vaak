"""
benchmarks/bench_concurrent.py — Concurrency load benchmark for vaak.

Spawns N concurrent sessions simultaneously (N=1, 2, 4, 8) and measures
p50 and p95 time-to-first-audio (TTFA) per concurrency level.
Writes results to benchmarks/results/load_YYYYMMDD.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from vaak.agent import VoiceHandler
from vaak.config import AgentConfig
from vaak.interfaces import BaseLLM, BaseSTT, BaseTTS


class BenchmarkSTT(BaseSTT):
    def transcribe(self, audio: tuple[int, object], *, language: str | None = None) -> str:
        time.sleep(0.04)  # simulate STT inference latency
        return "Hello, how are you doing today?"


class BenchmarkLLM(BaseLLM):
    def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        time.sleep(0.12)
        return "I am doing well, thank you! How can I assist you?"

    def generate_stream(self, messages: list[dict[str, str]], **kwargs: object):
        time.sleep(0.06)  # TTFT
        words = [
            "I",
            " am",
            " doing",
            " well,",
            " thank",
            " you!",
            " How",
            " can",
            " I",
            " assist",
            " you?",
        ]
        yield from words


class BenchmarkTTS(BaseTTS):
    def synthesize(self, text: str, *, language: str = "en"):
        time.sleep(0.05)  # first TTS chunk latency
        yield (24000, np.zeros(2400, dtype=np.float32))
        yield (24000, np.zeros(2400, dtype=np.float32))


def get_hardware_info() -> dict[str, str | int]:
    try:
        ollama_ver = subprocess.check_output(
            ["ollama", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        ollama_ver = "unknown"

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "cores": os.cpu_count() or -1,
        "python": platform.python_version(),
        "ollama_version": ollama_ver,
    }


def _worker(
    idx: int,
    barrier: threading.Barrier,
    handler: VoiceHandler,
    audio: tuple[int, np.ndarray],
    ttfa_out: list[float],
) -> None:
    barrier.wait()
    t0 = time.perf_counter()
    for _chunk in handler(audio):
        ttfa_out[idx] = (time.perf_counter() - t0) * 1000.0
        break


def run_concurrent_load(
    n_sessions: int,
    reps: int,
    max_concurrent: int = 4,
) -> dict[str, float]:
    """Run `reps` rounds of `n_sessions` simultaneous turns and return TTFA stats."""
    cfg = AgentConfig(max_concurrent_sessions=max_concurrent)
    template = VoiceHandler(
        cfg,
        _stt=BenchmarkSTT(),
        _llm=BenchmarkLLM(),
        _tts=BenchmarkTTS(),
    )
    dummy_audio = (16000, np.zeros(1600, dtype=np.float32))

    all_ttfa_ms: list[float] = []

    for _rep in range(reps):
        threads: list[threading.Thread] = []
        ttfa_per_thread: list[float] = [0.0] * n_sessions
        start_barrier = threading.Barrier(n_sessions)

        for i in range(n_sessions):
            handler = template.copy()
            t = threading.Thread(
                target=_worker,
                args=(i, start_barrier, handler, dummy_audio, ttfa_per_thread),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        all_ttfa_ms.extend(ttfa_per_thread)

    return {
        "p50_ms": float(np.percentile(all_ttfa_ms, 50)),
        "p95_ms": float(np.percentile(all_ttfa_ms, 95)),
        "min_ms": float(np.min(all_ttfa_ms)),
        "max_ms": float(np.max(all_ttfa_ms)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run concurrency load benchmarks.")
    parser.add_argument(
        "--n", nargs="+", type=int, default=[1, 2, 4], help="Concurrency levels (e.g. 1 2 4)"
    )
    parser.add_argument("--reps", type=int, default=3, help="Reps per concurrency level")
    parser.add_argument(
        "--max-concurrent", type=int, default=4, help="Config max_concurrent_sessions"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results: dict[str, object] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "hardware": get_hardware_info(),
        "load_levels": {},
    }

    print(f"Running concurrency benchmarks: levels={args.n}, reps={args.reps}...")
    for n in args.n:
        stats = run_concurrent_load(n, reps=args.reps, max_concurrent=args.max_concurrent)
        results["load_levels"][str(n)] = stats
        print(f"N={n:2d} | p50={stats['p50_ms']:.1f} ms | p95={stats['p95_ms']:.1f} ms")

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        out_path = results_dir / f"load_{date_str}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
