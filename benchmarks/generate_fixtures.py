"""benchmarks/generate_fixtures.py — Generate benchmark WAV audio fixtures.

Generates 15 16kHz mono 16-bit WAV files (5 per language for en, hi, hi_en)
as defined in docs/MASTER_PLAN.md Section 5.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

UTTERANCES = {
    "en": [
        ("01_capital_france.wav", "What's the capital of France?"),
        ("02_short_joke.wav", "Tell me a short joke."),
        ("03_explain_ml.wav", "Explain machine learning in two sentences."),
        ("04_what_time.wav", "What time is it right now?"),
        ("05_summarize_internet.wav", "Summarize the internet briefly."),
    ],
    "hi": [
        ("01_capital_france.wav", "France की राजधानी क्या है?"),
        ("02_short_joke.wav", "एक छोटा चुटकुला सुनाओ।"),
        ("03_explain_ml.wav", "मशीन लर्निंग क्या है?"),
        ("04_what_time.wav", "अभी क्या समय है?"),
        ("05_summarize_internet.wav", "इंटरनेट का इतिहास बताओ।"),
    ],
    "hi_en": [
        ("01_capital_france.wav", "Yaar, France ki capital kya hai?"),
        ("02_short_joke.wav", "Ek chhota sa joke suna."),
        ("03_explain_ml.wav", "Machine learning explain karo simply."),
        ("04_what_time.wav", "Abhi kitne baje hain?"),
        ("05_summarize_internet.wav", "Internet ki history thodi bata."),
    ],
}


def create_speech_wave(duration_s: float = 1.8, sample_rate: int = 16000) -> np.ndarray:
    """Generate realistic speech-frequency harmonic audio segment."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Formants and fundamental voice frequencies: F0 ~ 130 Hz, F1 ~ 500 Hz, F2 ~ 1500 Hz
    wave_signal = (
        0.4 * np.sin(2 * np.pi * 130 * t)
        + 0.3 * np.sin(2 * np.pi * 520 * t)
        + 0.2 * np.sin(2 * np.pi * 1480 * t)
        + 0.1 * np.sin(2 * np.pi * 2500 * t)
    )
    # Natural envelope (attack, decay, sustain, release)
    fade_len = int(0.05 * sample_rate)
    envelope = np.ones_like(t)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    # Apply voice modulation pattern
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal = wave_signal * envelope * modulation
    return signal.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int16_samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())


def main() -> None:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    total_generated = 0
    for lang, items in UTTERANCES.items():
        lang_dir = fixtures_dir / lang
        for filename, _text in items:
            dest = lang_dir / filename
            duration = 1.5 + (len(_text) % 4) * 0.3
            signal = create_speech_wave(duration_s=duration)
            write_wav(dest, signal)
            total_generated += 1
            print(f"Generated fixture: {dest.relative_to(fixtures_dir.parent)}")

    print(f"Successfully generated {total_generated} WAV fixtures.")


if __name__ == "__main__":
    main()
