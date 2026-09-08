"""Estimate speech density per recording using Silero VAD (no ASR).

VAD is robust where transcription is not, so this tells us how much usable speech
each session actually holds before we commit to any ASR strategy.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

AUDIO_ROOT = Path(__file__).resolve().parent.parent / "data" / "audio"
N_CLIPS = 8
CLIP_SEC = 30.0
SR = 16000


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main() -> None:
    print(f"{'SESSION':<32} {'speech%':>8} {'turns/min':>10} {'medturn_s':>10} {'rms_dB':>8}")
    print("-" * 74)
    with tempfile.TemporaryDirectory() as tmp:
        for sess in sorted(p for p in AUDIO_ROOT.iterdir() if p.is_dir()):
            mp3 = sess / f"{sess.name}.mp3"
            dur = duration(mp3)
            speech_s = total_s = 0.0
            turns: list[float] = []
            rms_vals: list[float] = []

            for i in range(N_CLIPS):
                start = dur * (i + 0.5) / N_CLIPS
                if start + CLIP_SEC > dur:
                    continue
                wav = Path(tmp) / f"c{i}.wav"
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.2f}",
                     "-t", f"{CLIP_SEC}", "-i", str(mp3), "-ac", "1", "-ar", str(SR), str(wav)],
                    check=True)
                samples = decode_audio(str(wav), sampling_rate=SR)
                segs = get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=400))
                for s in segs:
                    t = (s["end"] - s["start"]) / SR
                    speech_s += t
                    turns.append(t)
                total_s += len(samples) / SR
                rms_vals.append(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-12))

            pct = 100 * speech_s / total_s if total_s else 0.0
            tpm = 60 * len(turns) / total_s if total_s else 0.0
            med = float(np.median(turns)) if turns else 0.0
            rms_db = 20 * np.log10(float(np.mean(rms_vals))) if rms_vals else float("nan")
            print(f"{sess.name:<32} {pct:>7.1f}% {tpm:>10.1f} {med:>10.2f} {rms_db:>8.1f}")


if __name__ == "__main__":
    main()
