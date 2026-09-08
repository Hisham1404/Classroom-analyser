"""Find VAD settings that actually work on this audio.

The architecture puts VAD under the whole acoustic layer — M1 talk-time, M2 participation,
M3 interaction density, M4 longest stretch. If VAD is wrong, every "robust" metric is wrong,
silently. onnx-asr's Silero defaults returned *zero* segments on 2 of 3 real clips, so this
has to be settled before Phase 3 builds on it.

Sweeps thresholds against the real recordings and reports speech coverage per setting.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def sweep(clips: list[str]) -> None:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    SR = 16000
    grid = [
        # threshold, min_speech_ms, min_silence_ms
        (0.50, 250, 2000),   # faster-whisper defaults
        (0.50, 250, 400),    # what probe_vad.py used in Phase 0
        (0.35, 200, 300),
        (0.25, 150, 250),
        (0.20, 100, 200),
        (0.15, 100, 150),
    ]

    print(f"{'threshold':>9} {'min_speech':>11} {'min_sil':>8} | " +
          " | ".join(f"{Path(c).name[:14]:>14}" for c in clips) + " |    mean")
    print("-" * (32 + 17 * len(clips) + 10))

    for thr, msp, msl in grid:
        cov = []
        for clip in clips:
            audio = load_audio(clip, SR)
            opts = VadOptions(threshold=thr, min_speech_duration_ms=msp,
                              min_silence_duration_ms=msl)
            try:
                segs = get_speech_timestamps(audio, opts)
            except TypeError:
                segs = get_speech_timestamps(audio, VadOptions(threshold=thr))
            speech = sum((s["end"] - s["start"]) for s in segs) / SR
            cov.append(100 * speech / (len(audio) / SR))
        mean = sum(cov) / len(cov)
        print(f"{thr:>9.2f} {msp:>11} {msl:>8} | " +
              " | ".join(f"{c:>13.1f}%" for c in cov) + f" | {mean:>6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=3)
    args = ap.parse_args()

    clips = sorted(glob.glob("eval/clips/*.wav"))[:args.clips]
    if not clips:
        print("no clips found in eval/clips/", file=sys.stderr)
        return 1

    print(f"VAD sweep over {len(clips)} real classroom clips (30 s each)\n")
    sweep(clips)
    print("\nA setting is usable when coverage is non-zero on every clip and roughly "
          "matches the 20-86% speech density measured in Phase 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
