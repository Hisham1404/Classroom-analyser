"""Probe the MakerGhat classroom recordings: what language, how intelligible, how noisy.

Samples short clips from several points in each recording, runs Whisper language ID on
each, and transcribes one clip per session so we can eyeball real output quality.

Usage:  python scripts/probe_audio.py [--model small] [--clip 30]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_ROOT = Path(__file__).resolve().parent.parent / "data" / "audio"
SAMPLE_POINTS = (0.15, 0.40, 0.65, 0.90)  # fractions through the recording


def audio_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_clip(src: Path, start: float, length: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--clip", type=float, default=30.0)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    print(f"loading whisper '{args.model}' (int8/cpu) ...", flush=True)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("loaded\n", flush=True)

    sessions = sorted(p for p in AUDIO_ROOT.iterdir() if p.is_dir())
    report = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for sess in sessions:
            mp3 = sess / f"{sess.name}.mp3"
            meta = json.loads((sess / f"{sess.name}.json").read_text(encoding="utf-8"))
            dur = audio_duration(mp3)

            print("=" * 78)
            print(f"{sess.name}   {dur/60:.1f} min audio   "
                  f"activity={meta.get('activityType')}   "
                  f"teacher={meta.get('photoMetadata', {}).get('teacherName')}   "
                  f"students={meta.get('totalStudents')}")
            print("=" * 78, flush=True)

            langs = []
            for i, frac in enumerate(SAMPLE_POINTS):
                start = max(0.0, dur * frac)
                if start + args.clip > dur:
                    continue
                clip = tmpdir / f"{sess.name}_{i}.wav"
                extract_clip(mp3, start, args.clip, clip)

                samples = decode_audio(str(clip), sampling_rate=16000)
                lang, prob, all_probs = model.detect_language(samples)
                pairs = all_probs.items() if hasattr(all_probs, "items") else all_probs
                top = sorted(pairs, key=lambda kv: -kv[1])[:3]
                top_s = "  ".join(f"{k}:{v:.2f}" for k, v in top)
                print(f"  @{start/60:5.1f}min  ->  {lang} ({prob:.2f})   [{top_s}]", flush=True)
                langs.append(lang)

            # transcribe the mid-point clip so we can see real text
            mid = tmpdir / f"{sess.name}_1.wav"
            if mid.exists():
                print("\n  --- sample transcript (mid-recording, 30s) ---", flush=True)
                segs, info = model.transcribe(
                    str(mid), beam_size=1, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                text_bits = []
                for s in segs:
                    line = s.text.strip()
                    if line:
                        text_bits.append(line)
                        print(f"  [{s.start:5.1f}-{s.end:5.1f}] {line}", flush=True)
                if not text_bits:
                    print("  (VAD found no speech in this clip)", flush=True)
                print(f"  detected={info.language} p={info.language_probability:.2f}\n", flush=True)

            report.append({
                "session": sess.name,
                "audio_min": round(dur / 60, 1),
                "meta_min": round(meta["duration"] / 60000, 1),
                "activity": meta.get("activityType"),
                "students": meta.get("totalStudents"),
                "langs": langs,
            })

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in report:
        print(f"{r['session']:<32} {r['audio_min']:>6.1f}/{r['meta_min']:<6.1f}min  "
              f"{r['activity']:<16} n={r['students']:<3} langs={r['langs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
