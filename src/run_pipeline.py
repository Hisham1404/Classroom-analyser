"""Run the whole offline pass: audio in, `results/*.json` out.

    python -m src.run_pipeline                       # every session, with ASR
    python -m src.run_pipeline --no-asr              # acoustic metrics only, fast
    python -m src.run_pipeline --sessions 1 --minutes 5

Transcription is cached by content hash, so a second run costs nothing and a bug in the
metrics never means re-transcribing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src import config
from src.asr import get_backend
from src.asr.base import TranscriptionError
from src.attribution import Role, assign_roles
from src.diarization import diarize, speech_spans, split_segments_by_speaker
from src.ingest import build_corpus
from src.pipeline import build_result, write_index, write_result
from src.signal_layer import build_timeline, load_waveform


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.run_pipeline",
        description="Offline batch pass over the classroom recordings.",
    )
    p.add_argument("--sessions", type=int, default=None, help="only the first N")
    p.add_argument("--minutes", type=float, default=None,
                   help="only analyse the first N minutes of each session")
    p.add_argument("--backend", default=None, help=f"override {config.ASR_BACKEND!r}")
    p.add_argument("--no-asr", action="store_true", help="acoustic metrics only")
    p.add_argument("--attribution", choices=["diarization", "energy"], default=None,
                   help=f"override {config.ATTRIBUTION_METHOD!r}")
    p.add_argument("--out-dir", default=str(config.RESULTS_DIR))
    return p


def _window(src: Path, seconds: float | None, dst: Path) -> Path:
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    if seconds:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)

    try:
        sessions = build_corpus(config.DATA_DIR)
    except FileNotFoundError:
        sessions = []
    if not sessions:
        print(f"error: no sessions under {config.DATA_DIR}", file=sys.stderr)
        return 1
    if args.sessions:
        sessions = sessions[:args.sessions]

    backend = None
    if not args.no_asr:
        try:
            backend = get_backend(args.backend)
            print(f"ASR: {backend.name} ({backend.model_id})")
        except (ValueError, RuntimeError) as exc:
            print(f"warning: no ASR ({exc}) — acoustic metrics only", file=sys.stderr)

    print(f"{len(sessions)} sessions -> {out_dir}\n")
    written = []

    for session in sessions:
        sid = session.meta.session_id
        started = time.perf_counter()
        seconds = args.minutes * 60 if args.minutes else None

        with tempfile.TemporaryDirectory() as tmp:
            wav = _window(session.probe.path, seconds, Path(tmp) / "window.wav")
            wave = load_waveform(wav)
            audio_sec = len(wave) / 16000

            method = args.attribution or config.ATTRIBUTION_METHOD

            # Diarization has to run first: where it succeeds, its segmentation is the
            # speech detector, not Silero. Silero was leaving most of the speech out on
            # the hands-on sessions and `classify_non_speech` filed it as activity noise.
            speaker_turns = None
            if method == "diarization":
                try:
                    speaker_turns = diarize(wav)
                except (RuntimeError, OSError, ImportError) as exc:
                    print(f"  {sid}: diarization unavailable ({exc}) — falling back to "
                          f"energy, which measured 41.7%", file=sys.stderr)

            if speaker_turns:
                timeline = build_timeline(wave, speech_spans=speech_spans(speaker_turns))
                segments = split_segments_by_speaker(timeline, speaker_turns)
                attribution_label = config.DIARIZATION_LABEL
                vad_label = config.DIARIZATION_VAD_LABEL
            else:
                timeline = build_timeline(wave)
                segments = assign_roles(timeline)
                attribution_label = config.ENERGY_LABEL
                vad_label = f"silero@{config.VAD_THRESHOLD}"
                speaker_turns = None

            transcript = None
            if backend is not None:
                try:
                    transcript = backend.transcribe(wav, language=config.ASR_LANGUAGE)
                except (TranscriptionError, FileNotFoundError) as exc:
                    print(f"  {sid}: ASR failed ({exc}) — acoustic only", file=sys.stderr)

            result = build_result(session, segments, audio_sec, transcript,
                                  attribution=attribution_label,
                                  speaker_turns=speaker_turns,
                                  vad=vad_label)

        path = write_result(result, out_dir)
        written.append(path)

        conf = result["confidence"]
        ttr = result["metrics"]["M1_teacher_talk_ratio"]["value"]
        print(f"  {sid:<32} {audio_sec/60:>5.1f} min  "
              f"conf {conf['overall']:.2f} {conf['verdict']:<10} "
              f"TTR {'—' if ttr is None else f'{ttr:.0%}':>4}  "
              f"{time.perf_counter()-started:>5.0f}s")

    # The dashboard is static and cannot list a directory, so without this a new session
    # is written to disk and stays invisible.
    write_index(out_dir)
    print(f"\nwrote {len(written)} files + index.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
