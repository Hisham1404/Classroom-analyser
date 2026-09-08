"""Transcribe one audio file from the command line.

    python -m src.cli data/audio/OD11165_.../OD11165_....mp3
    python -m src.cli <file> --backend groq -o out.json

Which backend runs by default is one line in `src/config.py` (`ASR_BACKEND`);
`--backend` overrides it for a single run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.asr import BACKENDS, get_backend
from src.asr.base import TranscriptionError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Transcribe classroom audio with the configured ASR backend.",
    )
    p.add_argument("audio", help="path to an audio file")
    p.add_argument("--backend", choices=sorted(BACKENDS), default=None,
                   help=f"override config.ASR_BACKEND (currently {config.ASR_BACKEND!r})")
    p.add_argument("--language", default=config.ASR_LANGUAGE,
                   help=f"ISO language code (default {config.ASR_LANGUAGE!r})")
    p.add_argument("--model", default=None, help="override the backend's default model")
    p.add_argument("--no-cache", action="store_true", help="re-transcribe even if cached")
    p.add_argument("-o", "--out", default=None, help="write the transcript as JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = Path(args.audio)

    if not audio.is_file():
        print(f"error: audio file not found: {audio}", file=sys.stderr)
        return 1

    try:
        kwargs = {"model": args.model} if args.model else {}
        backend = get_backend(args.backend, **kwargs)
        transcript = backend.transcribe(audio, language=args.language,
                                        use_cache=not args.no_cache)
    except (RuntimeError, TranscriptionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    logprob = transcript.mean_logprob
    print(f"backend  {transcript.backend}  ({transcript.model})")
    print(f"language {transcript.language}"
          f"{f'  p={transcript.language_prob:.2f}' if transcript.language_prob else ''}")
    print(f"audio    {transcript.audio_sec / 60:.1f} min"
          f"   speech {transcript.speech_density:.0%}"
          f"   segments {len(transcript.segments)}"
          f"{f'   mean logprob {logprob:.2f}' if logprob is not None else ''}")
    print()
    print(transcript.text or "(no speech detected)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
