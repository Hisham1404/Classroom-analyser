"""Produce the hand-labelling sheet for teacher/student attribution.

    python scripts/make_labelling_sheet.py --minutes 5 --sessions 3

Writes `eval/labels/<session>.md` plus one small wav per speech segment, so each row can
be played without seeking through an hour of audio. Fill in the Truth column, then run
`scripts/score_labels.py` to turn the classifier from an assertion into a number.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                          # noqa: E402
from src.attribution import assign_roles                        # noqa: E402
from src.evaluation.labelling import build_sheet, render_sheet  # noqa: E402
from src.ingest import build_corpus                             # noqa: E402
from src.asr import get_backend                                # noqa: E402
from src.asr.base import TranscriptionError                    # noqa: E402
from src.signal_layer import load_waveform, build_timeline      # noqa: E402


def slice_audio(src: Path, start: float, seconds: float, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(src),
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0, help="window per session")
    ap.add_argument("--offset-min", type=float, default=10.0, help="where to start")
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--out-dir", default=str(config.EVAL_DIR / "labels"))
    ap.add_argument("--no-asr", action="store_true", help="omit the transcript column")
    args = ap.parse_args()

    sessions = build_corpus(config.DATA_DIR)
    if args.sessions:
        sessions = sessions[:args.sessions]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grand_total = 0

    backend = None
    if not args.no_asr:
        try:
            backend = get_backend()
        except Exception as exc:
            print(f'warning: no ASR ({exc})', file=sys.stderr)

    for session in sessions:
        sid = session.meta.session_id
        offset = min(args.offset_min * 60, max(0.0, session.duration_sec - args.minutes * 60))

        with tempfile.TemporaryDirectory() as tmp:
            window = slice_audio(session.probe.path, offset, args.minutes * 60,
                                 Path(tmp) / "window.wav")
            segments = assign_roles(build_timeline(load_waveform(window)))

            transcript = None
            if backend is not None:
                try:
                    transcript = backend.transcribe(window, language=config.ASR_LANGUAGE)
                except (TranscriptionError, FileNotFoundError) as exc:
                    print(f'  {sid}: ASR failed ({exc})', file=sys.stderr)
            rows = build_sheet(sid, segments, transcript=transcript)

            clip_dir = out_dir / sid
            clip_dir.mkdir(parents=True, exist_ok=True)
            for row in rows:
                slice_audio(window, row.start, row.duration,
                            clip_dir / f"{row.row_id.split('#')[-1]}.wav")

        (out_dir / f"{sid}.md").write_text(render_sheet(rows), encoding="utf-8")
        speech_min = sum(r.duration for r in rows) / 60
        print(f"  {sid:<32} {len(rows):>3} rows  {speech_min:>5.1f} min speech  "
              f"(offset {offset/60:.0f} min)")
        grand_total += len(rows)

    print(f"\n{grand_total} rows across {len(sessions)} sessions -> {out_dir}")
    print("Fill in the Truth column, then: python scripts/score_labels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
