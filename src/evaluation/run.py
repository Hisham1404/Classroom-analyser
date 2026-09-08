"""Run the ASR bake-off end to end.

    python -m src.evaluation.run                          # every configured model
    python -m src.evaluation.run --contenders baseline,primary --sessions 2 --clip-sec 30

Writes three files into `eval/`:
  asr_bakeoff.md        the readable comparison
  asr_handscoring.md    the blank sheet to fill in while listening
  asr_bakeoff.json      the raw rows, so a report can be re-rendered without re-running
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src import config
from src.asr import get_backend
from src.evaluation.bakeoff import Contender, rank_contenders, run_bakeoff
from src.evaluation.clips import clips_for_session
from src.evaluation.report import hand_scoring_sheet, render_report
from src.ingest import build_corpus

DEFAULT_CONTENDERS = ["baseline", "primary", "indic", "groq"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.run",
        description="Compare ASR models on identical clips from the real recordings.",
    )
    p.add_argument("--contenders", default=",".join(DEFAULT_CONTENDERS),
                   help="comma-separated: " + ", ".join(sorted(config.ASR_MODELS)) + ", indic, groq")
    p.add_argument("--clips", type=int, default=3, help="clips per session (default 3)")
    p.add_argument("--clip-sec", type=float, default=180.0, help="clip length (default 180)")
    p.add_argument("--sessions", type=int, default=None,
                   help="only use the first N sessions (default: all)")
    p.add_argument("--out-dir", default=str(config.EVAL_DIR), help="where to write the report")
    return p


def build_contenders(names: list[str]) -> list[Contender]:
    """Turn contender names into backends, skipping any that cannot be built."""
    contenders: list[Contender] = []

    for name in names:
        name = name.strip()
        if not name:
            continue

        if name == "indic":
            contenders.append(Contender("indic", get_backend("indic")))
            continue

        if name == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                print("warning: skipping 'groq' — GROQ_API_KEY is not set", file=sys.stderr)
                continue
            contenders.append(Contender("groq", get_backend("groq")))
            continue

        if name not in config.ASR_MODELS:
            options = ", ".join(sorted(config.ASR_MODELS)) + ", indic, groq"
            raise ValueError(f"unknown contender {name!r}; available: {options}")

        contenders.append(Contender(name, get_backend("local", model=config.ASR_MODELS[name])))

    if not contenders:
        raise ValueError("no usable contenders")
    return contenders


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)

    try:
        sessions = build_corpus(config.DATA_DIR)
    except FileNotFoundError:
        sessions = []
    if not sessions:
        print(f"error: no sessions found under {config.DATA_DIR}", file=sys.stderr)
        return 1
    if args.sessions:
        sessions = sessions[:args.sessions]

    try:
        contenders = build_contenders(args.contenders.split(","))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sources = {}
    for session in sessions:
        specs = clips_for_session(session.meta.session_id, session.duration_sec,
                                  args.clips, args.clip_sec)
        sources[session.probe.path] = specs

    total_clips = sum(len(v) for v in sources.values())
    print(f"{len(sessions)} sessions -> {total_clips} clips x {len(contenders)} contenders "
          f"= {total_clips * len(contenders)} transcriptions", flush=True)
    for c in contenders:
        print(f"  - {c.label:<12} {c.backend_name:<6} {c.model_id}", flush=True)

    result = run_bakeoff(sources, contenders, workdir=out_dir / "clips")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "asr_bakeoff.md").write_text(render_report(result), encoding="utf-8")
    (out_dir / "asr_handscoring.md").write_text(hand_scoring_sheet(result), encoding="utf-8")
    (out_dir / "asr_bakeoff.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    for position, s in enumerate(rank_contenders(result), start=1):
        print(f"  {position}. {s.label:<12} composite {s.composite:.3f}  "
              f"logprob {s.mean_logprob if s.mean_logprob is None else round(s.mean_logprob, 2)}  "
              f"looping {s.repetition_rate:.2f}  devanagari {s.devanagari_ratio:.2f}  "
              f"{s.mean_elapsed_sec:.0f}s/clip")
    if result.failures:
        print(f"\n{len(result.failures)} failure(s) — see the report")
    print(f"\nwrote {out_dir / 'asr_bakeoff.md'}")
    print(f"      {out_dir / 'asr_handscoring.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
