"""Score ASR backends against a public Hindi benchmark.

    python -m src.evaluation.run_benchmark --samples 20
    python -m src.evaluation.run_benchmark --contenders indic,groq --samples 40

Downloads N samples of real Hindi audio with human transcripts, runs each contender,
and reports corpus WER/CER. Writes `eval/asr_benchmark.md` and `.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from src import config
from src.asr import get_backend
from src.evaluation.benchmark import BenchmarkSample, normalize_audio, score_backend

DATASETS = {
    # clean read news prose - the standard multilingual benchmark
    "fleurs": ("google/fleurs", "hi_in", "test", "raw_transcription"),
    # AI4Bharat's own Hindi benchmark, read speech from across India
    "kathbath": ("Trelis/vistaar-hi-kathbath-test", None, "train", "transcription"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.run_benchmark",
        description="Measure real WER against a public Hindi dataset.",
    )
    p.add_argument("--dataset", default="fleurs", choices=sorted(DATASETS))
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--contenders", default="indic,primary,groq,baseline")
    p.add_argument("--out-dir", default=str(config.EVAL_DIR))
    return p


def load_samples(name: str, n: int, workdir: Path) -> list[BenchmarkSample]:
    """Pull N samples, writing the audio to disk so any backend can read it."""
    from datasets import Audio, load_dataset

    repo, subset, split, text_field = DATASETS[name]
    args = (repo, subset) if subset else (repo,)
    ds = load_dataset(*args, split=split, streaming=True).cast_column("audio", Audio(decode=False))

    workdir.mkdir(parents=True, exist_ok=True)
    samples: list[BenchmarkSample] = []
    for i, ex in enumerate(ds):
        if len(samples) >= n:
            break
        audio = ex.get("audio") or {}
        raw = audio.get("bytes")
        reference = (ex.get(text_field) or ex.get("transcription") or "").strip()
        if not raw or not reference:
            continue

        # Sample ids double as filenames, and the backend cache keys off content anyway.
        try:
            path = normalize_audio(raw, workdir / f"{name}_{i:04d}.wav")
        except ValueError as exc:
            print(f"  skipping sample {i}: {exc}", file=sys.stderr)
            continue
        samples.append(BenchmarkSample(sample_id=path.stem, audio_path=path,
                                       reference=reference))
    return samples


def build_contenders(names: list[str]) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for name in [n.strip() for n in names if n.strip()]:
        try:
            if name == "indic":
                out.append((name, get_backend("indic")))
            elif name == "groq":
                if not os.environ.get("GROQ_API_KEY"):
                    print("warning: skipping 'groq' — GROQ_API_KEY not set", file=sys.stderr)
                    continue
                out.append((name, get_backend("groq")))
            elif name in config.ASR_MODELS:
                out.append((name, get_backend("local", model=config.ASR_MODELS[name])))
            else:
                raise ValueError(f"unknown contender {name!r}")
        except Exception as exc:                       # noqa: BLE001
            print(f"warning: skipping {name!r} — {exc}", file=sys.stderr)
    if not out:
        raise ValueError("no usable contenders")
    return out


def render(results: list, dataset: str, n: int) -> str:
    lines = [
        "# ASR benchmark — measured WER",
        "",
        f"`{DATASETS[dataset][0]}` · {n} samples · reference transcripts written by humans.",
        "",
        "> **This is clean, read, single-speaker prose on good microphones.** It measures "
        "whether a model can transcribe Hindi at all. It does **not** predict performance on "
        "our recordings — 8–27 children near one phone in a classroom is a different problem. "
        "Use it to sanity-check the models and this harness, not to pick a winner for the "
        "actual corpus.",
        "",
        "| # | Contender | Model | Scored | **WER** | CER | Failed | Sec/sample |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(results, key=lambda r: (r.wer is None, r.wer if r.wer is not None else 9))
    for i, r in enumerate(ranked, start=1):
        wer = "—" if r.wer is None else f"**{r.wer:.1%}**"
        cer = "—" if r.cer is None else f"{r.cer:.1%}"
        per = r.elapsed_sec / max(1, r.scored)
        lines.append(f"| {i} | **{r.backend}** | `{r.model}` | {r.scored} | {wer} | {cer} "
                     f"| {r.failed} | {per:.1f} |")

    lines += ["", "## Sample outputs", ""]
    for r in ranked:
        if not r.rows:
            continue
        row = min(r.rows, key=lambda x: x.wer)
        lines += [f"### {r.backend} — best sample (WER {row.wer:.1%})", "",
                  f"- **reference:** {row.reference[:200]}",
                  f"- **hypothesis:** {row.hypothesis[:200]}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)

    try:
        contenders = build_contenders(args.contenders.split(","))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"loading {args.samples} samples from {DATASETS[args.dataset][0]} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        samples = load_samples(args.dataset, args.samples, Path(tmp))
        if not samples:
            print("error: no samples loaded", file=sys.stderr)
            return 1
        print(f"{len(samples)} samples · {len(contenders)} contenders\n", flush=True)

        results = []
        for label, backend in contenders:
            print(f"  running {label} ({getattr(backend, 'model_id', '?')}) ...", flush=True)
            r = score_backend(backend, samples)
            r.backend = label
            results.append(r)
            wer = "—" if r.wer is None else f"{r.wer:.1%}"
            print(f"    WER {wer}   scored {r.scored}   failed {r.failed}   "
                  f"{r.elapsed_sec:.0f}s\n", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "asr_benchmark.md").write_text(
        render(results, args.dataset, len(samples)), encoding="utf-8")
    (out_dir / "asr_benchmark.json").write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("=" * 60)
    for r in sorted(results, key=lambda r: (r.wer is None, r.wer if r.wer is not None else 9)):
        wer = "—" if r.wer is None else f"{r.wer:6.1%}"
        print(f"  {r.backend:<12} WER {wer}   ({r.scored} scored, {r.failed} failed)")
    print(f"\nwrote {out_dir / 'asr_benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
