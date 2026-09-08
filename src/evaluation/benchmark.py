"""Score ASR backends against a reference dataset — real WER, not proxies.

Every other measurement in `eval/` compares models against each other, because our own
recordings have no ground truth. This is the one place we get an actual error rate.

**Read the caveat before quoting any number from here.** FLEURS is clean, read,
single-speaker news prose recorded on good microphones. Winning it does not demonstrate
anything about 27 children shouting near one phone in Igatpuri. Treat it as a sanity check
on the models and on this harness — not as a prediction for our audio.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.scoring import _levenshtein, normalize_for_scoring


def normalize_audio(raw: bytes, dst: Path) -> Path:
    """Decode arbitrary audio bytes to 16 kHz mono wav.

    Two reasons this is not optional. Backends disagree about which wav encodings they
    accept — onnx-asr rejects FLEURS's files with "unknown format: 3" — and normalising
    means every contender is scored on byte-identical audio rather than on whose decoder
    is more forgiving.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise ValueError(f"could not decode audio into {dst.name}")
    return dst


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    audio_path: Path
    reference: str


@dataclass(frozen=True)
class BenchmarkRow:
    sample_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


def _corpus_rate(pairs: Iterable[tuple[str, str]], *, characters: bool) -> float:
    """Total edits over total reference units.

    This is the standard definition and it is *not* the mean of per-sample rates:
    averaging rates lets a one-word sample weigh as much as a fifty-word one.
    """
    edits = units = 0
    seen = False
    for reference, hypothesis in pairs:
        seen = True
        ref = normalize_for_scoring(reference)
        hyp = normalize_for_scoring(hypothesis)
        if characters:
            ref_units, hyp_units = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
        else:
            ref_units, hyp_units = ref.split(), hyp.split()
        if not ref_units:
            continue
        edits += _levenshtein(ref_units, hyp_units)
        units += len(ref_units)

    if not seen:
        raise ValueError("no samples to score")
    if units == 0:
        raise ValueError("every reference was empty after normalisation")
    return edits / units


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> float:
    return _corpus_rate(pairs, characters=False)


def corpus_cer(pairs: Iterable[tuple[str, str]]) -> float:
    return _corpus_rate(pairs, characters=True)


@dataclass
class BenchmarkResult:
    backend: str
    model: str
    rows: list[BenchmarkRow] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def scored(self) -> int:
        return len(self.rows)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def wer(self) -> float | None:
        """Corpus WER, or None if nothing scored.

        None rather than 0.0 on purpose: a backend that crashed on every sample must not
        read as a perfect transcriber.
        """
        if not self.rows:
            return None
        return corpus_wer([(r.reference, r.hypothesis) for r in self.rows])

    @property
    def cer(self) -> float | None:
        if not self.rows:
            return None
        return corpus_cer([(r.reference, r.hypothesis) for r in self.rows])

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "scored": self.scored,
            "failed": self.failed,
            "wer": None if self.wer is None else round(self.wer, 4),
            "cer": None if self.cer is None else round(self.cer, 4),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "rows": [r.to_dict() for r in self.rows],
            "failures": list(self.failures),
        }


def score_backend(backend, samples: list[BenchmarkSample],
                  language: str | None = "hi") -> BenchmarkResult:
    """Transcribe every sample and score it against its reference."""
    if not samples:
        raise ValueError("no samples to score")

    result = BenchmarkResult(
        backend=getattr(backend, "name", "?"),
        model=getattr(backend, "model_id", "?"),
    )
    started = time.perf_counter()

    for s in samples:
        t0 = time.perf_counter()
        try:
            transcript = backend.transcribe(s.audio_path, language=language)
        except Exception as exc:                       # noqa: BLE001
            result.failures.append(f"{s.sample_id}: {exc}")
            continue

        hypothesis = transcript.text
        # An empty transcript is fully wrong, not skipped — silence is an answer.
        result.rows.append(BenchmarkRow(
            sample_id=s.sample_id,
            reference=s.reference,
            hypothesis=hypothesis,
            wer=corpus_wer([(s.reference, hypothesis)]),
            cer=corpus_cer([(s.reference, hypothesis)]),
            elapsed_sec=time.perf_counter() - t0,
        ))

    result.elapsed_sec = time.perf_counter() - started
    return result
