"""Run every contender over the same clips and rank them.

One contender failing must not lose the whole run. These passes are expensive — a full
local model sweep is hours — so failures are collected and reported, never raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import config
from src.asr.base import Transcript
from src.evaluation.clips import ClipSpec, extract_clip
from src.evaluation.signals import QualitySignals, cross_model_agreement, quality_signals


@dataclass
class Contender:
    """A named entry in the comparison — usually a backend plus a specific model."""
    label: str
    backend: Any                      # anything satisfying ASRBackend

    @property
    def model_id(self) -> str:
        return getattr(self.backend, "model_id", "?")

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "?")


@dataclass
class BakeoffRow:
    clip: ClipSpec
    label: str
    backend_name: str
    model_id: str
    transcript: Transcript
    signals: QualitySignals
    elapsed_sec: float
    agreement: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip.clip_id,
            "label": self.label,
            "backend": self.backend_name,
            "model": self.model_id,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "text": self.transcript.text,
            "language": self.transcript.language,
            "agreement": None if self.agreement is None else round(self.agreement, 4),
            "signals": self.signals.to_dict(),
        }


@dataclass
class BakeoffResult:
    rows: list[BakeoffRow] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows], "failures": list(self.failures)}


@dataclass
class ContenderScore:
    label: str
    backend_name: str
    model_id: str
    clips_scored: int
    composite: float
    agreement: float | None
    mean_logprob: float | None
    repetition_rate: float
    devanagari_ratio: float
    mean_elapsed_sec: float


def run_bakeoff(sources: Mapping[Path, Iterable[ClipSpec]],
                contenders: list[Contender],
                workdir: Path,
                language: str | None = None) -> BakeoffResult:
    """Transcribe every clip with every contender.

    `sources` maps a source audio file to the clips to cut from it. Clips are extracted
    once and reused across contenders, so each model is judged on byte-identical audio.
    """
    if not contenders:
        raise ValueError("no contenders to compare")

    language = language or config.ASR_LANGUAGE
    result = BakeoffResult()

    for source, specs in sources.items():
        for spec in specs:
            clip_path = extract_clip(Path(source), spec, Path(workdir))

            for contender in contenders:
                started = time.perf_counter()
                try:
                    transcript = contender.backend.transcribe(clip_path, language=language)
                except Exception as exc:                       # noqa: BLE001
                    result.failures.append(
                        f"{contender.label} failed on {spec.clip_id}: {exc}")
                    continue

                result.rows.append(BakeoffRow(
                    clip=spec,
                    label=contender.label,
                    backend_name=contender.backend_name,
                    model_id=contender.model_id,
                    transcript=transcript,
                    signals=quality_signals(transcript),
                    elapsed_sec=time.perf_counter() - started,
                ))

    _annotate_agreement(result)
    return result


def _annotate_agreement(result: BakeoffResult) -> None:
    """Score each row against the other contenders on the same clip.

    Two models that share no weights producing the same words is evidence the words are
    real. A model that agrees with nobody is the one inventing - which is how
    Devanagari-shaped noise gets caught when every other signal says it is fine.
    """
    by_clip: dict[str, list[BakeoffRow]] = {}
    for row in result.rows:
        by_clip.setdefault(row.clip.clip_id, []).append(row)

    for rows in by_clip.values():
        if len(rows) < 2:
            continue
        for row in rows:
            # Score THIS row against each other contender separately. Passing the whole
            # set to cross_model_agreement returns the same all-pairs mean for every row,
            # which hides exactly the odd-one-out this is meant to expose.
            pairwise = [
                cross_model_agreement([row.transcript.text, other.transcript.text])
                for other in rows if other is not row
            ]
            row.agreement = sum(pairwise) / len(pairwise) if pairwise else None


def rank_contenders(result: BakeoffResult) -> list[ContenderScore]:
    """Average each contender's signals across its clips, best composite first."""
    grouped: dict[str, list[BakeoffRow]] = {}
    for row in result.rows:
        grouped.setdefault(row.label, []).append(row)

    scores: list[ContenderScore] = []
    for label, rows in grouped.items():
        logprobs = [r.signals.mean_logprob for r in rows if r.signals.mean_logprob is not None]
        scores.append(ContenderScore(
            label=label,
            backend_name=rows[0].backend_name,
            model_id=rows[0].model_id,
            clips_scored=len(rows),
            composite=sum(r.signals.composite for r in rows) / len(rows),
            agreement=(sum(a for r in rows if (a := r.agreement) is not None)
                       / len([r for r in rows if r.agreement is not None]))
                      if any(r.agreement is not None for r in rows) else None,
            mean_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
            repetition_rate=sum(r.signals.repetition_rate for r in rows) / len(rows),
            devanagari_ratio=sum(r.signals.devanagari_ratio for r in rows) / len(rows),
            mean_elapsed_sec=sum(r.elapsed_sec for r in rows) / len(rows),
        ))

    return sorted(scores, key=lambda s: -s.composite)
