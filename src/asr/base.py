"""The shared ASR contract.

Every backend returns a `Transcript`. Nothing downstream — metrics, confidence scoring,
the dashboard — is allowed to know which model produced it. That is what lets the backend
be swapped with a single line in `config.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TranscriptionError(RuntimeError):
    """A backend could not produce a transcript."""


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    word: str

    def offset(self, seconds: float) -> "Word":
        return replace(self, start=self.start + seconds, end=self.end + seconds)

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "word": self.word}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Word":
        return cls(start=float(raw["start"]), end=float(raw["end"]), word=raw["word"])


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[Word] | None = None

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"segment ends before it starts: {self.start} -> {self.end}")
        object.__setattr__(self, "text", self.text.strip())

    @property
    def duration(self) -> float:
        return self.end - self.start

    def offset(self, seconds: float) -> "TranscriptSegment":
        """Shift into whole-recording time. Returns a new segment."""
        return replace(
            self,
            start=self.start + seconds,
            end=self.end + seconds,
            words=[w.offset(seconds) for w in self.words] if self.words else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "words": [w.to_dict() for w in self.words] if self.words else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TranscriptSegment":
        words = raw.get("words")
        return cls(
            start=float(raw["start"]),
            end=float(raw["end"]),
            text=raw.get("text", ""),
            avg_logprob=raw.get("avg_logprob"),
            no_speech_prob=raw.get("no_speech_prob"),
            words=[Word.from_dict(w) for w in words] if words else None,
        )


@dataclass(frozen=True)
class Transcript:
    segments: list[TranscriptSegment]
    language: str
    language_prob: float | None
    backend: str
    model: str
    audio_sec: float

    def __post_init__(self) -> None:
        if self.audio_sec <= 0:
            raise ValueError("audio_sec must be positive")
        object.__setattr__(self, "segments", sorted(self.segments, key=lambda s: s.start))

    # ---------------------------------------------------------------- derived
    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()

    @property
    def speech_sec(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def speech_density(self) -> float:
        """Fraction of the recording that carried transcribed speech, capped at 1."""
        return min(1.0, self.speech_sec / self.audio_sec)

    @property
    def mean_logprob(self) -> float | None:
        """Duration-weighted average confidence, or None if no backend reported any.

        Weighted because a long garbled stretch should count for more than a short
        clean one — this feeds the M6 confidence gate.
        """
        scored = [s for s in self.segments if s.avg_logprob is not None and s.duration > 0]
        if not scored:
            return None
        total = sum(s.duration for s in scored)
        return sum(s.avg_logprob * s.duration for s in scored) / total

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "language_prob": self.language_prob,
            "backend": self.backend,
            "model": self.model,
            "audio_sec": round(self.audio_sec, 3),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Transcript":
        return cls(
            segments=[TranscriptSegment.from_dict(s) for s in raw.get("segments", [])],
            language=raw["language"],
            language_prob=raw.get("language_prob"),
            backend=raw["backend"],
            model=raw["model"],
            audio_sec=float(raw["audio_sec"]),
        )


@runtime_checkable
class ASRBackend(Protocol):
    """What every backend must offer."""

    name: str
    model_id: str
    max_upload_bytes: int | None

    def transcribe(self, path: Path, language: str | None = None,
                   use_cache: bool = True) -> Transcript: ...
