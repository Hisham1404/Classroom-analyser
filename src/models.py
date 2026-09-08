"""Typed structures for one recorded session.

Two rules are enforced here rather than left to convention:

1. The session JSON's `duration` is untrusted. `Session.duration_sec` always returns the
   measured audio length, because three of the five real recordings disagree with it.
2. The teacher's real name never appears in `to_public_dict()` — that is the only shape
   that gets written to `results/` and committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.config import COMPLETENESS_OK

_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- roster
@dataclass(frozen=True)
class Roster:
    """Ground-truth headcount, straight from their app."""

    total: int = 0
    boys: int = 0
    girls: int = 0

    def __post_init__(self) -> None:
        if min(self.total, self.boys, self.girls) < 0:
            raise ValueError("roster counts cannot be negative")

    @property
    def is_consistent(self) -> bool:
        return self.boys + self.girls == self.total


# --------------------------------------------------------------------------- photo
@dataclass(frozen=True)
class PhotoMeta:
    """Metadata attached to the classroom photo. Every field is optional —
    3 of 5 real sessions carry no coordinates at all."""

    teacher_name: str | None = None
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PhotoMeta":
        raw = raw or {}
        return cls(
            teacher_name=raw.get("teacherName"),
            captured_at=_parse_timestamp(raw.get("capturedAt")),
            latitude=raw.get("latitude"),
            longitude=raw.get("longitude"),
            address=raw.get("address"),
        )


# --------------------------------------------------------------------------- metadata
@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    teacher_id: str | None = None
    activity: str | None = None
    roster: Roster = field(default_factory=Roster)
    declared_sec: float | None = None
    recorded_at: datetime | None = None
    photo: PhotoMeta = field(default_factory=PhotoMeta)

    @classmethod
    def from_dict(cls, session_id: str, raw: dict[str, Any]) -> "SessionMeta":
        duration_ms = raw.get("duration")
        return cls(
            session_id=session_id,
            teacher_id=raw.get("teacherId"),
            activity=raw.get("activityType"),
            roster=Roster(
                total=int(raw.get("totalStudents") or 0),
                boys=int(raw.get("boys") or 0),
                girls=int(raw.get("girls") or 0),
            ),
            declared_sec=(duration_ms / 1000.0) if duration_ms else None,
            recorded_at=_parse_timestamp(raw.get("timestamp")),
            photo=PhotoMeta.from_dict(raw.get("photoMetadata")),
        )


# --------------------------------------------------------------------------- audio
@dataclass(frozen=True)
class AudioProbe:
    """What ffprobe actually found in the file."""

    path: Path | None
    actual_sec: float
    sample_rate: int
    channels: int
    bit_rate: int
    size_bytes: int

    def __post_init__(self) -> None:
        if self.actual_sec <= 0:
            raise ValueError("audio duration must be positive")


# --------------------------------------------------------------------------- aliases
def _letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA. Keeps working past 26 teachers."""
    out, n = "", index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


class AliasRegistry:
    """Stable pseudonyms for teacher IDs.

    Assignment is by sorted ID, so the same corpus always produces the same aliases
    regardless of the order sessions were discovered in.
    """

    def __init__(self, teacher_ids: Iterable[str]) -> None:
        ordered = sorted({t for t in teacher_ids if t})
        self._map = {t: f"Teacher {_letters(i)}" for i, t in enumerate(ordered)}

    def alias(self, teacher_id: str) -> str:
        if teacher_id not in self._map:
            raise KeyError(f"unknown teacher id: {teacher_id!r}")
        return self._map[teacher_id]

    def __len__(self) -> int:
        return len(self._map)


# --------------------------------------------------------------------------- session
@dataclass(frozen=True)
class Session:
    meta: SessionMeta
    probe: AudioProbe
    alias: str

    @property
    def duration_sec(self) -> float:
        """The measured length. Never the declared one."""
        return self.probe.actual_sec

    @property
    def completeness(self) -> float | None:
        """Measured length as a fraction of what their app claimed, or None if it claimed nothing."""
        if not self.meta.declared_sec:
            return None
        return self.probe.actual_sec / self.meta.declared_sec

    @property
    def is_truncated(self) -> bool:
        c = self.completeness
        return c is not None and c < COMPLETENESS_OK

    @property
    def flags(self) -> list[str]:
        out: list[str] = []
        if self.is_truncated:
            out.append("audio_truncated")
        if not self.meta.roster.is_consistent:
            out.append("roster_mismatch")
        if not self.meta.photo.has_gps:
            out.append("no_gps")
        return out

    def to_public_dict(self) -> dict[str, Any]:
        """The only shape that may be written to results/ and committed.

        Carries no teacher name and no device file paths.
        """
        c = self.completeness
        return {
            "session_id": self.meta.session_id,
            "teacher": {"id": self.meta.teacher_id, "alias": self.alias},
            "activity": self.meta.activity,
            "recorded_at": self.meta.recorded_at.isoformat() if self.meta.recorded_at else None,
            "roster": {
                "total": self.meta.roster.total,
                "boys": self.meta.roster.boys,
                "girls": self.meta.roster.girls,
            },
            "audio": {
                "declared_sec": round(self.meta.declared_sec, 1) if self.meta.declared_sec else None,
                "actual_sec": round(self.probe.actual_sec, 1),
                "completeness": round(c, 3) if c is not None else None,
                "sample_rate": self.probe.sample_rate,
                "channels": self.probe.channels,
                "has_gps": self.meta.photo.has_gps,
            },
            "flags": self.flags,
        }
