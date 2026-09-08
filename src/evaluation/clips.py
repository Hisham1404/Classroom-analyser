"""Picking and cutting the audio slices every contender is judged on.

Selection is deterministic and identical across models — otherwise the comparison
measures which clips a model happened to get, not how good it is.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClipSpec:
    session_id: str
    index: int
    start: float
    duration: float

    @property
    def clip_id(self) -> str:
        """Stable across re-selection — depends on position in the run, not on timing."""
        return f"{self.session_id}_c{self.index:02d}"

    @property
    def end(self) -> float:
        return self.start + self.duration


def select_clips(total_sec: float, n_clips: int, clip_sec: float,
                 margin: float = 0.05) -> list[ClipSpec]:
    """Evenly spaced, non-overlapping clips from the middle of a recording.

    `margin` trims a fraction off each end, where the phone is usually being picked up
    or put down rather than recording a lesson. If the usable span cannot hold
    `n_clips` without overlap, fewer are returned — overlapping clips would double-count
    the same audio in the averages.
    """
    if total_sec <= 0:
        raise ValueError("total_sec must be positive")
    if n_clips <= 0:
        raise ValueError("n_clips must be positive")
    if clip_sec <= 0:
        raise ValueError("clip_sec must be positive")

    if total_sec <= clip_sec:
        return [ClipSpec("", 0, 0.0, total_sec)]

    usable_start = total_sec * margin
    usable_end = total_sec * (1.0 - margin)
    span = usable_end - usable_start

    fits = int(span // clip_sec)
    count = max(1, min(n_clips, fits))

    # Spread the leftover room evenly between and around the clips.
    slack = span - count * clip_sec
    gap = slack / (count + 1)

    clips: list[ClipSpec] = []
    for i in range(count):
        start = usable_start + gap * (i + 1) + clip_sec * i
        clips.append(ClipSpec("", i, round(start, 3), clip_sec))
    return clips


def clips_for_session(session_id: str, total_sec: float, n_clips: int,
                      clip_sec: float, margin: float = 0.05) -> list[ClipSpec]:
    """`select_clips` with the session id filled in."""
    from dataclasses import replace
    return [replace(c, session_id=session_id)
            for c in select_clips(total_sec, n_clips, clip_sec, margin)]


def extract_clip(source: Path, spec: ClipSpec, workdir: Path) -> Path:
    """Cut one clip to 16 kHz mono wav.

    Every contender is handed the same prepared file, so no model gains or loses from
    resampling differences. Re-extracting an existing clip is a no-op.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"source audio not found: {source}")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / f"{spec.clip_id or source.stem}_{spec.index:02d}.wav"

    if dst.is_file() and dst.stat().st_size > 0:
        return dst

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-ss", f"{spec.start:.3f}", "-t", f"{spec.duration:.3f}", "-i", str(source),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )
    return dst
