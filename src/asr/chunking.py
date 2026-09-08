"""Splitting long audio and stitching the pieces back together.

Only the API backend needs this — the hosted endpoint caps uploads at 25 MB and four of
the five real recordings are larger. faster-whisper streams locally and needs none of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.asr.base import Transcript, TranscriptSegment


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def plan_chunks(total_sec: float, chunk_sec: float, overlap_sec: float = 0.0,
                min_tail_sec: float | None = None) -> list[Chunk]:
    """Cut `total_sec` into pieces of at most `chunk_sec`.

    `overlap_sec` pulls each chunk's start earlier so a sentence straddling a boundary
    is captured whole by at least one chunk. The duplicate text it creates is removed
    again in `merge_chunk_transcripts`.

    A trailing remainder shorter than `min_tail_sec` is absorbed into the previous chunk
    rather than becoming its own piece — a sliver of audio still costs a whole request
    against a metered quota, and yields nothing.
    """
    if total_sec <= 0:
        raise ValueError("total_sec must be positive")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if overlap_sec < 0:
        raise ValueError("overlap_sec cannot be negative")
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    min_tail = chunk_sec * 0.05 if min_tail_sec is None else min_tail_sec

    chunks: list[Chunk] = []
    cursor = 0.0
    index = 0
    while cursor < total_sec - 1e-9:
        start = max(0.0, cursor - overlap_sec) if index else 0.0
        end = min(cursor + chunk_sec, total_sec)

        if chunks and (end - cursor) < min_tail:
            prev = chunks[-1]
            chunks[-1] = Chunk(index=prev.index, start=prev.start, duration=end - prev.start)
            break

        chunks.append(Chunk(index=index, start=start, duration=end - start))
        cursor = end
        index += 1
    return chunks


def _is_duplicate(a: TranscriptSegment, b: TranscriptSegment, tolerance: float = 1.0) -> bool:
    """Same words at roughly the same moment — an artefact of the overlap."""
    return a.text == b.text and abs(a.start - b.start) <= tolerance


def merge_chunk_transcripts(parts: list[Transcript], chunks: list[Chunk],
                            audio_sec: float) -> Transcript:
    """Shift each chunk's segments into whole-recording time and concatenate."""
    if len(parts) != len(chunks):
        raise ValueError(f"got {len(parts)} transcripts for {len(chunks)} chunks")
    if not parts:
        raise ValueError("nothing to merge")

    merged: list[TranscriptSegment] = []
    for part, chunk in zip(parts, chunks):
        for seg in part.segments:
            shifted = seg.offset(chunk.start)
            if merged and _is_duplicate(merged[-1], shifted):
                continue
            merged.append(shifted)

    best = max(parts, key=lambda p: p.language_prob if p.language_prob is not None else -1.0)
    return Transcript(
        segments=merged,
        language=best.language,
        language_prob=best.language_prob,
        backend=parts[0].backend,
        model=parts[0].model,
        audio_sec=audio_sec,
    )
