"""Stage 02 — turn a waveform into a timeline of what kind of sound it was.

This is the robust half of the pipeline. It never sees a word, so it keeps working when
the transcript is garbage — which on this audio it sometimes is. M1–M4 are all built here.

One distinction is doing real work: non-speech that is **loud** is children building
things, not dead air. A generic tool would count the Shadow Art session's 80% non-speech
as failure when it is probably the class doing exactly what a maker lesson is for.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np

from src import config

SAMPLE_RATE = 16000
_EPS = 1e-12


class SegmentKind(str, Enum):
    SPEECH = "speech"
    HANDSON = "handson"      # loud non-speech: children working
    DEAD = "dead"            # quiet non-speech


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    kind: SegmentKind
    rms_db: float
    flatness: float
    role: str | None = None
    role_conf: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def with_role(self, role: str | None, confidence: float | None) -> "Segment":
        return replace(self, role=role, role_conf=confidence)

    def to_dict(self) -> dict:
        out = {
            "t0": round(self.start, 3),
            "t1": round(self.end, 3),
            "kind": self.kind.value,
            "rms_db": round(self.rms_db, 2),
        }
        if self.role:
            out |= {"role": self.role, "role_conf": round(self.role_conf or 0.0, 3)}
        return out


# --------------------------------------------------------------------------- io
def load_waveform(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr`."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# --------------------------------------------------------------------------- features
def rms_db(wave: np.ndarray) -> float:
    """Loudness in dBFS. Silence returns a very negative number, never -inf."""
    if wave.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(wave.astype(np.float64)))))
    return 20.0 * float(np.log10(max(rms, _EPS)))


def spectral_flatness(wave: np.ndarray) -> float:
    """Geometric mean over arithmetic mean of the power spectrum, 0-1.

    Near-field speech is tonal and scores low; far-field babble and room noise are closer
    to white and score high. It is the cheapest proxy we have for "how close was the mic".
    """
    if wave.size < 32:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wave.astype(np.float64) * np.hanning(wave.size))) ** 2
    spectrum = spectrum[1:]                       # drop DC
    if spectrum.size == 0 or not np.any(spectrum > 0):
        return 0.0
    spectrum = np.maximum(spectrum, _EPS)
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    return float(np.clip(geometric / max(arithmetic, _EPS), 0.0, 1.0))


def classify_non_speech(segment_rms_db: float) -> SegmentKind:
    """Loud non-speech is children working; quiet non-speech is dead air."""
    return (SegmentKind.HANDSON if segment_rms_db >= config.HANDSON_RMS_DB
            else SegmentKind.DEAD)


# --------------------------------------------------------------------------- timeline
def detect_speech(wave: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Speech spans in seconds, via Silero VAD.

    Uses faster-whisper's bundled Silero at the threshold validated in
    `scripts/tune_vad.py`. onnx-asr's wrapper of the same model returned zero segments on
    2 of 3 real clips; this one does not.
    """
    if wave.size == 0:
        return []

    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=config.VAD_THRESHOLD,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
    )
    return [(s["start"] / sr, s["end"] / sr) for s in get_speech_timestamps(wave, options)]


def _normalise_spans(spans, total: float) -> list[tuple[float, float]]:
    """Sort, clip to the audio and merge overlaps, so the tiling below holds."""
    clipped = sorted((max(0.0, float(a)), min(total, float(b))) for a, b in spans)
    merged: list[tuple[float, float]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_timeline(wave: np.ndarray, sr: int = SAMPLE_RATE,
                   speech_spans: list[tuple[float, float]] | None = None) -> list[Segment]:
    """Every moment of the recording, labelled speech / hands-on / dead.

    The segments tile the whole file with no gaps and no overlaps, so durations can be
    summed straight into the metrics without worrying about what fell between them.

    `speech_spans` overrides the VAD. Where diarization has run, its turns are a far
    better speech map than Silero on this audio - Silero was leaving most of the speech
    out and `classify_non_speech` was then filing it as hands-on activity. Passing an
    empty list means "diarization heard no speech", which is an answer; only `None`
    means "no opinion, run the VAD".
    """
    if wave.size == 0:
        return []

    total = wave.size / sr
    spans = (detect_speech(wave, sr) if speech_spans is None
             else _normalise_spans(speech_spans, total))

    # Interleave speech spans with the gaps between them.
    boundaries: list[tuple[float, float, bool]] = []
    cursor = 0.0
    for start, end in spans:
        start, end = max(0.0, start), min(total, end)
        if start > cursor:
            boundaries.append((cursor, start, False))
        if end > start:
            boundaries.append((start, end, True))
        cursor = max(cursor, end)
    if cursor < total:
        boundaries.append((cursor, total, False))

    segments: list[Segment] = []
    for start, end, is_speech in boundaries:
        if end - start <= 0:
            continue
        chunk = wave[int(start * sr):int(end * sr)]
        loudness = rms_db(chunk)
        segments.append(Segment(
            start=start,
            end=end,
            kind=SegmentKind.SPEECH if is_speech else classify_non_speech(loudness),
            rms_db=loudness,
            flatness=spectral_flatness(chunk),
        ))
    return segments


def timeline_for_file(path: Path, sr: int = SAMPLE_RATE) -> list[Segment]:
    return build_timeline(load_waveform(path, sr), sr)
