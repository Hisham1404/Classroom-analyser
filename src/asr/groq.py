"""Hosted backend — Groq's Whisper endpoint.

Fast and free, but metered (28.8K audio-seconds/day) and capped at 25 MB per upload,
so long recordings are split, sent separately and stitched back together.

The API key is read from the `GROQ_API_KEY` environment variable and never written into
a transcript, a cache entry or a results file.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from src import config
from src.asr.base import Transcript, TranscriptSegment, TranscriptionError, Word
from src.asr.cache import TranscriptCache
from src.asr.chunking import merge_chunk_transcripts, plan_chunks

ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

# Groq answers with an English language name; the pipeline speaks ISO codes.
_LANGUAGE_CODES = {
    "hindi": "hi", "marathi": "mr", "english": "en", "urdu": "ur",
    "bengali": "bn", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "gujarati": "gu", "punjabi": "pa", "odia": "or",
}


class GroqBackend:
    name = "groq"

    def __init__(self, model: str | None = None, *, cache: TranscriptCache | None = None,
                 chunk_sec: float | None = None, overlap_sec: float | None = None,
                 timeout: int = 300) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it, or switch config.ASR_BACKEND to 'local'."
            )
        self.model_id = model or config.GROQ_MODEL
        self.cache = cache if cache is not None else TranscriptCache(config.ASR_CACHE_DIR)
        self.chunk_sec = chunk_sec if chunk_sec is not None else config.GROQ_CHUNK_SEC
        # plan_chunks is strict about overlap < chunk; clamp here so a caller passing a
        # short chunk_sec gets sensible behaviour instead of a crash.
        requested = overlap_sec if overlap_sec is not None else config.GROQ_OVERLAP_SEC
        self.overlap_sec = max(0.0, min(requested, self.chunk_sec / 4))
        self.timeout = timeout
        self.max_upload_bytes = config.GROQ_MAX_UPLOAD_BYTES

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _form_fields(self, language: str | None) -> dict[str, str]:
        fields = {"model": self.model_id, "response_format": "verbose_json"}
        if language:
            fields["language"] = language
        return fields

    def _post_audio(self, path: Path, language: str | None = None) -> dict:
        import requests

        with Path(path).open("rb") as fh:
            resp = requests.post(
                ENDPOINT,
                headers=self._headers(),
                data=self._form_fields(language) | {"timestamp_granularities[]": "segment"},
                files={"file": (Path(path).name, fh, "application/octet-stream")},
                timeout=self.timeout,
            )

        if resp.status_code == 429:
            retry = resp.headers.get("retry-after", "?")
            raise TranscriptionError(f"groq rate limit (429), retry after {retry}s")
        if resp.status_code != 200:
            raise TranscriptionError(f"groq returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    # ------------------------------------------------------------------ main
    def transcribe(self, path: Path, language: str | None = None,
                   use_cache: bool = True) -> Transcript:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")

        if use_cache:
            hit = self.cache.get(path, self.name, self.model_id, language)
            if hit is not None:
                return hit

        if self.max_upload_bytes and path.stat().st_size > self.max_upload_bytes:
            transcript = self._transcribe_chunked(path, language)
        else:
            transcript = self._to_transcript(self._post_audio(path, language),
                                             audio_sec=_probe_duration(path))

        if use_cache:
            self.cache.put(path, self.name, self.model_id, language, transcript)
        return transcript

    def _transcribe_chunked(self, path: Path, language: str | None) -> Transcript:
        total = _probe_duration(path)
        chunks = plan_chunks(total, self.chunk_sec, self.overlap_sec)

        parts: list[Transcript] = []
        with tempfile.TemporaryDirectory() as tmp:
            for chunk in chunks:
                piece = Path(tmp) / f"{path.stem}_{chunk.index:03d}.flac"
                _extract(path, chunk.start, chunk.duration, piece)
                parts.append(self._to_transcript(self._post_audio(piece, language),
                                                 audio_sec=chunk.duration))
        return merge_chunk_transcripts(parts, chunks, audio_sec=total)

    def _to_transcript(self, payload: dict, audio_sec: float) -> Transcript:
        segments = [self._to_segment(s) for s in payload.get("segments", [])]
        raw_language = str(payload.get("language", "")).lower()
        duration = float(payload.get("duration") or 0.0) or audio_sec

        return Transcript(
            segments=segments,
            language=_LANGUAGE_CODES.get(raw_language, raw_language or "unknown"),
            language_prob=payload.get("language_probability"),
            backend=self.name,
            model=self.model_id,
            audio_sec=max(duration, 1e-3),
        )

    @staticmethod
    def _to_segment(raw: dict) -> TranscriptSegment:
        words = raw.get("words")
        return TranscriptSegment(
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=raw.get("text", ""),
            avg_logprob=raw.get("avg_logprob"),
            no_speech_prob=raw.get("no_speech_prob"),
            words=[Word(float(w["start"]), float(w["end"]), w["word"]) for w in words]
            if words else None,
        )


# ---------------------------------------------------------------------- ffmpeg
def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError as exc:
        raise TranscriptionError(f"could not measure {Path(path).name}") from exc


def _extract(src: Path, start: float, duration: float, dst: Path) -> None:
    """16 kHz mono FLAC — about half the size of the WAV the docs suggest,
    which matters against a 25 MB cap."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )
