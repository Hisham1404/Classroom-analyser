"""Local backend — faster-whisper running on the CPU.

Slow here (measured 0.42x realtime with `small` at int8), but it needs no network, no
API key and no quota, which keeps the offline-first claim honest and keeps the demo's
short-upload path working even if an external service is unreachable.
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.asr.base import Transcript, TranscriptSegment, TranscriptionError, Word
from src.asr.cache import TranscriptCache


class LocalWhisperBackend:
    name = "local"
    max_upload_bytes: int | None = None      # streams internally, never chunked

    def __init__(self, model: str | None = None, *, cache: TranscriptCache | None = None,
                 compute_type: str = "int8", cpu_threads: int | None = None) -> None:
        self.model_id = model or config.ASR_MODELS["primary"]
        self.cache = cache if cache is not None else TranscriptCache(config.ASR_CACHE_DIR)
        self.compute_type = compute_type
        # 16 threads measured slower than 8 on this CPU — the efficiency cores hurt.
        self.cpu_threads = cpu_threads if cpu_threads is not None else config.ASR_CPU_THREADS
        self._model = None

    def _load_model(self):
        from faster_whisper import WhisperModel

        config.configure_hf_cache()
        return WhisperModel(self.model_id, device="cpu",
                            compute_type=self.compute_type, cpu_threads=self.cpu_threads)

    def _model_once(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def transcribe(self, path: Path, language: str | None = None,
                   use_cache: bool = True) -> Transcript:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")

        if use_cache:
            hit = self.cache.get(path, self.name, self.model_id, language)
            if hit is not None:
                return hit

        try:
            segments, info = self._model_once().transcribe(
                str(path),
                language=language,
                beam_size=config.ASR_BEAM_SIZE,
                vad_filter=True,
                word_timestamps=True,
            )
            collected = [self._to_segment(s) for s in segments]
        except Exception as exc:                       # noqa: BLE001 - surfaced as our error
            raise TranscriptionError(f"local transcription failed for {path.name}: {exc}") from exc

        transcript = Transcript(
            segments=collected,
            language=getattr(info, "language", language or "unknown"),
            language_prob=getattr(info, "language_probability", None),
            backend=self.name,
            model=self.model_id,
            audio_sec=float(getattr(info, "duration", 0.0)) or _fallback_duration(collected),
        )

        if use_cache:
            self.cache.put(path, self.name, self.model_id, language, transcript)
        return transcript

    @staticmethod
    def _to_segment(s) -> TranscriptSegment:
        raw_words = getattr(s, "words", None)
        return TranscriptSegment(
            start=float(s.start),
            end=float(s.end),
            text=s.text or "",
            avg_logprob=getattr(s, "avg_logprob", None),
            no_speech_prob=getattr(s, "no_speech_prob", None),
            words=[Word(float(w.start), float(w.end), w.word) for w in raw_words]
            if raw_words else None,
        )


def _fallback_duration(segments: list[TranscriptSegment]) -> float:
    return max((s.end for s in segments), default=0.0) or 1e-3
