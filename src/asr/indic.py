"""AI4Bharat IndicConformer, run through onnx-asr.

A different architecture from the Whisper backends, and that difference is the point.

Whisper decodes autoregressively, so on noisy audio it can generate a fluent sentence that
was never spoken — we measured exactly that (`docs/asr-options.md` §1). A CTC/RNNT conformer
decodes frame by frame with no generative decoder to invent with, so where the two disagree
wildly, Whisper is usually the one making things up.

It also needs no torch: onnxruntime already ships with faster-whisper, which matters on a
drive with 3.2 GB free.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src import config
from src.asr.base import Transcript, TranscriptSegment, TranscriptionError
from src.asr.cache import TranscriptCache
from src.asr.chunking import merge_chunk_transcripts, plan_chunks


class IndicConformerBackend:
    name = "indic"
    max_upload_bytes: int | None = None      # runs locally, nothing is uploaded

    def __init__(self, model: str | None = None, *, cache: TranscriptCache | None = None,
                 quantization: str = "int8", use_vad: bool = False) -> None:
        self.model_id = model or config.INDIC_MODEL
        self.cache = cache if cache is not None else TranscriptCache(config.ASR_CACHE_DIR)
        self.quantization = quantization
        self.use_vad = use_vad
        self.max_audio_sec = config.INDIC_MAX_AUDIO_SEC
        self._model = None
        self._recognizer = None
        self._plain = None

    # ------------------------------------------------------------------ model
    def _base_model(self):
        import onnx_asr

        config.configure_hf_cache()
        if self._model is None:
            self._model = onnx_asr.load_model(self.model_id, quantization=self.quantization)
        return self._model

    def _load_recognizer(self):
        """Timestamped pipeline. VAD is off by default - see `use_vad`."""
        import onnx_asr

        pipeline = self._base_model().with_timestamps()
        if self.use_vad:
            pipeline = pipeline.with_vad(onnx_asr.load_vad("silero"), batch_size=4)
        return pipeline

    def _load_plain_recognizer(self):
        """Same model, no VAD. The fallback when VAD rejects a whole clip."""
        return self._base_model().with_timestamps()

    def _recognizer_once(self):
        if self._recognizer is None:
            self._recognizer = self._load_recognizer()
        return self._recognizer

    def _plain_once(self):
        if self._plain is None:
            self._plain = self._load_plain_recognizer()
        return self._plain

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

        audio_sec = _probe_duration(path)

        # The ONNX graph has a fixed positional-encoding length; past ~100 s it dies with
        # "Attempting to broadcast an axis ... 2501 by 7501". Split and stitch instead.
        if audio_sec > self.max_audio_sec:
            transcript = self._transcribe_chunked(path, language, audio_sec)
            if use_cache:
                self.cache.put(path, self.name, self.model_id, language, transcript)
            return transcript

        try:
            raw = self._recognizer_once().recognize(str(path))
            results = _as_list(raw)

            # Preferred path: rebuild segments from the token stream, which keeps the
            # per-token logprobs that SegmentResult (the VAD path) throws away.
            segments: list[TranscriptSegment] = []
            for r in results:
                tokens = getattr(r, "tokens", None)
                if tokens:
                    segments.extend(segments_from_tokens(
                        tokens=tokens,
                        timestamps=getattr(r, "timestamps", None) or [],
                        logprobs=getattr(r, "logprobs", None),
                        gap_sec=config.INDIC_SEGMENT_GAP_SEC,
                        audio_sec=audio_sec,
                    ))
                else:
                    seg = self._to_segment(r, span=audio_sec)
                    if seg:
                        segments.append(seg)

            # Measured: Silero returns zero segments on 2 of 3 real classroom clips where
            # the same model without VAD finds 300+ characters of real speech. An
            # over-aggressive VAD must not turn a usable transcript into silence.
            if not segments and self.use_vad:
                fallback = self._plain_once().recognize(str(path))
                segments = [
                    s for s in (self._to_segment(r, span=audio_sec)
                                for r in _as_list(fallback)) if s
                ]
        except Exception as exc:                        # noqa: BLE001 - surfaced as ours
            raise TranscriptionError(
                f"indic transcription failed for {path.name}: {exc}") from exc

        transcript = Transcript(
            segments=segments,
            # The model is single-language by construction, so trust the caller/config
            # rather than pretending to have detected anything.
            language=language or config.ASR_LANGUAGE,
            language_prob=None,
            backend=self.name,
            model=self.model_id,
            audio_sec=audio_sec,
        )

        if use_cache:
            self.cache.put(path, self.name, self.model_id, language, transcript)
        return transcript

    def _transcribe_chunked(self, path: Path, language: str | None,
                            audio_sec: float) -> Transcript:
        """Cut into model-sized pieces, transcribe each, shift the times back."""
        # plan_chunks is strict about overlap < chunk; clamp so a short max_audio_sec
        # gives sensible behaviour instead of a crash.
        overlap = max(0.0, min(config.INDIC_CHUNK_OVERLAP_SEC, self.max_audio_sec / 4))
        chunks = plan_chunks(audio_sec, self.max_audio_sec, overlap)

        parts: list[Transcript] = []
        with tempfile.TemporaryDirectory() as tmp:
            for chunk in chunks:
                piece = Path(tmp) / f"{path.stem}_{chunk.index:03d}.wav"
                _extract(path, chunk.start, chunk.duration, piece)
                parts.append(self._transcribe_one(piece, language, chunk.duration))

        return merge_chunk_transcripts(parts, chunks, audio_sec=audio_sec)

    def _transcribe_one(self, path: Path, language: str | None,
                        audio_sec: float) -> Transcript:
        """One model-sized piece. Shared by the plain and chunked paths."""
        try:
            raw = self._recognizer_once().recognize(str(path))
            segments: list[TranscriptSegment] = []
            for r in _as_list(raw):
                tokens = getattr(r, "tokens", None)
                if tokens:
                    segments.extend(segments_from_tokens(
                        tokens=tokens,
                        timestamps=getattr(r, "timestamps", None) or [],
                        logprobs=getattr(r, "logprobs", None),
                        gap_sec=config.INDIC_SEGMENT_GAP_SEC,
                        audio_sec=audio_sec,
                    ))
                else:
                    seg = self._to_segment(r, span=audio_sec)
                    if seg:
                        segments.append(seg)
        except Exception as exc:                        # noqa: BLE001
            raise TranscriptionError(
                f"indic transcription failed for {path.name}: {exc}") from exc

        return Transcript(
            segments=segments,
            language=language or config.ASR_LANGUAGE,
            language_prob=None,
            backend=self.name,
            model=self.model_id,
            audio_sec=audio_sec,
        )

    @staticmethod
    def _to_segment(result, span: float | None = None) -> TranscriptSegment | None:
        """Map one onnx-asr result. Returns None for VAD's silent stretches.

        `span` is set on the no-VAD fallback, where a single result covers the whole clip
        and carries no start/end of its own.
        """
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return None

        logprobs = getattr(result, "logprobs", None)
        return TranscriptSegment(
            start=float(getattr(result, "start", 0.0) or 0.0),
            end=float(getattr(result, "end", 0.0) or 0.0) or (span or 0.0),
            text=text,
            # None, not 0.0 — zero would read downstream as perfect confidence.
            avg_logprob=(sum(logprobs) / len(logprobs)) if logprobs else None,
            no_speech_prob=None,
            words=None,
        )


def segments_from_tokens(tokens, timestamps, logprobs, gap_sec: float,
                         audio_sec: float) -> list[TranscriptSegment]:
    """Rebuild timed segments from a CTC token stream.

    Tokens carry their own leading spaces and reconstruct the text exactly, so they are
    joined bare rather than space-separated. A pause longer than `gap_sec` between two
    token timestamps ends the segment.
    """
    if not tokens or not timestamps:
        return []

    n = min(len(tokens), len(timestamps))
    probs = list(logprobs)[:n] if logprobs else None

    runs: list[list[int]] = []
    for i in range(n):
        if i and (timestamps[i] - timestamps[i - 1]) > gap_sec:
            runs.append([i])
        elif runs:
            runs[-1].append(i)
        else:
            runs.append([i])

    out: list[TranscriptSegment] = []
    for run in runs:
        text = "".join(tokens[i] for i in run).strip()
        if not text:
            continue

        start = float(timestamps[run[0]])
        # The last stamp is where that token *starts*, so extend by a typical token
        # length rather than ending the segment mid-word.
        spans = [timestamps[b] - timestamps[a] for a, b in zip(run, run[1:])]
        tail = min(spans) if spans else 0.2
        end = min(float(timestamps[run[-1]]) + max(tail, 0.05), audio_sec)

        # A short logprobs list must not take the run down with it — the three lists come
        # from the model and are only usually the same length.
        scored = [probs[i] for i in run
                  if probs and i < len(probs) and probs[i] is not None] if probs else []
        out.append(TranscriptSegment(
            start=start,
            end=max(end, start),
            text=text,
            avg_logprob=(sum(scored) / len(scored)) if scored else None,
            no_speech_prob=None,
            words=None,
        ))
    return out


def _extract(src: Path, start: float, duration: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def _as_list(result) -> list:
    """onnx-asr returns an iterator with VAD and a single object without it."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    return list(result) if hasattr(result, "__iter__") else [result]


def _probe_duration(path: Path) -> float:
    """Measured length. Segment end times cover only speech, so they are not a stand-in."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return max(float(out.stdout.strip()), 1e-3)
    except ValueError as exc:
        raise TranscriptionError(f"could not measure {Path(path).name}") from exc
