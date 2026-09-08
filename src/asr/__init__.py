"""ASR backends.

Switching which one runs is a single line in `src/config.py`:

    ASR_BACKEND = "local"    # or "groq" or "indic"

Nothing downstream imports a concrete backend. Ask for one here and you always get the
same `Transcript` shape back:

    from src.asr import get_backend

    backend = get_backend()                     # whatever config says
    backend = get_backend("groq")               # override for one call
    transcript = backend.transcribe(path, language="hi")
"""

from __future__ import annotations

from typing import Any, Callable

from src import config
from src.asr.base import (
    ASRBackend,
    Transcript,
    TranscriptionError,
    TranscriptSegment,
    Word,
)
from src.asr.cache import TranscriptCache
from src.asr.groq import GroqBackend
from src.asr.indic import IndicConformerBackend
from src.asr.local import LocalWhisperBackend

BACKENDS: dict[str, Callable[..., ASRBackend]] = {
    "local": LocalWhisperBackend,
    "groq": GroqBackend,
    "indic": IndicConformerBackend,
}


def get_backend(name: str | None = None, **kwargs: Any) -> ASRBackend:
    """Build a backend by name, defaulting to `config.ASR_BACKEND`."""
    chosen = (name or config.ASR_BACKEND).lower()
    if chosen not in BACKENDS:
        options = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown ASR backend {chosen!r}; available: {options}")
    return BACKENDS[chosen](**kwargs)


__all__ = [
    "ASRBackend",
    "BACKENDS",
    "GroqBackend",
    "IndicConformerBackend",
    "LocalWhisperBackend",
    "Transcript",
    "TranscriptCache",
    "TranscriptSegment",
    "TranscriptionError",
    "Word",
    "get_backend",
]
