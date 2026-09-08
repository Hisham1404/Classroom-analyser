"""Disk cache for transcripts, keyed by audio content rather than filename.

Transcription is the most expensive step in the project: 0.42x realtime on this CPU, and
a metered daily quota on the hosted API. A bug in a later stage must never cost a re-run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.asr.base import Transcript

_READ_CHUNK = 1 << 20


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_READ_CHUNK):
            h.update(block)
    return h.hexdigest()[:20]


class TranscriptCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def key(self, path: Path, backend: str, model: str, language: str | None) -> str:
        digest = _hash_file(Path(path))
        stamp = hashlib.sha256(f"{backend}|{model}|{language or 'auto'}".encode()).hexdigest()[:10]
        return f"{digest}-{stamp}"

    def _path(self, *args) -> Path:
        return self.root / f"{self.key(*args)}.json"

    def get(self, path: Path, backend: str, model: str,
            language: str | None) -> Transcript | None:
        entry = self._path(path, backend, model, language)
        if not entry.is_file():
            return None
        try:
            return Transcript.from_dict(json.loads(entry.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None          # a damaged entry is a miss, never a crash

    def put(self, path: Path, backend: str, model: str, language: str | None,
            transcript: Transcript) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = self._path(path, backend, model, language)
        entry.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False, indent=1),
                         encoding="utf-8")
        return entry

    def clear(self) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        for f in self.root.glob("*.json"):
            f.unlink()
            removed += 1
        return removed
