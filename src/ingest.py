"""Stage 01 — find sessions, read their metadata, measure the audio.

Nothing here loads a model. It exists so that everything downstream can rely on one
guarantee: durations come from the decoded stream, not from what the capture app claimed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.models import AliasRegistry, AudioProbe, Session, SessionMeta


def discover_session_dirs(root: Path) -> list[Path]:
    """Session folders under `root` that hold both an mp3 and its json sidecar.

    Sorted, so a batch run is reproducible. Folders missing either file are skipped
    rather than raising — real field data has gaps.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio root does not exist: {root}")

    found = [
        d for d in root.iterdir()
        if d.is_dir() and (d / f"{d.name}.mp3").is_file() and (d / f"{d.name}.json").is_file()
    ]
    return sorted(found, key=lambda d: d.name)


def load_session_meta(session_dir: Path) -> SessionMeta:
    """Parse the json sidecar next to the recording."""
    session_dir = Path(session_dir)
    path = session_dir / f"{session_dir.name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no metadata for session {session_dir.name}: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable metadata for session {session_dir.name}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"unreadable metadata for session {session_dir.name}: not an object")

    return SessionMeta.from_dict(session_dir.name, raw)


def probe_audio(path: Path) -> AudioProbe:
    """Measure the file with ffprobe. This is the authoritative duration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")

    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffprobe could not read {path.name}")

    payload = json.loads(proc.stdout)
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError(f"no audio stream in {path.name}")
    stream, fmt = streams[0], payload.get("format", {})

    return AudioProbe(
        path=path,
        actual_sec=float(fmt.get("duration") or stream.get("duration") or 0.0),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        bit_rate=int(float(fmt.get("bit_rate") or 0)),
        size_bytes=path.stat().st_size,
    )


def build_session(session_dir: Path, alias: str | None = None) -> Session:
    """Metadata + measured audio for one session."""
    session_dir = Path(session_dir)
    meta = load_session_meta(session_dir)
    probe = probe_audio(session_dir / f"{session_dir.name}.mp3")

    if alias is None:
        registry = AliasRegistry([meta.teacher_id] if meta.teacher_id else [])
        alias = registry.alias(meta.teacher_id) if meta.teacher_id else "Teacher ?"

    return Session(meta=meta, probe=probe, alias=alias)


def build_corpus(root: Path) -> list[Session]:
    """Every session under `root`, with one stable alias per teacher across the whole set."""
    dirs = discover_session_dirs(root)
    metas = [load_session_meta(d) for d in dirs]
    registry = AliasRegistry(m.teacher_id for m in metas if m.teacher_id)

    sessions: list[Session] = []
    for d, meta in zip(dirs, metas):
        alias = registry.alias(meta.teacher_id) if meta.teacher_id else "Teacher ?"
        sessions.append(Session(
            meta=meta,
            probe=probe_audio(d / f"{d.name}.mp3"),
            alias=alias,
        ))
    return sessions
