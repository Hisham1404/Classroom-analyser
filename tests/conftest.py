"""Shared fixtures.

Synthetic sessions are built with ffmpeg so the unit tests exercise real audio decoding
without depending on the real dataset, which is gitignored and must never reach CI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = PROJECT_ROOT / "data" / "audio"


def _write_tone(dst: Path, seconds: float) -> None:
    """Generate a real mp3 of an exact duration."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ac", "1", "-ar", "44100", "-b:a", "128k", str(dst)],
        check=True,
    )


def _make_session(root: Path, session_id: str, *, declared_ms: int, actual_sec: float,
                  teacher_id: str = "OD90001", activity: str = "Trumpet",
                  boys: int = 5, girls: int = 3, gps: bool = True,
                  teacher_name: str = "Testteacher") -> Path:
    d = root / session_id
    d.mkdir(parents=True, exist_ok=True)
    _write_tone(d / f"{session_id}.mp3", actual_sec)

    photo: dict = {
        "capturedAt": "2026-01-06 12:51:13",
        "photoPath": f"/storage/emulated/0/.../{session_id}.jpg",
        "teacherName": teacher_name,
    }
    if gps:
        photo |= {
            "address": "Shop No 1 Igatpuri Giranare, Igatpuri, Maharashtra, India",
            "latitude": 19.6962197,
            "longitude": 73.5798404,
        }

    (d / f"{session_id}.json").write_text(json.dumps({
        "activityType": activity,
        "boys": boys,
        "girls": girls,
        "duration": declared_ms,
        "photoMetadata": photo,
        "recordingFilePath": f"/storage/emulated/0/.../{session_id}.mp3",
        "teacherId": teacher_id,
        "timestamp": "2026-01-06 12:51:13",
        "totalStudents": boys + girls,
    }, indent=2), encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    """Never let a test write into the project's real caches.

    Without this, a cached result from one test silently satisfies the next one —
    which is exactly how three tests passed for the wrong reason.

    The diarization cache repeated the lesson the moment it was added: every synthetic
    session is built from the same 6 s tone, so they hash identically, and three tests
    that assert what `diarize` does on a MISS started reading each other's hits instead.
    A content-addressed cache makes identical fixtures indistinguishable — which is the
    point of it, and the reason it has to be isolated per test.
    """
    from src import config
    monkeypatch.setattr(config, "ASR_CACHE_DIR", tmp_path / "asr-cache")
    monkeypatch.setattr(config, "DIARIZATION_CACHE_DIR", tmp_path / "turn-cache")


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def intact_session(tmp_path: Path, ffmpeg_available: bool) -> Path:
    """Declared duration matches the audio."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _make_session(tmp_path, "OD90001_2026-01-06-114155",
                         declared_ms=6000, actual_sec=6.0)


@pytest.fixture
def truncated_session(tmp_path: Path, ffmpeg_available: bool) -> Path:
    """Declared 12 s, only 4 s of audio present — the bug seen in 3 of 5 real files."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return _make_session(tmp_path, "OD90002_2026-01-12-142132",
                         declared_ms=12000, actual_sec=4.0,
                         teacher_id="OD90002", activity="Shadow Art", gps=False)


@pytest.fixture
def corpus(tmp_path: Path, ffmpeg_available: bool) -> Path:
    """Three sessions across two teachers, plus junk that must be ignored."""
    if not ffmpeg_available:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    root = tmp_path / "audio"
    _make_session(root, "OD90002_2026-01-20-115538", declared_ms=5000, actual_sec=5.0,
                  teacher_id="OD90002", activity="Wind Anemometer", gps=False)
    _make_session(root, "OD90001_2025-12-23-121239", declared_ms=5000, actual_sec=5.0,
                  teacher_id="OD90001")
    _make_session(root, "OD90001_2026-01-28-121933", declared_ms=9000, actual_sec=3.0,
                  teacher_id="OD90001")

    # a folder with metadata but no audio — must be skipped, not crash
    orphan = root / "OD90003_2026-02-01-101010"
    orphan.mkdir(parents=True)
    (orphan / "OD90003_2026-02-01-101010.json").write_text("{}", encoding="utf-8")

    # a stray file at the top level — must be ignored
    (root / "notes.txt").write_text("ignore me", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    if not REAL_DATA.exists():
        pytest.skip("real dataset not present at data/audio")
    return REAL_DATA
