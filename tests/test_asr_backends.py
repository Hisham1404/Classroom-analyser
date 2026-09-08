"""Both backends, tested without a model download and without a network call.

The point of these tests is the *contract*: whatever runs, the caller gets the same
Transcript shape back. That is what makes the one-line switch safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.asr import get_backend
from src.asr.base import Transcript
from src.asr.groq import GroqBackend
from src.asr.local import LocalWhisperBackend


# ============================================================ local backend
class _FakeSegment:
    def __init__(self, start, end, text, logprob=-0.6, no_speech=0.2):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob = logprob, no_speech
        self.words = None


class _FakeModel:
    """Stands in for faster_whisper.WhisperModel."""

    last_kwargs: dict = {}

    def __init__(self, *a, **kw):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        _FakeModel.last_kwargs = kw
        segs = [_FakeSegment(0.0, 2.0, "यह क्या है?"), _FakeSegment(2.0, 4.0, "कागज़")]
        info = SimpleNamespace(language="hi", language_probability=0.94, duration=4.0)
        return iter(segs), info


@pytest.fixture
def local_backend(monkeypatch):
    b = LocalWhisperBackend(model="openai/whisper-small")
    monkeypatch.setattr(b, "_load_model", lambda: _FakeModel())
    return b


def test_local_returns_the_shared_transcript_type(local_backend, intact_session: Path):
    t = local_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert isinstance(t, Transcript)
    assert t.backend == "local"
    assert t.model == "openai/whisper-small"


def test_local_maps_segments_and_confidence(local_backend, intact_session: Path):
    t = local_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert t.text == "यह क्या है? कागज़"
    assert t.language == "hi"
    assert t.mean_logprob == pytest.approx(-0.6)


def test_local_passes_the_language_through(local_backend, intact_session: Path):
    local_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3", language="hi")
    assert _FakeModel.last_kwargs["language"] == "hi"


def test_local_does_not_chunk(local_backend, intact_session: Path):
    """faster-whisper streams internally — splitting would only hurt accuracy."""
    assert local_backend.max_upload_bytes is None


def test_local_missing_file_raises(local_backend, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        local_backend.transcribe(tmp_path / "ghost.mp3")


# ============================================================ groq backend
GROQ_RESPONSE = {
    "task": "transcribe",
    "language": "hindi",
    "duration": 4.0,
    "text": "यह क्या है? कागज़",
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": " यह क्या है?",
         "avg_logprob": -0.4, "no_speech_prob": 0.05},
        {"id": 1, "start": 2.0, "end": 4.0, "text": " कागज़",
         "avg_logprob": -0.8, "no_speech_prob": 0.12},
    ],
}


@pytest.fixture
def groq_backend(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    b = GroqBackend()
    monkeypatch.setattr(b, "_post_audio", lambda path, language=None: dict(GROQ_RESPONSE))
    return b


def test_groq_returns_the_same_transcript_type(groq_backend, intact_session: Path):
    t = groq_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert isinstance(t, Transcript)
    assert t.backend == "groq"
    assert "whisper-large-v3" in t.model


def test_groq_maps_segments_and_confidence(groq_backend, intact_session: Path):
    t = groq_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert t.text == "यह क्या है? कागज़"
    assert t.mean_logprob == pytest.approx(-0.6)


def test_groq_normalises_the_language_name_to_a_code(groq_backend, intact_session: Path):
    """Groq answers 'hindi'; the rest of the pipeline speaks ISO codes."""
    assert groq_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3").language == "hi"


def test_groq_declares_the_upload_cap():
    import os
    os.environ["GROQ_API_KEY"] = "k"
    assert GroqBackend().max_upload_bytes == 25 * 1024 * 1024


def test_groq_request_carries_auth_and_verbose_json(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-abc")
    b = GroqBackend()
    assert b._headers()["Authorization"] == "Bearer secret-abc"
    assert b._form_fields(language="hi")["response_format"] == "verbose_json"
    assert b._form_fields(language="hi")["language"] == "hi"


def test_groq_api_key_is_never_in_the_transcript(groq_backend, intact_session: Path):
    t = groq_backend.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert "test-key-123" not in json.dumps(t.to_dict())


def test_groq_surfaces_a_rate_limit_clearly(monkeypatch, intact_session: Path):
    from src.asr.base import TranscriptionError
    monkeypatch.setenv("GROQ_API_KEY", "k")
    b = GroqBackend()

    def boom(path, language=None):
        raise TranscriptionError("groq rate limit (429), retry after 60s")

    monkeypatch.setattr(b, "_post_audio", boom)
    with pytest.raises(TranscriptionError) as e:
        b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert "429" in str(e.value)


def test_groq_chunks_a_file_over_the_cap(monkeypatch, intact_session: Path):
    """Force the cap tiny so the 6 s fixture has to be split."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    b = GroqBackend(chunk_sec=2.0)
    monkeypatch.setattr(b, "max_upload_bytes", 10)          # everything is "too big"

    calls: list[Path] = []

    def fake_post(path, language=None):
        calls.append(path)
        return {"language": "hindi", "duration": 2.0,
                "segments": [{"start": 0.0, "end": 1.0, "text": f"c{len(calls)}",
                              "avg_logprob": -0.5, "no_speech_prob": 0.1}]}

    monkeypatch.setattr(b, "_post_audio", fake_post)
    t = b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")

    assert len(calls) == 3                                   # 6 s / 2 s
    assert t.text == "c1 c2 c3"
    assert t.audio_sec == pytest.approx(6.0, abs=0.3)

    # Each chunk's segments are shifted into whole-recording time. With chunk_sec=2.0
    # the overlap clamps to 0.5 s, so chunks begin at 0.0, 1.5 and 3.5.
    assert [round(s.start, 2) for s in t.segments] == [0.0, 1.5, 3.5]


# ============================================================ shared contract
@pytest.mark.parametrize("name", ["local", "groq"])
def test_switching_backend_changes_nothing_the_caller_sees(monkeypatch, name, intact_session: Path):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    b = get_backend(name)

    if name == "local":
        monkeypatch.setattr(b, "_load_model", lambda: _FakeModel())
    else:
        monkeypatch.setattr(b, "_post_audio", lambda path, language=None: dict(GROQ_RESPONSE))

    t = b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3", language="hi")

    assert isinstance(t, Transcript)
    assert t.language == "hi"
    assert t.text
    assert t.mean_logprob is not None
    assert t.backend == name
    assert set(t.to_dict()) == {
        "segments", "language", "language_prob", "backend", "model", "audio_sec",
    }


def test_groq_clamps_overlap_to_fit_a_short_chunk(monkeypatch):
    """A short chunk_sec with the default 5 s overlap must not blow up."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    b = GroqBackend(chunk_sec=2.0)
    assert b.overlap_sec < b.chunk_sec
    assert b.overlap_sec == pytest.approx(0.5)


def test_groq_keeps_a_sane_overlap_at_the_real_chunk_size(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    b = GroqBackend()
    assert b.chunk_sec == 600.0
    assert b.overlap_sec == pytest.approx(5.0)
