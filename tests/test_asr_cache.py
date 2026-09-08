"""Transcripts are cached to disk.

Re-transcribing is the most expensive thing in the project — 0.42x realtime locally,
and a metered daily quota on the API. A downstream bug must never cost a re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.asr.base import Transcript, TranscriptSegment
from src.asr.cache import TranscriptCache


def make_transcript(text="नमस्ते", backend="local", model="m1"):
    return Transcript(
        segments=[TranscriptSegment(0.0, 2.0, text, -0.5, 0.1, None)],
        language="hi", language_prob=0.9, backend=backend, model=model, audio_sec=2.0,
    )


@pytest.fixture
def cache(tmp_path: Path) -> TranscriptCache:
    return TranscriptCache(tmp_path / "asr")


@pytest.fixture
def audio(intact_session: Path) -> Path:
    return intact_session / "OD90001_2026-01-06-114155.mp3"


def test_miss_on_an_empty_cache(cache, audio):
    assert cache.get(audio, "local", "m1", "hi") is None


def test_store_then_hit(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    hit = cache.get(audio, "local", "m1", "hi")
    assert hit is not None and hit.text == "नमस्ते"


def test_hit_preserves_confidence_and_timings(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    hit = cache.get(audio, "local", "m1", "hi")
    assert hit.mean_logprob == pytest.approx(-0.5)
    assert hit.segments[0].end == pytest.approx(2.0)


def test_a_different_backend_is_a_different_entry(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript("local text"))
    assert cache.get(audio, "groq", "m1", "hi") is None


def test_a_different_model_is_a_different_entry(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    assert cache.get(audio, "local", "m2", "hi") is None


def test_a_different_language_is_a_different_entry(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    assert cache.get(audio, "local", "m1", "mr") is None


def test_two_backends_coexist_for_the_same_file(cache, audio):
    """The bake-off depends on this."""
    cache.put(audio, "local", "m1", "hi", make_transcript("from local", "local"))
    cache.put(audio, "groq", "m2", "hi", make_transcript("from groq", "groq"))
    assert cache.get(audio, "local", "m1", "hi").text == "from local"
    assert cache.get(audio, "groq", "m2", "hi").text == "from groq"


def test_key_follows_content_not_filename(cache, audio, tmp_path: Path):
    copy = tmp_path / "renamed.mp3"
    copy.write_bytes(audio.read_bytes())
    cache.put(audio, "local", "m1", "hi", make_transcript())
    assert cache.get(copy, "local", "m1", "hi") is not None


def test_changed_audio_invalidates_the_entry(cache, audio, tmp_path: Path):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    edited = tmp_path / "edited.mp3"
    edited.write_bytes(audio.read_bytes() + b"\x00extra")
    assert cache.get(edited, "local", "m1", "hi") is None


def test_cache_dir_is_created_on_demand(tmp_path: Path, audio):
    c = TranscriptCache(tmp_path / "deep" / "nested")
    c.put(audio, "local", "m1", "hi", make_transcript())
    assert (tmp_path / "deep" / "nested").is_dir()


def test_corrupt_entry_is_a_miss_not_a_crash(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    for f in cache.root.glob("*.json"):
        f.write_text("{ broken", encoding="utf-8")
    assert cache.get(audio, "local", "m1", "hi") is None


def test_clear_empties_the_cache(cache, audio):
    cache.put(audio, "local", "m1", "hi", make_transcript())
    cache.clear()
    assert cache.get(audio, "local", "m1", "hi") is None


def test_backend_uses_the_cache_on_the_second_call(monkeypatch, cache, audio):
    """Second identical request must not re-run transcription."""
    from src.asr.local import LocalWhisperBackend
    from tests.test_asr_backends import _FakeModel

    model = _FakeModel()
    b = LocalWhisperBackend(model="m1", cache=cache)
    monkeypatch.setattr(b, "_load_model", lambda: model)

    b.transcribe(audio, language="hi")
    b.transcribe(audio, language="hi")
    assert model.calls == 1


def test_cache_can_be_bypassed(monkeypatch, cache, audio):
    from src.asr.local import LocalWhisperBackend
    from tests.test_asr_backends import _FakeModel

    model = _FakeModel()
    b = LocalWhisperBackend(model="m1", cache=cache)
    monkeypatch.setattr(b, "_load_model", lambda: model)

    b.transcribe(audio, language="hi")
    b.transcribe(audio, language="hi", use_cache=False)
    assert model.calls == 2


def test_the_model_is_loaded_only_once_across_calls(monkeypatch, cache, audio):
    """Reusing a loaded model is deliberate - it is the expensive part."""
    from src.asr.local import LocalWhisperBackend
    from tests.test_asr_backends import _FakeModel

    b = LocalWhisperBackend(model="m1", cache=cache)
    loads = {"n": 0}

    def counting_load():
        loads["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(b, "_load_model", counting_load)
    b.transcribe(audio, language="hi", use_cache=False)
    b.transcribe(audio, language="hi", use_cache=False)
    assert loads["n"] == 1
