"""The one-line switch: which ASR backend runs.

`config.ASR_BACKEND` is the single setting. `get_backend()` honours it; passing a name
overrides it. Nothing downstream should ever import a concrete backend directly.
"""

from __future__ import annotations

import pytest

from src import config
from src.asr import BACKENDS, get_backend
from src.asr.base import ASRBackend


def test_both_backends_are_registered():
    assert set(BACKENDS) == {"local", "groq", "indic"}


def test_default_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "ASR_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert get_backend().name == "groq"


def test_explicit_name_overrides_config(monkeypatch):
    monkeypatch.setattr(config, "ASR_BACKEND", "groq")
    assert get_backend("local").name == "local"


def test_unknown_backend_names_the_valid_options():
    with pytest.raises(ValueError) as e:
        get_backend("whisper.cpp")
    for name in ("local", "groq", "indic"):
        assert name in str(e.value)


def test_config_default_is_a_registered_backend():
    assert config.ASR_BACKEND in BACKENDS


def test_every_backend_satisfies_the_protocol(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for name in BACKENDS:
        b = get_backend(name)
        assert isinstance(b, ASRBackend)
        assert b.name == name
        assert isinstance(b.model_id, str) and b.model_id


def test_model_can_be_overridden_per_call():
    b = get_backend("local", model="openai/whisper-tiny")
    assert b.model_id == "openai/whisper-tiny"


def test_groq_backend_refuses_to_construct_without_a_key(monkeypatch):
    """Fail loudly at construction, not silently at the first HTTP call."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        get_backend("groq")
    assert "GROQ_API_KEY" in str(e.value)


def test_local_backend_needs_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert get_backend("local").name == "local"
