"""The command-line front door.

    python -m src.cli data/audio/<session>/<session>.mp3
    python -m src.cli <file> --backend groq
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.cli import build_parser, main


def test_backend_defaults_to_config():
    args = build_parser().parse_args(["some.mp3"])
    assert args.backend is None                       # resolved later, from config


def test_backend_can_be_chosen_on_the_command_line():
    assert build_parser().parse_args(["a.mp3", "--backend", "groq"]).backend == "groq"


def test_only_known_backends_are_accepted(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["a.mp3", "--backend", "nonsense"])


def test_language_defaults_to_the_configured_one():
    assert build_parser().parse_args(["a.mp3"]).language == config.ASR_LANGUAGE


def test_cache_can_be_disabled():
    assert build_parser().parse_args(["a.mp3", "--no-cache"]).no_cache is True


def test_json_output_path_is_optional():
    assert build_parser().parse_args(["a.mp3"]).out is None
    assert build_parser().parse_args(["a.mp3", "-o", "x.json"]).out == "x.json"


# ---------------------------------------------------------------- running it
@pytest.fixture
def stub_backend(monkeypatch):
    from src.asr.local import LocalWhisperBackend
    from tests.test_asr_backends import _FakeModel

    def fake_get_backend(name=None, **kw):
        b = LocalWhisperBackend(model="stub-model")
        b._load_model = lambda: _FakeModel()
        return b

    monkeypatch.setattr("src.cli.get_backend", fake_get_backend)


def test_main_prints_the_transcript(stub_backend, intact_session: Path, capsys):
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    assert main([str(audio)]) == 0
    assert "यह क्या है?" in capsys.readouterr().out


def test_main_reports_which_backend_ran(stub_backend, intact_session: Path, capsys):
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    main([str(audio)])
    assert "local" in capsys.readouterr().out


def test_main_writes_json_when_asked(stub_backend, intact_session: Path, tmp_path: Path):
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    out = tmp_path / "t.json"
    assert main([str(audio), "-o", str(out)]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["backend"] == "local"
    assert data["segments"]


def test_main_returns_nonzero_for_a_missing_file(stub_backend, tmp_path: Path, capsys):
    assert main([str(tmp_path / "ghost.mp3")]) == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_main_reports_a_backend_failure_without_a_traceback(monkeypatch, intact_session: Path,
                                                            capsys):
    from src.asr.base import TranscriptionError

    class Boom:
        name, model_id = "groq", "whisper-large-v3"

        def transcribe(self, *a, **kw):
            raise TranscriptionError("groq rate limit (429), retry after 60s")

    monkeypatch.setattr("src.cli.get_backend", lambda name=None, **kw: Boom())
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"

    assert main([str(audio), "--backend", "groq"]) == 1
    assert "429" in capsys.readouterr().err


def test_missing_api_key_is_a_clear_message_not_a_crash(monkeypatch, intact_session: Path,
                                                        capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"

    assert main([str(audio), "--backend", "groq"]) == 1
    assert "GROQ_API_KEY" in capsys.readouterr().err
