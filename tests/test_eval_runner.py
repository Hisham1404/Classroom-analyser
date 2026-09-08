"""The bake-off command line.

    python -m src.evaluation.run --clips 4 --clip-sec 180
    python -m src.evaluation.run --contenders baseline,primary --sessions 2
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.evaluation.run import build_contenders, build_parser, main


# ---------------------------------------------------------------- arguments
def test_defaults_are_sane():
    args = build_parser().parse_args([])
    assert args.clips >= 2
    assert args.clip_sec >= 30
    assert args.contenders


def test_clip_count_and_length_are_settable():
    args = build_parser().parse_args(["--clips", "6", "--clip-sec", "60"])
    assert args.clips == 6 and args.clip_sec == 60.0


def test_sessions_can_be_limited_for_a_quick_run():
    assert build_parser().parse_args(["--sessions", "2"]).sessions == 2


def test_output_directory_defaults_into_eval():
    assert Path(build_parser().parse_args([]).out_dir) == config.EVAL_DIR


# ---------------------------------------------------------------- contenders
def test_named_config_models_become_contenders(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    contenders = build_contenders(["baseline", "primary"])
    assert [c.label for c in contenders] == ["baseline", "primary"]
    assert contenders[0].model_id == config.ASR_MODELS["baseline"]


def test_groq_is_included_when_a_key_is_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    labels = [c.label for c in build_contenders(["baseline", "groq"])]
    assert "groq" in labels


def test_groq_is_skipped_with_a_warning_when_no_key(monkeypatch, capsys):
    """Missing a key must cost you that column, not the whole run."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    contenders = build_contenders(["baseline", "groq"])

    assert [c.label for c in contenders] == ["baseline"]
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_an_unknown_contender_name_raises():
    with pytest.raises(ValueError) as e:
        build_contenders(["nonsense"])
    assert "nonsense" in str(e.value)


def test_no_usable_contenders_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        build_contenders(["groq"])


# ---------------------------------------------------------------- running
def test_main_writes_both_documents(monkeypatch, corpus: Path, tmp_path: Path):
    from src.asr.base import Transcript, TranscriptSegment

    class Stub:
        name, model_id = "local", "stub"

        def transcribe(self, path, language=None, use_cache=True):
            return Transcript(
                segments=[TranscriptSegment(0.0, 2.0, "यह कागज़ है", -0.4, 0.1, None)],
                language="hi", language_prob=0.9, backend="local", model="stub",
                audio_sec=2.0,
            )

    from src.evaluation.bakeoff import Contender
    monkeypatch.setattr("src.evaluation.run.build_contenders",
                        lambda names: [Contender("stub", Stub())])
    monkeypatch.setattr(config, "DATA_DIR", corpus)

    out = tmp_path / "eval"
    assert main(["--clips", "1", "--clip-sec", "2", "--out-dir", str(out)]) == 0

    assert (out / "asr_bakeoff.md").is_file()
    assert (out / "asr_handscoring.md").is_file()
    assert (out / "asr_bakeoff.json").is_file()
    assert "यह कागज़ है" in (out / "asr_bakeoff.md").read_text(encoding="utf-8")


def test_main_fails_clearly_when_there_is_no_audio(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "missing")
    assert main(["--out-dir", str(tmp_path / "eval")]) == 1
    assert "no sessions" in capsys.readouterr().err.lower()
