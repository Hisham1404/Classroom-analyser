"""Running every model over the same clips, and writing it up.

The runner must survive a model that fails — one broken contender should cost you that
row, not the whole comparison you just spent two hours computing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asr.base import Transcript, TranscriptSegment, TranscriptionError
from src.evaluation.bakeoff import Contender, BakeoffResult, rank_contenders, run_bakeoff
from src.evaluation.clips import ClipSpec
from src.evaluation.report import hand_scoring_sheet, render_report


def transcript(text: str, logprob: float = -0.5, model: str = "m", backend: str = "local"):
    return Transcript(
        segments=[TranscriptSegment(0.0, 5.0, text, logprob, 0.1, None)],
        language="hi", language_prob=0.9, backend=backend, model=model, audio_sec=10.0,
    )


class FakeBackend:
    def __init__(self, name="local", model="m", text="यह कागज़ है", logprob=-0.5, fail=False):
        self.name, self.model_id = name, model
        self.text, self.logprob, self.fail = text, logprob, fail
        self.seen: list[Path] = []

    def transcribe(self, path, language=None, use_cache=True):
        if self.fail:
            raise TranscriptionError("model exploded")
        self.seen.append(Path(path))
        return transcript(self.text, self.logprob, self.model_id, self.name)


@pytest.fixture
def clips(intact_session: Path, tmp_path: Path):
    src = intact_session / "OD90001_2026-01-06-114155.mp3"
    return src, [ClipSpec("OD90001_2026-01-06-114155", i, float(i), 2.0) for i in range(2)]


# ================================================================ running
def test_every_contender_sees_every_clip(clips, tmp_path: Path):
    src, specs = clips
    a, b = FakeBackend(model="a"), FakeBackend(model="b")
    result = run_bakeoff({src: specs},
                         [Contender("A", a), Contender("B", b)],
                         workdir=tmp_path)

    assert len(result.rows) == 4                    # 2 clips x 2 contenders
    assert len(a.seen) == 2 and len(b.seen) == 2


def test_all_contenders_get_the_identical_audio(clips, tmp_path: Path):
    src, specs = clips
    a, b = FakeBackend(model="a"), FakeBackend(model="b")
    run_bakeoff({src: specs}, [Contender("A", a), Contender("B", b)], workdir=tmp_path)
    assert a.seen == b.seen


def test_rows_carry_signals_and_timing(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("A", FakeBackend())], workdir=tmp_path)
    row = result.rows[0]
    assert row.signals.devanagari_ratio == pytest.approx(1.0)
    assert row.elapsed_sec >= 0.0
    assert row.clip.clip_id.endswith("_c00")


def test_a_failing_contender_does_not_kill_the_run(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("good", FakeBackend()),
                          Contender("broken", FakeBackend(fail=True))],
                         workdir=tmp_path)

    assert len(result.rows) == 2                    # only the good one produced rows
    assert result.failures and "broken" in result.failures[0]


def test_every_contender_failing_is_reported_not_raised(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("x", FakeBackend(fail=True))], workdir=tmp_path)
    assert result.rows == []
    assert len(result.failures) == 2                # once per clip


def test_no_contenders_raises(clips, tmp_path: Path):
    src, specs = clips
    with pytest.raises(ValueError):
        run_bakeoff({src: specs}, [], workdir=tmp_path)


def test_result_is_json_serialisable(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("A", FakeBackend())], workdir=tmp_path)
    json.loads(json.dumps(result.to_dict()))


# ================================================================ ranking
def test_the_better_model_ranks_first(clips, tmp_path: Path):
    src, specs = clips
    good = FakeBackend(model="good", text="यह कागज़ है इसे मोड़कर देखो", logprob=-0.3)
    bad = FakeBackend(model="bad", text="और लगता आँ " * 12, logprob=-1.7)

    result = run_bakeoff({src: specs},
                         [Contender("good", good), Contender("bad", bad)],
                         workdir=tmp_path)
    assert [r.label for r in rank_contenders(result)][0] == "good"


def test_ranking_averages_over_every_clip(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("A", FakeBackend())], workdir=tmp_path)
    assert rank_contenders(result)[0].clips_scored == 2


def test_a_contender_with_no_successful_rows_is_excluded(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("ok", FakeBackend()),
                          Contender("dead", FakeBackend(fail=True))],
                         workdir=tmp_path)
    assert [r.label for r in rank_contenders(result)] == ["ok"]


def test_ranking_an_empty_result_gives_an_empty_list():
    assert rank_contenders(BakeoffResult(rows=[], failures=[])) == []


# ================================================================ report
def test_report_names_every_contender(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("whisper-small", FakeBackend(model="openai/whisper-small")),
                          Contender("hindi-medium", FakeBackend(model="vasista22/whisper-hindi-medium"))],
                         workdir=tmp_path)
    md = render_report(result)
    assert "whisper-small" in md and "hindi-medium" in md


def test_report_states_that_proxies_are_not_ground_truth(clips, tmp_path: Path):
    """The honesty clause — a reviewer must not read these numbers as WER."""
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("A", FakeBackend())], workdir=tmp_path)
    md = render_report(result).lower()
    assert "not" in md and ("ground truth" in md or "hand" in md)


def test_report_lists_failures(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("ok", FakeBackend()),
                          Contender("broken", FakeBackend(fail=True))],
                         workdir=tmp_path)
    assert "broken" in render_report(result)


def test_report_of_an_empty_run_still_renders():
    md = render_report(BakeoffResult(rows=[], failures=["everything died"]))
    assert "everything died" in md


def test_report_includes_a_transcript_sample_to_eyeball(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("A", FakeBackend(text="यह कागज़ है"))], workdir=tmp_path)
    assert "यह कागज़ है" in render_report(result)


# ================================================================ hand scoring
def test_hand_scoring_sheet_has_a_row_per_clip_and_contender(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("A", FakeBackend(model="a")),
                          Contender("B", FakeBackend(model="b"))],
                         workdir=tmp_path)
    sheet = hand_scoring_sheet(result)
    assert sheet.count("_c00") + sheet.count("_c01") >= 4


def test_hand_scoring_sheet_leaves_the_verdict_blank(clips, tmp_path: Path):
    """This is the column Mohammed fills in after listening."""
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("A", FakeBackend())], workdir=tmp_path)
    assert "| _ |" in hand_scoring_sheet(result) or "|  |" in hand_scoring_sheet(result)


def test_hand_scoring_sheet_shows_the_text_to_judge(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("A", FakeBackend(text="यह कागज़ है"))], workdir=tmp_path)
    assert "यह कागज़ है" in hand_scoring_sheet(result)


# ================================================================ agreement
def test_rows_carry_cross_model_agreement(clips, tmp_path: Path):
    """Two models saying the same thing is evidence; one agreeing with nobody is a warning."""
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("a", FakeBackend(model="a", text="यह कागज़ है")),
                          Contender("b", FakeBackend(model="b", text="यह कागज़ है"))],
                         workdir=tmp_path)
    assert all(r.agreement == pytest.approx(1.0) for r in result.rows)


def test_a_contender_agreeing_with_nobody_scores_zero_agreement(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("real", FakeBackend(model="a", text="यह कागज़ है")),
                          Contender("noise", FakeBackend(model="b", text="रबअब पूके टिया"))],
                         workdir=tmp_path)
    by = {r.label: r.agreement for r in result.rows if r.clip.index == 0}
    assert by["noise"] == pytest.approx(0.0)


def test_agreement_is_none_with_a_single_contender(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs}, [Contender("solo", FakeBackend())], workdir=tmp_path)
    assert all(r.agreement is None for r in result.rows)


def test_ranking_reports_mean_agreement(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("a", FakeBackend(model="a", text="यह कागज़ है")),
                          Contender("b", FakeBackend(model="b", text="यह कागज़ है"))],
                         workdir=tmp_path)
    assert rank_contenders(result)[0].agreement == pytest.approx(1.0)


def test_report_shows_the_agreement_column(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("a", FakeBackend(model="a")),
                          Contender("b", FakeBackend(model="b"))],
                         workdir=tmp_path)
    assert "Agreement" in render_report(result)


def test_agreement_is_per_contender_not_a_single_set_score(clips, tmp_path: Path):
    """Two models that agree must score high; the odd one out must score low.

    With only two contenders the value is symmetric and hides a bug, so this needs three.
    """
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("a", FakeBackend(model="a", text="यह कागज़ है इसे मोड़ो")),
                          Contender("b", FakeBackend(model="b", text="यह कागज़ है इसे मोड़ो")),
                          Contender("noise", FakeBackend(model="c", text="रबअब पूके टिया गेाहयी"))],
                         workdir=tmp_path)

    by = {r.label: r.agreement for r in result.rows if r.clip.index == 0}
    assert by["noise"] == pytest.approx(0.0)
    assert by["a"] > 0.4
    assert by["a"] == pytest.approx(by["b"])


def test_agreement_ranks_the_odd_one_out_last(clips, tmp_path: Path):
    src, specs = clips
    result = run_bakeoff({src: specs},
                         [Contender("a", FakeBackend(model="a", text="यह कागज़ है इसे मोड़ो")),
                          Contender("b", FakeBackend(model="b", text="यह कागज़ है इसे मोड़ो")),
                          Contender("noise", FakeBackend(model="c", text="रबअब पूके टिया गेाहयी"))],
                         workdir=tmp_path)
    scores = {s.label: s.agreement for s in rank_contenders(result)}
    assert scores["noise"] < scores["a"]
