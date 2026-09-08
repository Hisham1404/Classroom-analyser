"""The validation harness for teacher/student attribution.

Without this, "the classifier works" is an assertion. The sheet makes it a number.

Design constraint that matters: an **unlabelled row must never count as correct**. The
tempting bug is to treat a blank truth column as agreement with the prediction, which
would report 100% accuracy on a sheet nobody has filled in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.attribution import Role, assign_roles
from src.evaluation.labelling import (
    LabelRow,
    build_sheet,
    parse_sheet,
    render_sheet,
    score_labels,
)
from src.signal_layer import Segment, SegmentKind


def speech(start, end, rms_db):
    return Segment(start, end, SegmentKind.SPEECH, rms_db, 0.2)


def handson(start, end):
    return Segment(start, end, SegmentKind.HANDSON, -30.0, 0.6)


@pytest.fixture
def segments():
    return assign_roles([speech(0, 8, -18), speech(8, 11, -34), handson(11, 40),
                         speech(40, 48, -19), speech(48, 51, -35)])


# ================================================================ building
def test_one_row_per_speech_segment(segments):
    """Hands-on noise has no speaker, so there is nothing to label."""
    rows = build_sheet("SESSION_A", segments)
    assert len(rows) == 4


def test_rows_carry_the_prediction_and_its_confidence(segments):
    row = build_sheet("SESSION_A", segments)[0]
    assert row.predicted is Role.TEACHER
    assert 0.0 <= row.confidence <= 1.0


def test_rows_carry_timing_for_seeking(segments):
    row = build_sheet("SESSION_A", segments)[0]
    assert row.start == pytest.approx(0.0)
    assert row.end == pytest.approx(8.0)


def test_row_ids_are_stable_and_sortable(segments):
    ids = [r.row_id for r in build_sheet("SESSION_A", segments)]
    assert ids == sorted(ids)
    assert ids[0].startswith("SESSION_A")


def test_truth_starts_empty(segments):
    assert all(r.truth is None for r in build_sheet("SESSION_A", segments))


def test_a_session_with_no_speech_gives_an_empty_sheet():
    assert build_sheet("SESSION_B", [handson(0, 30)]) == []


# ================================================================ round trip
def test_sheet_renders_as_markdown_with_a_blank_column(segments):
    md = render_sheet(build_sheet("SESSION_A", segments))
    assert "| Truth |" in md or "Truth" in md
    assert "SESSION_A" in md


def test_rendered_sheet_survives_a_round_trip(segments):
    rows = build_sheet("SESSION_A", segments)
    assert [r.row_id for r in parse_sheet(render_sheet(rows))] == [r.row_id for r in rows]


def test_parsing_reads_back_a_filled_truth_column(segments):
    rows = build_sheet("SESSION_A", segments)
    md = render_sheet(rows).replace("|  |", "| teacher |", 1)
    assert any(r.truth is Role.TEACHER for r in parse_sheet(md))


def test_parsing_tolerates_stray_whitespace_and_case(segments):
    rows = build_sheet("SESSION_A", segments)
    md = render_sheet(rows).replace("|  |", "|  Student  |", 1)
    assert any(r.truth is Role.STUDENT for r in parse_sheet(md))


def test_parsing_ignores_an_unrecognised_truth_value(segments):
    rows = build_sheet("SESSION_A", segments)
    md = render_sheet(rows).replace("|  |", "| ??? |", 1)
    assert all(r.truth is None for r in parse_sheet(md))


def test_parsing_an_empty_document_gives_no_rows():
    assert parse_sheet("# nothing here\n") == []


# ================================================================ scoring
def row(row_id, predicted, truth, confidence=0.8):
    return LabelRow(row_id=row_id, start=0.0, end=1.0, predicted=predicted,
                    confidence=confidence, truth=truth)


def test_perfect_agreement_scores_one():
    rows = [row("a", Role.TEACHER, Role.TEACHER), row("b", Role.STUDENT, Role.STUDENT)]
    assert score_labels(rows).accuracy == pytest.approx(1.0)


def test_total_disagreement_scores_zero():
    rows = [row("a", Role.TEACHER, Role.STUDENT), row("b", Role.STUDENT, Role.TEACHER)]
    assert score_labels(rows).accuracy == 0.0


def test_unlabelled_rows_are_excluded_not_counted_as_correct():
    """The dangerous bug: an empty sheet reporting 100%."""
    rows = [row("a", Role.TEACHER, Role.TEACHER), row("b", Role.STUDENT, None)]
    result = score_labels(rows)
    assert result.labelled == 1
    assert result.accuracy == pytest.approx(1.0)


def test_a_completely_unlabelled_sheet_has_no_accuracy():
    result = score_labels([row("a", Role.TEACHER, None)])
    assert result.labelled == 0
    assert result.accuracy is None


def test_per_role_recall_is_reported():
    rows = [row("a", Role.TEACHER, Role.TEACHER), row("b", Role.STUDENT, Role.TEACHER),
            row("c", Role.STUDENT, Role.STUDENT)]
    result = score_labels(rows)
    assert result.recall[Role.TEACHER] == pytest.approx(0.5)
    assert result.recall[Role.STUDENT] == pytest.approx(1.0)


def test_confusion_counts_are_reported():
    rows = [row("a", Role.TEACHER, Role.TEACHER), row("b", Role.STUDENT, Role.TEACHER)]
    assert score_labels(rows).confusion[(Role.TEACHER, Role.STUDENT)] == 1


def test_accuracy_on_confident_rows_is_reported_separately():
    """If the classifier is right when it is sure, the confidence gate is doing its job."""
    rows = [row("a", Role.TEACHER, Role.TEACHER, confidence=0.9),
            row("b", Role.TEACHER, Role.STUDENT, confidence=0.05)]
    result = score_labels(rows, confidence_floor=0.5)
    assert result.accuracy == pytest.approx(0.5)
    assert result.confident_accuracy == pytest.approx(1.0)


def test_result_is_json_serialisable():
    import json
    rows = [row("a", Role.TEACHER, Role.TEACHER)]
    json.loads(json.dumps(score_labels(rows).to_dict()))


def test_scoring_no_rows_at_all():
    result = score_labels([])
    assert result.labelled == 0
    assert result.accuracy is None


# ================================================================ transcript column
def test_rows_can_carry_transcript_text(segments):
    from src.asr.base import Transcript, TranscriptSegment

    t = Transcript(
        segments=[TranscriptSegment(0, 8, "आज हमारे क्लास में", -0.4, 0.1, None)],
        language="hi", language_prob=0.9, backend="s", model="m", audio_sec=60.0)
    rows = build_sheet("SESSION_A", segments, transcript=t)
    assert rows[0].text.startswith("आज हमारे")


def test_text_is_empty_without_a_transcript(segments):
    assert all(r.text == "" for r in build_sheet("SESSION_A", segments))


def test_the_sheet_shows_the_text(segments):
    from src.asr.base import Transcript, TranscriptSegment

    t = Transcript(
        segments=[TranscriptSegment(0, 8, "आज हमारे क्लास में", -0.4, 0.1, None)],
        language="hi", language_prob=0.9, backend="s", model="m", audio_sec=60.0)
    assert "आज हमारे" in render_sheet(build_sheet("SESSION_A", segments, transcript=t))


def test_a_sheet_with_text_still_round_trips(segments):
    from src.asr.base import Transcript, TranscriptSegment

    t = Transcript(
        segments=[TranscriptSegment(0, 8, "आज हमारे क्लास में", -0.4, 0.1, None)],
        language="hi", language_prob=0.9, backend="s", model="m", audio_sec=60.0)
    rows = build_sheet("SESSION_A", segments, transcript=t)
    parsed = parse_sheet(render_sheet(rows))
    assert [r.row_id for r in parsed] == [r.row_id for r in rows]


def test_truth_still_parses_with_a_text_column(segments):
    rows = build_sheet("SESSION_A", segments)
    # Empty truth AND empty transcript render as "|  |  |" — target the first of the two.
    md = render_sheet(rows).replace("|  |  |", "| teacher |  |", 1)
    assert any(r.truth is Role.TEACHER for r in parse_sheet(md))


def test_a_pipe_in_the_transcript_does_not_break_the_table(segments):
    from src.asr.base import Transcript, TranscriptSegment

    t = Transcript(
        segments=[TranscriptSegment(0, 8, "यह | वह", -0.4, 0.1, None)],
        language="hi", language_prob=0.9, backend="s", model="m", audio_sec=60.0)
    rows = build_sheet("SESSION_A", segments, transcript=t)
    assert len(parse_sheet(render_sheet(rows))) == len(rows)
