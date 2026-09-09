"""Teacher/student from real diarization instead of loudness.

The energy split was measured at 41.7% — worse than chance, and beaten 100% to 41.7% by a
constant "always teacher". This replaces the feature, not the threshold.

The prior that makes diarization work here is much stronger than the loudness one:
**the teacher is one person who talks a lot; students are many people who each talk little.**
That holds whether or not the phone is nearest her, which is exactly where energy failed.
"""

from __future__ import annotations

import pytest

from src.attribution import Role
from src.diarization import (
    SpeakerTurn,
    assign_roles_from_turns,
    speaker_for_span,
    speech_spans,
    teacher_speaker,
)
from src.signal_layer import Segment, SegmentKind


def turn(start, end, speaker):
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def speech(start, end, rms=-20.0):
    return Segment(start, end, SegmentKind.SPEECH, rms, 0.2)


def handson(start, end):
    return Segment(start, end, SegmentKind.HANDSON, -30.0, 0.6)


# ================================================================ picking the teacher
def test_the_speaker_with_the_most_time_is_the_teacher():
    turns = [turn(0, 100, "SPEAKER_00"), turn(100, 110, "SPEAKER_01"),
             turn(110, 200, "SPEAKER_00"), turn(200, 205, "SPEAKER_02")]
    label, conf = teacher_speaker(turns)
    assert label == "SPEAKER_00"
    assert conf > 0.5


def test_many_students_each_speaking_briefly_do_not_outvote_the_teacher():
    """8-27 children is the real case: lots of speakers, each with little airtime."""
    turns = [turn(0, 200, "TEACHER")] + [turn(200 + i * 6, 205 + i * 6, f"S{i}")
                                         for i in range(20)]
    label, _ = teacher_speaker(turns)
    assert label == "TEACHER"


def test_a_close_call_reports_low_confidence():
    turns = [turn(0, 100, "A"), turn(100, 198, "B")]
    _, conf = teacher_speaker(turns)
    assert conf < 0.2


def test_a_single_speaker_is_the_teacher_with_full_confidence():
    label, conf = teacher_speaker([turn(0, 100, "ONLY")])
    assert label == "ONLY"
    assert conf == pytest.approx(1.0)


def test_no_turns_gives_no_teacher():
    label, conf = teacher_speaker([])
    assert label is None
    assert conf == 0.0


def test_confidence_is_bounded():
    for turns in ([turn(0, 1, "A")], [turn(0, 100, "A"), turn(100, 101, "B")],
                  [turn(0, 50, "A"), turn(50, 100, "B")]):
        _, conf = teacher_speaker(turns)
        assert 0.0 <= conf <= 1.0


# ================================================================ mapping to segments
def test_a_span_takes_the_speaker_who_overlaps_it_most():
    turns = [turn(0, 10, "A"), turn(10, 30, "B")]
    assert speaker_for_span(turns, 8, 28) == "B"


def test_a_span_with_no_overlap_has_no_speaker():
    assert speaker_for_span([turn(0, 10, "A")], 50, 60) is None


def test_span_matching_with_no_turns():
    assert speaker_for_span([], 0, 10) is None


# ================================================================ role assignment
def test_teacher_segments_are_labelled_teacher():
    turns = [turn(0, 100, "T"), turn(100, 110, "S1")]
    segs = [speech(0, 100), speech(100, 110)]
    out = assign_roles_from_turns(segs, turns)
    assert [s.role for s in out] == [Role.TEACHER, Role.STUDENT]


def test_every_non_teacher_speaker_becomes_a_student():
    """Per-child identity is not claimed — 27 children on one mic is not solvable."""
    turns = [turn(0, 100, "T"), turn(100, 110, "S1"), turn(110, 120, "S2")]
    segs = [speech(0, 100), speech(100, 110), speech(110, 120)]
    assert [s.role for s in assign_roles_from_turns(segs, turns)][1:] == \
           [Role.STUDENT, Role.STUDENT]


def test_non_speech_is_left_unassigned():
    turns = [turn(0, 100, "T")]
    out = assign_roles_from_turns([speech(0, 100), handson(100, 300)], turns)
    assert out[1].role is None


def test_a_segment_diarization_never_covered_is_left_unassigned():
    """Better to say nothing than to guess — that is what got us 41.7%."""
    turns = [turn(0, 50, "T")]
    out = assign_roles_from_turns([speech(0, 50), speech(200, 260)], turns)
    assert out[0].role is Role.TEACHER
    assert out[1].role is None


def test_confidence_comes_from_the_teacher_margin():
    clear = assign_roles_from_turns([speech(0, 100)],
                                    [turn(0, 100, "T"), turn(100, 105, "S")])
    murky = assign_roles_from_turns([speech(0, 100)],
                                    [turn(0, 100, "A"), turn(100, 198, "B")])
    assert clear[0].role_conf > murky[0].role_conf


def test_empty_inputs():
    assert assign_roles_from_turns([], []) == []
    assert assign_roles_from_turns([speech(0, 10)], []) == \
           [s.with_role(None, None) for s in [speech(0, 10)]]


def test_the_original_segments_are_not_mutated():
    segs = [speech(0, 100)]
    assign_roles_from_turns(segs, [turn(0, 100, "T")])
    assert segs[0].role is None


def test_assignment_is_deterministic():
    turns = [turn(0, 100, "T"), turn(100, 110, "S1")]
    segs = [speech(0, 100), speech(100, 110)]
    assert [s.role for s in assign_roles_from_turns(segs, turns)] == \
           [s.role for s in assign_roles_from_turns(segs, turns)]


# ================================================================ the real shape
def test_a_teacher_led_lesson_recovers_the_teacher_even_when_quieter():
    """The case energy got wrong: the teacher is not the loudest voice, but she is the
    one who holds the floor."""
    turns = [turn(0, 60, "T"), turn(60, 64, "S1"), turn(64, 130, "T"), turn(130, 133, "S2")]
    segs = [speech(0, 60, rms=-29.0),        # teacher, quiet
            speech(60, 64, rms=-18.0),       # student near the phone, loud
            speech(64, 130, rms=-29.5),
            speech(130, 133, rms=-17.0)]
    roles = [s.role for s in assign_roles_from_turns(segs, turns)]
    assert roles == [Role.TEACHER, Role.STUDENT, Role.TEACHER, Role.STUDENT]


def test_talk_time_totals_work_off_diarized_roles():
    from src.attribution import talk_time

    turns = [turn(0, 100, "T"), turn(100, 120, "S1")]
    segs = assign_roles_from_turns([speech(0, 100), speech(100, 120)], turns)
    totals = talk_time(segs)
    assert totals[Role.TEACHER] == pytest.approx(100.0)
    assert totals[Role.STUDENT] == pytest.approx(20.0)


# ================================================================ audio input
def test_audio_is_decoded_before_pyannote_sees_it(monkeypatch, intact_session):
    """pyannote 4.x decodes via torchcodec, whose DLL needs FFmpeg's shared libraries -
    absent on Windows. We already decode with ffmpeg ourselves, so hand it a tensor."""
    from src import diarization

    captured = {}

    class FakePipeline:
        def __call__(self, audio, **kwargs):
            captured["audio"] = audio
            captured["kwargs"] = kwargs
            return _FakeAnnotation()

    class _FakeAnnotation:
        def itertracks(self, yield_label=False):
            from types import SimpleNamespace
            yield SimpleNamespace(start=0.0, end=2.0), None, "SPEAKER_00"

    monkeypatch.setattr(diarization, "_load_pipeline", lambda token: FakePipeline())
    turns = diarization.diarize(
        intact_session / "OD90001_2026-01-06-114155.mp3", token="fake")

    assert isinstance(captured["audio"], dict)
    assert "waveform" in captured["audio"]
    assert captured["audio"]["sample_rate"] == 16000
    assert len(turns) == 1


def test_the_waveform_has_a_channel_dimension(monkeypatch, intact_session):
    """pyannote expects (channel, time); a bare 1-D array is silently misread."""
    from src import diarization

    captured = {}

    class FakePipeline:
        def __call__(self, audio, **kwargs):
            captured["audio"] = audio
            return _Empty()

    class _Empty:
        def itertracks(self, yield_label=False):
            return iter(())

    monkeypatch.setattr(diarization, "_load_pipeline", lambda token: FakePipeline())
    diarization.diarize(intact_session / "OD90001_2026-01-06-114155.mp3", token="fake")

    assert captured["audio"]["waveform"].ndim == 2
    assert captured["audio"]["waveform"].shape[0] == 1


def test_missing_token_raises_a_useful_message(monkeypatch, intact_session):
    from src import diarization

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as e:
        diarization.diarize(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert "HF_TOKEN" in str(e.value)


def test_num_speakers_is_passed_through(monkeypatch, intact_session):
    from src import diarization

    captured = {}

    class FakePipeline:
        def __call__(self, audio, **kwargs):
            captured.update(kwargs)
            return _Empty()

    class _Empty:
        def itertracks(self, yield_label=False):
            return iter(())

    monkeypatch.setattr(diarization, "_load_pipeline", lambda token: FakePipeline())
    diarization.diarize(intact_session / "OD90001_2026-01-06-114155.mp3",
                        token="fake", num_speakers=3)
    assert captured["num_speakers"] == 3


def test_pyannote_4x_output_wrapper_is_unwrapped(monkeypatch, intact_session):
    """4.x returns a DiarizeOutput holding the Annotation, not the Annotation itself."""
    from types import SimpleNamespace

    from src import diarization

    class Annotation:
        def itertracks(self, yield_label=False):
            yield SimpleNamespace(start=1.0, end=4.0), None, "SPEAKER_01"

    class DiarizeOutput:
        speaker_diarization = Annotation()

    monkeypatch.setattr(diarization, "_load_pipeline",
                        lambda token: lambda audio, **kw: DiarizeOutput())
    turns = diarization.diarize(
        intact_session / "OD90001_2026-01-06-114155.mp3", token="fake")

    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_01"
    assert turns[0].start == pytest.approx(1.0)


def test_a_bare_annotation_still_works(monkeypatch, intact_session):
    """Older pyannote returns the Annotation directly — keep both paths alive."""
    from types import SimpleNamespace

    from src import diarization

    class Annotation:
        def itertracks(self, yield_label=False):
            yield SimpleNamespace(start=0.0, end=2.0), None, "S0"

    monkeypatch.setattr(diarization, "_load_pipeline",
                        lambda token: lambda audio, **kw: Annotation())
    assert len(diarization.diarize(
        intact_session / "OD90001_2026-01-06-114155.mp3", token="fake")) == 1


# ================================================================ confidence semantics
def test_confidence_survives_an_evenly_split_room():
    """Measured: OD11166_2026-01-20 had a 0.06 airtime margin and still scored 75%.
    Margin measures how dominant the teacher is, not how reliable the split is — gating
    on it alone would suppress a session that was mostly right."""
    turns = [turn(0, 100, "A"), turn(100, 198, "B")]
    out = assign_roles_from_turns([speech(0, 100)], turns)
    from src import config
    assert out[0].role_conf > config.ROLE_CONF_FLOOR


def test_a_dominant_teacher_still_scores_higher_than_an_even_split():
    clear = assign_roles_from_turns([speech(0, 100)],
                                    [turn(0, 100, "T"), turn(100, 105, "S")])
    even = assign_roles_from_turns([speech(0, 100)],
                                   [turn(0, 100, "A"), turn(100, 198, "B")])
    assert clear[0].role_conf >= even[0].role_conf


def test_poor_coverage_drags_confidence_down():
    """If diarization only spoke for half the segments, we know less than if it covered
    all of them — that IS a reliability signal, unlike the margin."""
    turns = [turn(0, 50, "T"), turn(50, 55, "S")]
    full = assign_roles_from_turns([speech(0, 50)], turns)
    sparse = assign_roles_from_turns([speech(0, 50), speech(500, 560), speech(600, 660)],
                                     turns)
    assert sparse[0].role_conf < full[0].role_conf


def test_confidence_stays_bounded_across_shapes():
    for turns, segs in (
        ([turn(0, 100, "T")], [speech(0, 100)]),
        ([turn(0, 100, "A"), turn(100, 200, "B")], [speech(0, 100), speech(100, 200)]),
        ([turn(0, 1, "A")], [speech(0, 1), speech(900, 960)]),
    ):
        for s in assign_roles_from_turns(segs, turns):
            if s.role_conf is not None:
                assert 0.0 <= s.role_conf <= 1.0


# ================================================================ splitting on speaker change
from src.diarization import split_segments_by_speaker


def test_a_segment_spanning_two_speakers_is_split():
    """Measured: VAD segments run up to 63 s while diarization turns are seconds long.
    Assigning the whole segment to whoever dominates it erased every student turn and
    made M1 report 100% teacher talk on three sessions."""
    turns = [turn(0, 50, "T"), turn(50, 55, "S1"), turn(55, 63, "T")]
    out = split_segments_by_speaker([speech(0, 63)], turns)

    assert len(out) == 3
    assert [s.role for s in out] == [Role.TEACHER, Role.STUDENT, Role.TEACHER]


def test_splitting_preserves_the_total_duration():
    turns = [turn(0, 50, "T"), turn(50, 55, "S1"), turn(55, 63, "T")]
    out = split_segments_by_speaker([speech(0, 63)], turns)
    assert sum(s.duration for s in out) == pytest.approx(63.0)


def test_split_pieces_tile_without_gaps_or_overlap():
    turns = [turn(0, 20, "T"), turn(20, 30, "S1")]
    out = split_segments_by_speaker([speech(0, 30)], turns)
    for a, b in zip(out, out[1:]):
        assert a.end == pytest.approx(b.start)


def test_non_speech_passes_through_untouched():
    turns = [turn(0, 50, "T")]
    out = split_segments_by_speaker([speech(0, 50), handson(50, 300)], turns)
    assert out[-1].kind is SegmentKind.HANDSON
    assert out[-1].role is None


def test_a_stretch_no_speaker_covered_keeps_no_role():
    turns = [turn(0, 20, "T")]
    out = split_segments_by_speaker([speech(0, 40)], turns)
    assert out[0].role is Role.TEACHER
    assert out[-1].role is None


def test_tiny_slivers_are_absorbed_not_emitted():
    """A 0.05 s fragment is a boundary artefact, not a turn."""
    turns = [turn(0, 30, "T"), turn(30, 30.05, "S1"), turn(30.05, 60, "T")]
    out = split_segments_by_speaker([speech(0, 60)], turns)
    assert all(s.duration > 0.1 for s in out)


def test_a_segment_with_one_speaker_is_not_split():
    out = split_segments_by_speaker([speech(0, 30)], [turn(0, 30, "T")])
    assert len(out) == 1


def test_student_turns_survive_a_teacher_dominated_segment():
    """The exact failure: 57 s teacher + 3 s student came back 100% teacher."""
    turns = [turn(0, 57, "T"), turn(57, 60, "S1")]
    out = split_segments_by_speaker([speech(0, 60)], turns)
    students = [s for s in out if s.role is Role.STUDENT]
    assert students and students[0].duration == pytest.approx(3.0)


def test_split_segments_carry_confidence():
    out = split_segments_by_speaker([speech(0, 30)],
                                    [turn(0, 25, "T"), turn(25, 30, "S")])
    assert all(s.role_conf is not None for s in out if s.role)


def test_splitting_an_empty_timeline():
    assert split_segments_by_speaker([], [turn(0, 10, "T")]) == []


def test_splitting_with_no_turns_leaves_roles_unset():
    out = split_segments_by_speaker([speech(0, 30)], [])
    assert out[0].role is None


def test_talk_time_after_splitting_is_realistic():
    from src.attribution import talk_time

    turns = [turn(0, 57, "T"), turn(57, 60, "S1"), turn(60, 90, "T"), turn(90, 95, "S2")]
    out = split_segments_by_speaker([speech(0, 95)], turns)
    totals = talk_time(out)
    assert totals[Role.STUDENT] == pytest.approx(8.0)
    assert totals[Role.TEACHER] == pytest.approx(87.0)


# ================================================================ merging fragments
def test_adjacent_pieces_with_the_same_role_are_merged():
    """Measured: 185 speech pieces in 4 minutes, median 0.68 s, with runs like
    [7.46-7.96] student / [7.96-8.69] student left unmerged. That inflated M2 and M3 and
    collapsed M4 from 241 s to 16 s."""
    turns = [turn(0, 10, "T"), turn(10, 20, "T"), turn(20, 25, "S")]
    out = split_segments_by_speaker([speech(0, 25)], turns)
    roles = [s.role for s in out if s.role]
    assert roles == [Role.TEACHER, Role.STUDENT]


def test_merging_preserves_the_span():
    turns = [turn(0, 10, "T"), turn(10, 20, "T")]
    out = split_segments_by_speaker([speech(0, 20)], turns)
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(20.0)


def test_a_short_silent_gap_between_same_speaker_turns_is_absorbed():
    """A 0.3 s hole between two teacher turns is a breath, not a speaker change."""
    turns = [turn(0, 10, "T"), turn(10.3, 20, "T")]
    out = split_segments_by_speaker([speech(0, 20)], turns)
    assert len([s for s in out if s.role is Role.TEACHER]) == 1


def test_a_long_gap_is_not_absorbed():
    turns = [turn(0, 10, "T"), turn(18, 20, "T")]
    out = split_segments_by_speaker([speech(0, 20)], turns)
    assert any(s.role is None for s in out)


def test_a_real_speaker_change_is_never_merged_away():
    turns = [turn(0, 10, "T"), turn(10, 12, "S"), turn(12, 20, "T")]
    out = split_segments_by_speaker([speech(0, 20)], turns)
    assert [s.role for s in out if s.role] == [Role.TEACHER, Role.STUDENT, Role.TEACHER]


def test_merging_restores_a_long_teacher_stretch():
    """The M4 regression, directly."""
    from src.metrics import longest_stretch

    turns = [turn(i * 2.0, i * 2.0 + 2.0, "T") for i in range(100)]     # 200 s, fragmented
    out = split_segments_by_speaker([speech(0, 200)], turns)
    assert longest_stretch(out, Role.TEACHER) > 150


def test_merging_keeps_turn_counts_realistic():
    turns = ([turn(i * 3.0, i * 3.0 + 3.0, "T") for i in range(20)]
             + [turn(60, 63, "S"), turn(63, 90, "T")])
    out = split_segments_by_speaker([speech(0, 90)], turns)
    student_turns = [s for s in out if s.role is Role.STUDENT]
    assert len(student_turns) == 1


def test_pieces_below_the_minimum_turn_length_do_not_survive_alone():
    turns = [turn(0, 30, "T"), turn(30, 30.2, "S"), turn(30.2, 60, "T")]
    out = split_segments_by_speaker([speech(0, 60)], turns)
    assert all(s.duration >= 0.3 for s in out if s.role)


def test_the_timeline_still_tiles_after_merging():
    turns = [turn(0, 10, "T"), turn(10, 12, "S"), turn(12, 20, "T")]
    out = split_segments_by_speaker([speech(0, 20), handson(20, 60)], turns)
    for a, b in zip(out, out[1:]):
        assert a.end == pytest.approx(b.start)
    assert sum(s.duration for s in out) == pytest.approx(60.0)


# ======================================== overlapping turns: the enclosure problem
#
# pyannote turns overlap, and a teacher's turn routinely *encloses* a student's - on
# OD11163_2026-01-28 all 25 student turns sat inside one, on OD11165_2026-01-06 all 38.
# Both turns then overlap the student's span by exactly the same amount, so a strict `>`
# handed it to whichever came first in the list, which is always the enclosing turn.
# Result: 100% of student turns mislabelled, M1 reporting 100% teacher talk, and M3
# reporting zero exchanges - on sessions where M5 had just found 28 answered questions.

def test_a_student_speaking_inside_a_teacher_turn_is_not_credited_to_the_teacher():
    turns = [turn(0, 20, "T"), turn(5, 6, "S1")]
    assert speaker_for_span(turns, 5, 6) == "S1"


def test_the_enclosing_speaker_still_wins_the_span_they_actually_dominate():
    turns = [turn(0, 20, "T"), turn(5, 6, "S1")]
    assert speaker_for_span(turns, 0, 20) == "T"


def test_a_brief_interjection_does_not_take_a_span_it_barely_touches():
    turns = [turn(0, 20, "T"), turn(5, 6, "S1")]
    assert speaker_for_span(turns, 5.5, 15) == "T"


def test_a_student_holding_the_floor_beats_a_teacher_talking_over_them():
    """The tie-break must not simply favour whoever is shorter - a student with the
    floor and a teacher chipping in is the mirror image of the enclosure case."""
    turns = [turn(0, 30, "S1"), turn(10, 11, "T")]
    assert speaker_for_span(turns, 0, 30) == "S1"


def test_students_nested_inside_teacher_turns_reach_the_timeline():
    """The integration failure this actually caused: one long VAD segment, a teacher
    turn blanketing it, students speaking inside. Before the fix the timeline came back
    100% teacher and the students were simply gone."""
    segments = [speech(0, 60)]
    turns = [turn(0, 60, "T"),
             turn(10, 12, "S1"), turn(25, 27.5, "S2"), turn(40, 43, "S1")]

    out = split_segments_by_speaker(segments, turns)
    student_sec = sum(s.duration for s in out if s.role is Role.STUDENT)
    teacher_sec = sum(s.duration for s in out if s.role is Role.TEACHER)

    assert student_sec > 0, "students nested inside a teacher turn vanished"
    assert student_sec == pytest.approx(7.5, abs=0.6)
    assert teacher_sec > student_sec
    # And the timeline still tiles the original segment exactly.
    assert sum(s.duration for s in out) == pytest.approx(60.0)


def test_the_split_still_tiles_when_turns_are_nested():
    segments = [speech(0, 60)]
    turns = [turn(0, 60, "T"), turn(10, 12, "S1"), turn(40, 43, "S1")]
    out = sorted(split_segments_by_speaker(segments, turns), key=lambda s: s.start)
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(60.0)
    for a, b in zip(out, out[1:]):
        assert a.end == pytest.approx(b.start)


# ============================== diarization as the speech detector
#
# Silero was missing most of the speech on the two hands-on sessions, and the
# "loud but not speech" rule then claimed it as activity noise:
#
#   OD11166_2026-01-12: 142 s of 161 s of diarization turns landed in `handson`;
#                       of the student turns, 67 s of 70 s did.
#   OD11166_2026-01-20: 242 s of 279 s landed in `handson`; students 109 s of 118 s.
#
# Students who spoke for 67 s and 109 s were published as 1.6 s and 6.5 s. pyannote's
# segmentation is itself a strong VAD and the pipeline already pays for it, so where
# diarization ran, its turns are the speech map.

def test_speech_spans_are_the_union_of_every_speaker():
    turns = [turn(0, 10, "T"), turn(8, 15, "S1")]
    assert speech_spans(turns) == [(0.0, 15.0)]


def test_speech_spans_keep_separate_turns_apart():
    turns = [turn(0, 5, "T"), turn(9, 12, "S1")]
    assert speech_spans(turns) == [(0.0, 5.0), (9.0, 12.0)]


def test_speech_spans_merge_turns_that_merely_touch():
    turns = [turn(0, 5, "T"), turn(5, 9, "S1")]
    assert speech_spans(turns) == [(0.0, 9.0)]


def test_speech_spans_come_back_in_order_whatever_order_they_went_in():
    turns = [turn(20, 25, "S1"), turn(0, 5, "T"), turn(10, 12, "S2")]
    assert speech_spans(turns) == [(0.0, 5.0), (10.0, 12.0), (20.0, 25.0)]


def test_a_turn_nested_inside_another_adds_no_new_span():
    turns = [turn(0, 30, "T"), turn(10, 12, "S1"), turn(20, 21, "S2")]
    assert speech_spans(turns) == [(0.0, 30.0)]


def test_speech_spans_of_nothing():
    assert speech_spans([]) == []


# ================================================================ the turn cache
#
# Diarization is the most expensive thing in the project - 1.24x realtime, ~3 h for the
# corpus - and its output was never persisted. Every change to a downstream metric
# therefore cost a full re-run, which is exactly what happened when the question count
# turned out to be double-counting: two sessions of compute thrown away to fix arithmetic
# that runs in milliseconds.
#
# Keyed by audio CONTENT, like the transcript cache, so a renamed or re-copied file still
# hits and a re-encoded one correctly misses.

from src.diarization import TurnCache


def test_a_miss_returns_nothing(tmp_path, intact_session):
    cache = TurnCache(tmp_path / "turns")
    assert cache.get(intact_session / "OD90001_2026-01-06-114155.mp3") is None


def test_turns_round_trip_exactly(tmp_path, intact_session):
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    cache = TurnCache(tmp_path / "turns")
    turns = [SpeakerTurn(0.0, 2.5, "SPEAKER_00"),
             SpeakerTurn(2.1, 4.0, "SPEAKER_01"),      # overlapping, on purpose
             SpeakerTurn(4.0, 6.0, "SPEAKER_00")]
    cache.put(audio, turns)
    assert cache.get(audio) == turns


def test_the_key_follows_the_content_not_the_name(tmp_path, intact_session):
    """A copy under a different name is the same audio and must hit."""
    import shutil
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    twin = tmp_path / "renamed.mp3"
    shutil.copyfile(audio, twin)

    cache = TurnCache(tmp_path / "turns")
    cache.put(audio, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")])
    assert cache.get(twin) is not None


def test_different_audio_does_not_collide(tmp_path, intact_session):
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    other = tmp_path / "other.mp3"
    other.write_bytes(audio.read_bytes() + b"\x00trailing")

    cache = TurnCache(tmp_path / "turns")
    cache.put(audio, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")])
    assert cache.get(other) is None


def test_a_damaged_entry_is_a_miss_not_a_crash(tmp_path, intact_session):
    """Same rule as the transcript cache: a corrupt file costs a re-run, never a
    traceback in the middle of a three-hour batch."""
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    cache = TurnCache(tmp_path / "turns")
    cache.put(audio, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")])

    entry = next((tmp_path / "turns").glob("*.json"))
    entry.write_text("{ not json", encoding="utf-8")
    assert cache.get(audio) is None


def test_an_empty_result_is_cached_as_an_answer(tmp_path, intact_session):
    """"pyannote heard nobody" is a finding worth not recomputing. It must not be
    indistinguishable from a miss, or silent files re-run every single time."""
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    cache = TurnCache(tmp_path / "turns")
    cache.put(audio, [])
    assert cache.get(audio) == []


# ---------------------------------------------------------------- wired into diarize()

def test_diarize_uses_the_cache_instead_of_the_model(monkeypatch, tmp_path, intact_session):
    from src import diarization

    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    monkeypatch.setattr(diarization.config, "DIARIZATION_CACHE_DIR", tmp_path / "turns")
    TurnCache(tmp_path / "turns").put(audio, [SpeakerTurn(0.0, 3.0, "SPEAKER_07")])

    def explode(token):
        raise AssertionError("the pipeline must not load on a cache hit")

    monkeypatch.setattr(diarization, "_load_pipeline", explode)
    assert diarization.diarize(audio, token="fake") == [SpeakerTurn(0.0, 3.0, "SPEAKER_07")]


def test_a_cache_hit_needs_no_token(monkeypatch, tmp_path, intact_session):
    """The token is for the gated DOWNLOAD. Once the turns are on disk, reproducing a
    result needs neither the token nor the weights - which is the difference between a
    reviewer being able to re-run this and not."""
    from src import diarization

    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    monkeypatch.setattr(diarization.config, "DIARIZATION_CACHE_DIR", tmp_path / "turns")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    TurnCache(tmp_path / "turns").put(audio, [SpeakerTurn(0.0, 3.0, "SPEAKER_00")])

    assert len(diarization.diarize(audio)) == 1


def test_diarize_fills_the_cache_after_a_real_run(monkeypatch, tmp_path, intact_session):
    from src import diarization

    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    monkeypatch.setattr(diarization.config, "DIARIZATION_CACHE_DIR", tmp_path / "turns")

    class FakePipeline:
        def __call__(self, audio, **kwargs):
            from types import SimpleNamespace

            class Ann:
                def itertracks(self, yield_label=False):
                    yield SimpleNamespace(start=0.0, end=2.0), None, "SPEAKER_00"
            return Ann()

    monkeypatch.setattr(diarization, "_load_pipeline", lambda token: FakePipeline())
    first = diarization.diarize(audio, token="fake")

    monkeypatch.setattr(diarization, "_load_pipeline",
                        lambda token: (_ for _ in ()).throw(AssertionError("re-ran")))
    assert diarization.diarize(audio, token="fake") == first


def test_num_speakers_is_part_of_the_key(monkeypatch, tmp_path, intact_session):
    """Asking for a fixed speaker count is a different question about the same audio."""
    audio = intact_session / "OD90001_2026-01-06-114155.mp3"
    cache = TurnCache(tmp_path / "turns")
    cache.put(audio, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")], num_speakers=2)
    assert cache.get(audio, num_speakers=2) is not None
    assert cache.get(audio, num_speakers=3) is None
    assert cache.get(audio) is None
