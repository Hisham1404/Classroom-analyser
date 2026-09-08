"""Stage 04 — who was speaking, teacher or student.

Binary, not per-child. 27 children on one phone mic is not a solvable identification
problem, and pretending otherwise is how a demo falls apart under questioning.

The prior that makes this tractable: **the recording is made on the teacher's own phone**,
so she is consistently the nearest, loudest, highest-SNR voice in the room.
"""

from __future__ import annotations

import pytest

from src import config
from src.attribution import Role, assign_roles, talk_time
from src.signal_layer import Segment, SegmentKind


def speech(start, end, rms_db, flatness=0.2):
    return Segment(start=start, end=end, kind=SegmentKind.SPEECH,
                   rms_db=rms_db, flatness=flatness)


def handson(start, end, rms_db=-30.0):
    return Segment(start=start, end=end, kind=SegmentKind.HANDSON,
                   rms_db=rms_db, flatness=0.6)


# ================================================================ the split
def test_the_louder_cluster_is_the_teacher():
    """The phone is hers, so she is the nearest and loudest voice."""
    segs = [speech(0, 8, -18), speech(8, 10, -34),
            speech(10, 18, -19), speech(18, 20, -33)]
    roles = [s.role for s in assign_roles(segs)]
    assert roles == [Role.TEACHER, Role.STUDENT, Role.TEACHER, Role.STUDENT]


def test_non_speech_segments_are_left_unassigned():
    segs = [speech(0, 5, -18), handson(5, 40), speech(40, 45, -34)]
    out = assign_roles(segs)
    assert out[1].role is None


def test_a_clear_split_reports_high_confidence():
    segs = [speech(0, 8, -15), speech(8, 12, -45),
            speech(12, 20, -15), speech(20, 24, -45)]
    assert all(s.role_conf > 0.5 for s in assign_roles(segs) if s.role)


def test_an_ambiguous_split_reports_low_confidence():
    """Everyone the same loudness — the split is a guess and must say so."""
    segs = [speech(0, 5, -25), speech(5, 10, -25.2),
            speech(10, 15, -24.9), speech(15, 20, -25.1)]
    assert all(s.role_conf < config.ROLE_CONF_FLOOR for s in assign_roles(segs) if s.role)


def test_confidence_is_bounded():
    segs = [speech(0, 5, -5), speech(5, 10, -70), speech(10, 15, -6)]
    assert all(0.0 <= s.role_conf <= 1.0 for s in assign_roles(segs) if s.role)


# ================================================================ edge cases
def test_a_single_speech_segment_is_the_teacher_with_low_confidence():
    """One voice and nothing to compare it against — assume the phone's owner, say so."""
    out = assign_roles([speech(0, 10, -20)])
    assert out[0].role is Role.TEACHER
    assert out[0].role_conf < config.ROLE_CONF_FLOOR


def test_no_speech_segments_at_all():
    out = assign_roles([handson(0, 30)])
    assert all(s.role is None for s in out)


def test_empty_input():
    assert assign_roles([]) == []


def test_identical_segments_do_not_crash():
    out = assign_roles([speech(0, 5, -25), speech(5, 10, -25)])
    assert all(s.role is not None for s in out)


def test_assignment_is_deterministic():
    segs = [speech(0, 8, -18), speech(8, 10, -34), speech(10, 18, -19)]
    assert [s.role for s in assign_roles(segs)] == [s.role for s in assign_roles(segs)]


def test_the_original_segments_are_not_mutated():
    segs = [speech(0, 8, -18), speech(8, 10, -34)]
    assign_roles(segs)
    assert all(s.role is None for s in segs)


# ================================================================ talk time
def test_talk_time_totals_by_role():
    segs = assign_roles([speech(0, 10, -18), speech(10, 14, -35), speech(14, 20, -19)])
    totals = talk_time(segs)
    assert totals[Role.TEACHER] == pytest.approx(16.0)
    assert totals[Role.STUDENT] == pytest.approx(4.0)


def test_talk_time_counts_hands_on_and_dead_separately():
    segs = assign_roles([speech(0, 10, -18), handson(10, 50),
                         Segment(50, 60, SegmentKind.DEAD, -60.0, 0.5)])
    totals = talk_time(segs)
    assert totals["handson"] == pytest.approx(40.0)
    assert totals["dead"] == pytest.approx(10.0)


def test_talk_time_of_nothing_is_all_zero():
    totals = talk_time([])
    assert set(totals.values()) == {0.0}


def test_talk_time_never_double_counts():
    segs = assign_roles([speech(0, 10, -18), speech(10, 20, -35), handson(20, 30)])
    assert sum(talk_time(segs).values()) == pytest.approx(30.0)


# ================================================================ the realistic shape
def test_a_teacher_led_lesson_splits_sensibly():
    """Long loud teacher turns, short quiet student answers — the common pattern."""
    segs = []
    t = 0.0
    for _ in range(6):
        segs.append(speech(t, t + 12, -17))       # teacher explains
        t += 12
        segs.append(speech(t, t + 3, -33))        # student answers
        t += 3

    out = assign_roles(segs)
    totals = talk_time(out)
    assert totals[Role.TEACHER] > totals[Role.STUDENT]
    assert totals[Role.TEACHER] == pytest.approx(72.0)


def test_a_hands_on_session_can_have_more_student_talk():
    """Nothing in the design forces the teacher to dominate — a maker session
    where children talk more must not be silently relabelled."""
    segs = [speech(0, 5, -16)] + [speech(5 + i * 4, 9 + i * 4, -32) for i in range(8)]
    totals = talk_time(assign_roles(segs))
    assert totals[Role.STUDENT] > totals[Role.TEACHER]
