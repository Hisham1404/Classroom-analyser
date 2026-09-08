"""Stage 05 — M1 to M6.

Four acoustic metrics that survive a broken transcript, one lexical metric that does not,
and a confidence score that decides how much of the fragile half anyone is shown.

The assignment grades these on *formula, explanation, interpretation* — so the numbers have
to be defensible, and the ones that cannot be trusted have to say so rather than quietly
returning something plausible.
"""

from __future__ import annotations

import pytest

from src import config
from src.asr.base import Transcript, TranscriptSegment
from src.attribution import Role, assign_roles
from src.metrics import (
    Verdict,
    compute_metrics,
    is_question,
    longest_stretch,
    text_for_span,
)
from src.models import Roster
from src.signal_layer import Segment, SegmentKind


def speech(start, end, role=None, conf=0.8, rms=-20.0):
    return Segment(start, end, SegmentKind.SPEECH, rms, 0.2, role, conf)


def handson(start, end):
    return Segment(start, end, SegmentKind.HANDSON, -30.0, 0.6)


def dead(start, end):
    return Segment(start, end, SegmentKind.DEAD, -60.0, 0.5)


def roster(total=20, boys=10, girls=10):
    return Roster(total=total, boys=boys, girls=girls)


def transcript(pieces, audio_sec=600.0):
    """pieces: list of (start, end, text)."""
    return Transcript(
        segments=[TranscriptSegment(s, e, t, -0.4, 0.1, None) for s, e, t in pieces],
        language="hi", language_prob=0.95, backend="stub", model="m", audio_sec=audio_sec,
    )


# ================================================================ M1 teacher talk ratio
def test_m1_is_teacher_share_of_spoken_time():
    segs = [speech(0, 60, Role.TEACHER), speech(60, 100, Role.STUDENT)]
    m = compute_metrics(segs, roster(), audio_sec=100.0)
    assert m["M1_teacher_talk_ratio"].value == pytest.approx(0.6)


def test_m1_ignores_non_speech():
    """Hands-on noise must not dilute the ratio — it has no speaker."""
    segs = [speech(0, 60, Role.TEACHER), speech(60, 100, Role.STUDENT), handson(100, 500)]
    m = compute_metrics(segs, roster(), audio_sec=500.0)
    assert m["M1_teacher_talk_ratio"].value == pytest.approx(0.6)


def test_m1_is_none_when_nobody_spoke():
    m = compute_metrics([handson(0, 300)], roster(), audio_sec=300.0)
    assert m["M1_teacher_talk_ratio"].value is None


def test_m1_carries_the_maker_session_benchmark():
    m = compute_metrics([speech(0, 60, Role.TEACHER)], roster(), audio_sec=60.0)
    assert m["M1_teacher_talk_ratio"].benchmark == pytest.approx(config.TEACHER_TALK_BENCHMARK)


def test_m1_bands_a_dominant_teacher_as_high():
    segs = [speech(0, 90, Role.TEACHER), speech(90, 100, Role.STUDENT)]
    assert compute_metrics(segs, roster(), audio_sec=100.0)["M1_teacher_talk_ratio"].band == "high"


def test_m1_bands_a_student_led_session_as_low():
    segs = [speech(0, 10, Role.TEACHER), speech(10, 100, Role.STUDENT)]
    assert compute_metrics(segs, roster(), audio_sec=100.0)["M1_teacher_talk_ratio"].band == "low"


# ================================================================ M2 participation
def test_m2_is_student_turns_per_child_per_hour():
    segs = [speech(i * 10, i * 10 + 5, Role.STUDENT) for i in range(10)]
    m = compute_metrics(segs, roster(total=20), audio_sec=1800.0)
    # 10 turns / 20 children over half an hour -> 1.0 per child per hour
    assert m["M2_student_participation"].value == pytest.approx(1.0)


def test_m2_normalises_by_class_size():
    """8 students and 27 students cannot be compared on raw turn counts."""
    segs = [speech(i * 10, i * 10 + 5, Role.STUDENT) for i in range(10)]
    small = compute_metrics(segs, roster(total=8), audio_sec=1800.0)
    large = compute_metrics(segs, roster(total=27), audio_sec=1800.0)
    assert small["M2_student_participation"].value > large["M2_student_participation"].value


def test_m2_is_none_without_a_roster():
    """Their app sometimes records no headcount. Better to say nothing than to guess."""
    segs = [speech(0, 5, Role.STUDENT)]
    m = compute_metrics(segs, Roster(total=0), audio_sec=600.0)
    assert m["M2_student_participation"].value is None


def test_m2_is_zero_when_no_student_spoke():
    m = compute_metrics([speech(0, 60, Role.TEACHER)], roster(), audio_sec=600.0)
    assert m["M2_student_participation"].value == 0.0


# ================================================================ M3 interaction density
def test_m3_counts_floor_changes_per_minute():
    segs = [speech(0, 10, Role.TEACHER), speech(10, 20, Role.STUDENT),
            speech(20, 30, Role.TEACHER), speech(30, 40, Role.STUDENT)]
    m = compute_metrics(segs, roster(), audio_sec=60.0)
    assert m["M3_interaction_density"].value == pytest.approx(3.0)     # 3 switches / 1 min


def test_m3_ignores_same_speaker_continuations():
    segs = [speech(0, 10, Role.TEACHER), speech(10, 20, Role.TEACHER),
            speech(20, 30, Role.STUDENT)]
    m = compute_metrics(segs, roster(), audio_sec=60.0)
    assert m["M3_interaction_density"].value == pytest.approx(1.0)


def test_m3_is_not_broken_by_hands_on_between_turns():
    segs = [speech(0, 10, Role.TEACHER), handson(10, 200), speech(200, 210, Role.STUDENT)]
    m = compute_metrics(segs, roster(), audio_sec=600.0)
    assert m["M3_interaction_density"].value > 0


def test_m3_of_a_monologue_is_zero():
    m = compute_metrics([speech(0, 300, Role.TEACHER)], roster(), audio_sec=600.0)
    assert m["M3_interaction_density"].value == 0.0


# ================================================================ M4 longest stretch
def test_m4_finds_the_longest_unbroken_teacher_run():
    segs = [speech(0, 100, Role.TEACHER), speech(100, 110, Role.STUDENT),
            speech(110, 400, Role.TEACHER)]
    m = compute_metrics(segs, roster(), audio_sec=600.0)
    assert m["M4_longest_teacher_stretch"].value == pytest.approx(290.0)


def test_m4_merges_across_a_short_breath():
    """A two-second pause is the same stretch of talking, not two."""
    segs = [speech(0, 100, Role.TEACHER), dead(100, 102), speech(102, 200, Role.TEACHER)]
    assert longest_stretch(segs, Role.TEACHER) == pytest.approx(200.0)


def test_m4_does_not_merge_across_a_long_gap():
    segs = [speech(0, 100, Role.TEACHER), handson(100, 300), speech(300, 380, Role.TEACHER)]
    assert longest_stretch(segs, Role.TEACHER) == pytest.approx(100.0)


def test_m4_does_not_merge_across_another_speaker():
    segs = [speech(0, 100, Role.TEACHER), speech(100, 101, Role.STUDENT),
            speech(101, 150, Role.TEACHER)]
    assert longest_stretch(segs, Role.TEACHER) == pytest.approx(100.0)


def test_m4_is_zero_when_the_teacher_never_spoke():
    assert longest_stretch([speech(0, 50, Role.STUDENT)], Role.TEACHER) == 0.0


# ================================================================ questions
def test_a_hindi_question_word_makes_it_a_question():
    assert is_question("यह क्या है")
    assert is_question("कौन बताएगा")
    assert is_question("कैसे बनाया")


def test_a_question_mark_makes_it_a_question():
    assert is_question("यह ठीक है?")


def test_a_plain_statement_is_not_a_question():
    assert not is_question("यह कागज़ है")


def test_empty_text_is_not_a_question():
    assert not is_question("")
    assert not is_question("   ")


def test_question_detection_ignores_a_word_appearing_mid_compound():
    """'क्या' inside a longer token should not trigger on its own."""
    assert not is_question("कयामत")


# ================================================================ text alignment
def test_text_for_span_picks_overlapping_transcript_segments():
    t = transcript([(0, 5, "पहला"), (10, 15, "दूसरा"), (20, 25, "तीसरा")])
    assert text_for_span(t, 9, 16) == "दूसरा"


def test_text_for_span_joins_multiple_overlaps():
    t = transcript([(0, 5, "एक"), (5, 10, "दो")])
    assert text_for_span(t, 0, 10) == "एक दो"


def test_text_for_span_with_no_overlap_is_empty():
    t = transcript([(0, 5, "एक")])
    assert text_for_span(t, 50, 60) == ""


def test_text_for_span_without_a_transcript_is_empty():
    assert text_for_span(None, 0, 10) == ""


# ================================================================ M5 wait time
#
# M5 needs config.WAIT_TIME_MIN_SAMPLES question-answer pairs before it will report a
# median, so these build a lesson of several exchanges rather than one.

def lesson(gaps, question=True, audio_sec=600.0):
    """A teacher question every 30 s, each answered after the given gap."""
    segs, pieces = [], []
    text = "यह क्या है" if question else "यह कागज़ है"
    for i, gap in enumerate(gaps):
        start = i * 30.0
        segs.append(speech(start, start + 10, Role.TEACHER))
        segs.append(speech(start + 10 + gap, start + 14 + gap, Role.STUDENT))
        pieces.append((start, start + 10, text))
    return segs, transcript(pieces, audio_sec=audio_sec)


def test_m5_measures_the_pause_after_a_teacher_question():
    segs, t = lesson([4.0, 4.0, 4.0])
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].value == pytest.approx(4.0)


def test_m5_takes_the_median_across_questions():
    segs, t = lesson([1.0, 3.0, 5.0])
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].value == pytest.approx(3.0)


def test_m5_ignores_teacher_statements():
    segs, t = lesson([4.0, 4.0, 4.0], question=False)
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].value is None


def test_m5_is_none_without_a_transcript():
    """The whole point of the acoustic/lexical split — no transcript, no wait time."""
    segs, _ = lesson([4.0, 4.0, 4.0])
    m = compute_metrics(segs, roster(), audio_sec=120.0)
    assert m["M5_wait_time_1"].value is None


def test_m5_bands_a_short_pause_as_low():
    segs, t = lesson([0.5, 0.5, 0.5])
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].band == "low"


def test_m5_bands_three_seconds_or_more_as_normal():
    segs, t = lesson([3.5, 3.5, 3.5])
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].band in {"normal", "high"}


# ================================================================ M6 confidence
def test_m6_is_high_for_a_clean_session():
    segs = assign_roles([speech(0, 200, rms=-15), speech(200, 260, rms=-40),
                         speech(260, 400, rms=-16), speech(400, 460, rms=-41)])
    t = transcript([(0, 400, "यह कागज़ है " * 40)], audio_sec=600.0)
    m = compute_metrics(segs, roster(), audio_sec=600.0, transcript=t, completeness=1.0)
    assert m.confidence > config.CONF_USABLE
    assert m.verdict is Verdict.USABLE


def test_m6_is_unreliable_when_almost_nothing_is_speech():
    """The Shadow Art case: 20% speech density, loudest file of the five."""
    segs = [speech(0, 60, Role.TEACHER, conf=0.1), handson(60, 600)]
    m = compute_metrics(segs, roster(), audio_sec=600.0, completeness=1.0)
    assert m.verdict is Verdict.UNRELIABLE


def test_m6_is_dragged_down_by_a_truncated_recording():
    segs = assign_roles([speech(0, 200, rms=-15), speech(200, 260, rms=-40)])
    whole = compute_metrics(segs, roster(), audio_sec=600.0, completeness=1.0)
    cut = compute_metrics(segs, roster(), audio_sec=600.0, completeness=0.34)
    assert cut.confidence < whole.confidence


def test_m6_is_dragged_down_by_a_weak_speaker_split():
    strong = assign_roles([speech(0, 200, rms=-15), speech(200, 400, rms=-45)])
    weak = assign_roles([speech(0, 200, rms=-25), speech(200, 400, rms=-25.1)])
    a = compute_metrics(strong, roster(), audio_sec=600.0, completeness=1.0)
    b = compute_metrics(weak, roster(), audio_sec=600.0, completeness=1.0)
    assert b.confidence < a.confidence


def test_m6_stays_within_bounds():
    for segs in ([], [handson(0, 600)], [speech(0, 600, Role.TEACHER, conf=1.0)]):
        m = compute_metrics(segs, roster(), audio_sec=600.0)
        assert 0.0 <= m.confidence <= 1.0


def test_an_unreliable_session_suppresses_the_lexical_metric():
    """Below the floor, wait time must not be reported at all."""
    segs = [speech(0, 10, Role.TEACHER, conf=0.02), speech(14, 18, Role.STUDENT, conf=0.02),
            handson(18, 600)]
    t = transcript([(0, 10, "यह क्या है")])
    m = compute_metrics(segs, roster(), audio_sec=600.0, transcript=t)
    assert m.verdict is Verdict.UNRELIABLE
    assert m["M5_wait_time_1"].value is None


# ================================================================ output shape
def test_every_metric_carries_formula_and_interpretation():
    """The assignment grades exactly this."""
    m = compute_metrics([speech(0, 60, Role.TEACHER)], roster(), audio_sec=60.0)
    for metric in m.metrics.values():
        assert metric.formula
        assert metric.explanation
        assert metric.interpretation


def test_metrics_serialise_to_json():
    import json
    m = compute_metrics([speech(0, 60, Role.TEACHER)], roster(), audio_sec=60.0)
    json.loads(json.dumps(m.to_dict()))


def test_totals_account_for_the_whole_recording():
    segs = [speech(0, 60, Role.TEACHER), handson(60, 500), dead(500, 600)]
    m = compute_metrics(segs, roster(), audio_sec=600.0)
    assert sum(m.totals.values()) == pytest.approx(600.0)


def test_all_six_metrics_are_always_present():
    """A metric that could not be computed reports None — it never vanishes."""
    m = compute_metrics([], roster(), audio_sec=600.0)
    assert set(m.metrics) == {
        "M1_teacher_talk_ratio", "M2_student_participation", "M3_interaction_density",
        "M4_longest_teacher_stretch", "M5_wait_time_1",
    }
    assert all(v.value is None or v.value == 0.0 for v in m.metrics.values())


# ================================================================ attribution gating
def test_metrics_that_need_the_speaker_split_are_suppressed_when_it_is_a_guess():
    """Measured on OD11163_2025-12-23: every segment between -27.1 and -29.7 dB, a 2.6 dB
    spread with nothing to cluster on. The split was noise and every role confidence came
    back under 0.15 — but M1 still reported a confident-looking 45%."""
    segs = [speech(0, 60, Role.STUDENT, conf=0.14, rms=-29.2),
            speech(60, 90, Role.TEACHER, conf=0.04, rms=-28.1),
            speech(90, 150, Role.TEACHER, conf=0.12, rms=-27.6)]
    m = compute_metrics(segs, roster(), audio_sec=300.0)

    for key in ("M1_teacher_talk_ratio", "M2_student_participation",
                "M3_interaction_density", "M4_longest_teacher_stretch"):
        assert m[key].value is None, key


def test_a_confident_split_still_reports_those_metrics():
    segs = [speech(0, 60, Role.TEACHER, conf=0.8, rms=-15.0),
            speech(60, 90, Role.STUDENT, conf=0.7, rms=-40.0)]
    m = compute_metrics(segs, roster(), audio_sec=300.0)
    assert m["M1_teacher_talk_ratio"].value is not None


def test_suppressed_metrics_say_why():
    segs = [speech(0, 60, Role.STUDENT, conf=0.05), speech(60, 90, Role.TEACHER, conf=0.05)]
    m = compute_metrics(segs, roster(), audio_sec=300.0)
    assert m["M1_teacher_talk_ratio"].band == "unknown"
    assert "split" in m["M1_teacher_talk_ratio"].interpretation.lower() or \
           m["M1_teacher_talk_ratio"].confidence < config.ROLE_CONF_FLOOR


def test_metrics_that_do_not_need_the_split_survive_a_weak_one():
    """Total speech, hands-on and dead time never depended on who was talking."""
    segs = [speech(0, 60, Role.STUDENT, conf=0.05), handson(60, 200), dead(200, 300)]
    m = compute_metrics(segs, roster(), audio_sec=300.0)
    assert m.totals["handson"] == pytest.approx(140.0)
    assert m.totals["dead"] == pytest.approx(100.0)


def test_attribution_confidence_is_reported_on_the_affected_metrics():
    segs = [speech(0, 60, Role.TEACHER, conf=0.9, rms=-15.0),
            speech(60, 90, Role.STUDENT, conf=0.9, rms=-40.0)]
    m = compute_metrics(segs, roster(), audio_sec=300.0)
    assert m["M1_teacher_talk_ratio"].confidence <= m.confidence + 1e-9


# ================================================================ M5 from diarization turns
from src.diarization import SpeakerTurn


def dturn(start, end, speaker):
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def test_wait_time_from_turns_measures_the_real_silence():
    """The bug: splitting makes segments tile exactly, so a student turn begins the instant
    the teacher's ends and the gap is 0.00 by construction. Diarization turns have real
    silences between them."""
    segs = [speech(0, 10, Role.TEACHER), speech(10, 15, Role.STUDENT)]   # tiled, gap 0
    turns = [dturn(0, 8.5, "T"), dturn(12.0, 15, "S1"),                  # real 3.5 s gaps
             dturn(30, 38.5, "T"), dturn(42.0, 45, "S2"),
             dturn(60, 68.5, "T"), dturn(72.0, 75, "S3")]
    t = transcript([(0, 10, "यह क्या है"), (30, 38.5, "यह क्या है"),
                    (60, 68.5, "यह क्या है")])

    m = compute_metrics(segs, roster(), audio_sec=90.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value == pytest.approx(3.5)


def test_tiled_segments_alone_no_longer_report_a_zero_wait():
    """Without turns there is no silence to measure, so say nothing rather than 0.00."""
    segs = [speech(0, 10, Role.TEACHER), speech(10, 15, Role.STUDENT)]
    t = transcript([(0, 10, "यह क्या है")])
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t)
    assert m["M5_wait_time_1"].value is None


def test_a_genuine_gap_between_segments_is_still_usable_without_turns():
    """The energy path leaves real gaps, so it can still measure wait time."""
    segs, t = lesson([3.5, 3.5, 3.5])
    m = compute_metrics(segs, roster(), audio_sec=120.0, transcript=t)
    assert m["M5_wait_time_1"].value == pytest.approx(3.5)


def test_wait_time_from_turns_takes_the_median():
    segs = [speech(0, 30, Role.TEACHER), speech(30, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(12, 14, "S1"),          # 2 s
             dturn(20, 28, "T"), dturn(31, 36, "S2"),         # 3 s
             dturn(40, 48, "T"), dturn(52, 56, "S3")]         # 4 s
    t = transcript([(0, 10, "यह क्या है"), (20, 28, "कौन बताएगा"),
                    (40, 48, "यह क्या है")])
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value == pytest.approx(3.0)


def test_only_questions_start_the_clock():
    segs = [speech(0, 30, Role.TEACHER), speech(30, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(14, 18, "S1")]
    t = transcript([(0, 10, "यह कागज़ है")])          # statement, not a question
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value is None


def test_a_student_answering_much_later_is_a_new_exchange():
    segs = [speech(0, 60, Role.TEACHER), speech(60, 90, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(45, 50, "S1")]     # 35 s later
    t = transcript([(0, 10, "यह क्या है")])
    m = compute_metrics(segs, roster(), audio_sec=90.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value is None


def test_turns_from_the_same_speaker_do_not_count_as_an_answer():
    segs = [speech(0, 30, Role.TEACHER), speech(30, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(13, 20, "T")]      # teacher continues
    t = transcript([(0, 10, "यह क्या है")])
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value is None


def test_wait_time_needs_a_transcript_even_with_turns():
    segs = [speech(0, 30, Role.TEACHER), speech(30, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(13, 18, "S1")]
    m = compute_metrics(segs, roster(), audio_sec=60.0, speaker_turns=turns)
    assert m["M5_wait_time_1"].value is None


def test_wait_time_survives_turns_with_no_teacher():
    m = compute_metrics([speech(0, 30, Role.TEACHER)], roster(), audio_sec=60.0,
                        transcript=transcript([(0, 10, "यह क्या है")]), speaker_turns=[])
    assert m["M5_wait_time_1"].value is None


# ============================================ M5: pairing a question with its answer
#
# Three defects found by running the real turns through the old pairing rule, on
# OD11166_2026-01-20-115538 (157 diarization turns in four minutes):
#
#   1. It walked the turn list in pairs, so a question was thrown away whenever the
#      teacher happened to speak again before the student answered. 1 of 3 questions
#      survived on that session, 5 of 8 on the next.
#   2. Diarization turns overlap. Pairing by adjacency produced gaps as negative as
#      -11.0 s - a student turn nested inside a long teacher turn - and those were
#      silently dropped instead of being read as "did not wait at all".
#   3. There was no minimum sample count, so the published 0.00 s rested on one pair.
#
# The helper is exercised directly here: it returns the list, and the median and the
# sample floor are separate decisions tested at the metric level.

from src.metrics import wait_times_from_turns


def q_transcript(spans, audio_sec=600.0):
    """A transcript where each given span reads as a question."""
    return transcript([(a, b, "यह क्या है") for a, b in spans], audio_sec=audio_sec)


def test_a_question_still_counts_when_the_teacher_speaks_again_first():
    """The old rule looked only at the very next turn, so a teacher who carried on
    talking for a moment erased the question entirely - the common case in a room
    where one adult holds the floor."""
    turns = [dturn(0, 10, "T"), dturn(11, 13, "T"), dturn(16, 20, "S1")]
    waits = wait_times_from_turns(turns, "T", q_transcript([(0, 10)]))
    assert waits == pytest.approx([6.0])


def test_a_turn_nested_inside_the_question_does_not_swallow_the_answer():
    """A student murmuring in the middle of a long question is not the answer to it.
    Pairing by adjacency gave this a gap of -15 s and then discarded it."""
    turns = [dturn(0, 20, "T"), dturn(5, 7, "S1"), dturn(23, 28, "S2")]
    waits = wait_times_from_turns(turns, "T", q_transcript([(0, 20)]))
    assert waits == pytest.approx([3.0])


def test_a_student_talking_over_the_end_of_the_question_waited_no_time():
    """An interruption is a real observation of zero wait, not missing data."""
    turns = [dturn(0, 20, "T"), dturn(18, 25, "S1")]
    waits = wait_times_from_turns(turns, "T", q_transcript([(0, 20)]))
    assert waits == pytest.approx([0.0])


def test_a_gap_of_exactly_zero_is_a_snapped_boundary_not_a_measurement():
    """Real wait times never land on exactly 0.000. Three of eleven samples on the
    real sessions did, which is the signature of two turns sharing a boundary - no
    silence was resolved there, so there is nothing to report."""
    turns = [dturn(0, 10, "T"), dturn(10, 15, "S1")]
    assert wait_times_from_turns(turns, "T", q_transcript([(0, 10)])) == []


def test_every_question_gets_its_own_sample():
    turns = [dturn(0, 10, "T"), dturn(12, 14, "S1"),
             dturn(20, 30, "T"), dturn(33, 36, "S2"),
             dturn(40, 50, "T"), dturn(51, 55, "S3")]
    waits = wait_times_from_turns(turns, "T", q_transcript([(0, 10), (20, 30), (40, 50)]))
    assert sorted(waits) == pytest.approx([1.0, 2.0, 3.0])


def test_an_answer_beyond_the_window_is_a_new_exchange_not_a_long_wait():
    turns = [dturn(0, 10, "T"), dturn(10 + config.WAIT_TIME_MAX_SEC + 1, 60, "S1")]
    assert wait_times_from_turns(turns, "T", q_transcript([(0, 10)])) == []


def test_a_question_nobody_answers_contributes_nothing():
    turns = [dturn(0, 10, "T"), dturn(12, 20, "T")]
    assert wait_times_from_turns(turns, "T", q_transcript([(0, 10)])) == []


# --------------------------------------------------------------- the sample floor

def three_questions_and_answers():
    """Three clean question-answer pairs with waits of 1, 2 and 3 seconds."""
    segs = [speech(0, 50, Role.TEACHER), speech(50, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(12, 14, "S1"),
             dturn(20, 30, "T"), dturn(33, 36, "S2"),
             dturn(40, 50, "T"), dturn(51, 55, "S3")]
    t = q_transcript([(0, 10), (20, 30), (40, 50)])
    return segs, turns, t


def test_the_median_is_reported_once_there_are_enough_samples():
    segs, turns, t = three_questions_and_answers()
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t, speaker_turns=turns)
    assert m["M5_wait_time_1"].value == pytest.approx(2.0)


def test_too_few_samples_is_withheld_rather_than_published():
    """A median over one pair is not a median. The real session published 0.00 s from
    a single sample, which is exactly the kind of number this project refuses."""
    segs = [speech(0, 50, Role.TEACHER), speech(50, 60, Role.STUDENT)]
    turns = [dturn(0, 10, "T"), dturn(12, 14, "S1")]
    m = compute_metrics(segs, roster(), audio_sec=60.0,
                        transcript=q_transcript([(0, 10)]), speaker_turns=turns)
    assert m["M5_wait_time_1"].value is None


def test_the_sample_count_is_stated_so_the_number_can_be_judged():
    """Two seconds from three pairs and two seconds from thirty are different claims."""
    segs, turns, t = three_questions_and_answers()
    m = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t, speaker_turns=turns)
    assert "3 question-answer pairs" in m["M5_wait_time_1"].interpretation


def test_confidence_falls_as_the_sample_count_approaches_the_floor():
    segs, turns, t = three_questions_and_answers()
    thin = compute_metrics(segs, roster(), audio_sec=60.0, transcript=t,
                           speaker_turns=turns)["M5_wait_time_1"]

    many_turns = list(turns)
    many_spans = [(0, 10), (20, 30), (40, 50)]
    for i in range(6):
        start = 60 + i * 20
        many_turns += [dturn(start, start + 10, "T"), dturn(start + 12, start + 14, "S1")]
        many_spans.append((start, start + 10))
    thick = compute_metrics([speech(0, 200, Role.TEACHER), speech(200, 220, Role.STUDENT)],
                            roster(), audio_sec=240.0,
                            transcript=q_transcript(many_spans),
                            speaker_turns=many_turns)["M5_wait_time_1"]

    assert thick.confidence > thin.confidence


def test_totals_are_keyed_by_plain_role_names():
    """`str(Role.TEACHER)` is "Role.TEACHER", which leaked the Python enum into the
    published JSON. The timeline already uses "teacher"/"student"; totals must match,
    or a reader has to know what a Python enum repr looks like to find the number."""
    segs = [speech(0, 60, Role.TEACHER), speech(60, 90, Role.STUDENT), handson(90, 120)]
    payload = compute_metrics(segs, roster(), audio_sec=120.0).to_dict()
    assert set(payload["totals_sec"]) >= {"teacher", "student", "handson"}
    assert not any(key.startswith("Role.") for key in payload["totals_sec"])
    assert payload["totals_sec"]["teacher"] == pytest.approx(60.0)
    assert payload["totals_sec"]["student"] == pytest.approx(30.0)
