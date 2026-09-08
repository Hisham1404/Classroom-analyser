"""Stage 06 — turning metrics into something a teacher can act on.

Santana teaches 22 children in a government school in Igatpuri. She has a phone, patchy
data, about five minutes, and no interest in machine learning. MakerGhat already reaches
her through MakerDost on WhatsApp, so the dashboard is the M&E team's surface — hers is a
message.

Rule-based, not an LLM. Three reasons, and the third is the one that matters:
deterministic output is explainable, it runs offline, and **it structurally cannot invent
a statistic**. On a project where the confidence gate exists to stop confident wrong
numbers reaching a teacher, a generative layer writing the headline would undo the point.
"""

from __future__ import annotations

import pytest

from src import config
from src.attribution import Role
from src.insight import build_insight, whatsapp_message
from src.metrics import Verdict, compute_metrics
from src.models import Roster
from src.signal_layer import Segment, SegmentKind


def speech(start, end, role=None, conf=0.8, rms=-20.0):
    return Segment(start, end, SegmentKind.SPEECH, rms, 0.2, role, conf)


def handson(start, end):
    return Segment(start, end, SegmentKind.HANDSON, -30.0, 0.6)


def metrics_for(segments, audio_sec=600.0, roster=Roster(20, 10, 10), **kw):
    return compute_metrics(segments, roster, audio_sec, completeness=1.0, **kw)


def teacher_heavy():
    """90% teacher talk — a maker session that became a lecture."""
    return [speech(0, 270, Role.TEACHER, conf=0.9), speech(270, 300, Role.STUDENT, conf=0.9)]


def balanced():
    segs = []
    for i in range(10):
        segs.append(speech(i * 30, i * 30 + 12, Role.TEACHER, conf=0.9))
        segs.append(speech(i * 30 + 12, i * 30 + 30, Role.STUDENT, conf=0.9))
    return segs


# ================================================================ headline
def test_the_headline_leads_with_teacher_talk():
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    assert "%" in ins.headline
    assert ins.headline


def test_the_headline_is_one_sentence():
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    assert ins.headline.count(".") <= 2
    assert len(ins.headline) < 200


def test_a_dominant_teacher_is_told_so():
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    assert any(w in ins.headline.lower() for w in ("spoke", "talk"))


def test_a_balanced_session_gets_a_different_headline():
    a = build_insight(metrics_for(teacher_heavy()), activity="Trumpet").headline
    b = build_insight(metrics_for(balanced()), activity="Trumpet").headline
    assert a != b


def test_the_activity_is_named():
    ins = build_insight(metrics_for(balanced()), activity="Shadow Art")
    assert "Shadow Art" in ins.headline or "Shadow Art" in ins.summary


# ================================================================ the suggestion
def test_there_is_exactly_one_suggestion():
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    assert isinstance(ins.suggestion, str)
    assert ins.suggestion


def test_the_suggestion_targets_the_weakest_metric():
    """A five-minute monologue should be told about the monologue, not about wait time."""
    segs = [speech(0, 400, Role.TEACHER, conf=0.9), speech(400, 420, Role.STUDENT, conf=0.9)]
    ins = build_insight(metrics_for(segs), activity="Trumpet")
    assert "stretch" in ins.suggestion.lower() or "minute" in ins.suggestion.lower()


def test_short_wait_time_produces_the_count_to_three_suggestion():
    from src.asr.base import Transcript, TranscriptSegment
    from src.diarization import SpeakerTurn

    segs = balanced()
    # The teacher must hold more of the floor than the student, or teacher_speaker
    # correctly identifies the other one as the teacher. Three answered questions,
    # because M5 withholds a median resting on fewer than that.
    turns = [SpeakerTurn(0, 12, "T"), SpeakerTurn(12.4, 16, "S"),
             SpeakerTurn(20, 60, "T"), SpeakerTurn(60.4, 64, "S"),
             SpeakerTurn(70, 110, "T"), SpeakerTurn(110.4, 114, "S"),
             SpeakerTurn(120, 200, "T")]
    t = Transcript([TranscriptSegment(0, 12, "यह क्या है", -0.4, 0.1, None),
                    TranscriptSegment(20, 60, "यह क्या है", -0.4, 0.1, None),
                    TranscriptSegment(70, 110, "यह क्या है", -0.4, 0.1, None)],
                   "hi", 0.9, "stub", "m", 600.0)
    ins = build_insight(metrics_for(segs, transcript=t, speaker_turns=turns),
                        activity="Trumpet")
    assert "three" in ins.suggestion.lower() or "wait" in ins.suggestion.lower()


def test_a_healthy_session_still_gets_something_useful():
    ins = build_insight(metrics_for(balanced()), activity="Trumpet")
    assert ins.suggestion


def test_the_suggestion_never_cites_a_withheld_metric():
    """Nothing may be advised on the basis of a number we refused to report."""
    weak = [speech(0, 100, Role.TEACHER, conf=0.02), speech(100, 150, Role.STUDENT, conf=0.02)]
    m = metrics_for(weak)
    ins = build_insight(m, activity="Trumpet")
    assert "%" not in ins.suggestion


# ================================================================ the confidence gate
def test_an_unreliable_session_reports_no_numbers_at_all():
    segs = [speech(0, 30, Role.TEACHER, conf=0.02), handson(30, 600)]
    m = compute_metrics(segs, Roster(20, 10, 10), 600.0, completeness=0.3)
    assert m.verdict is Verdict.UNRELIABLE

    ins = build_insight(m, activity="Shadow Art")
    assert "%" not in ins.headline
    assert not ins.metrics_shown


def test_an_unreliable_session_gets_recording_advice_instead():
    segs = [speech(0, 30, Role.TEACHER, conf=0.02), handson(30, 600)]
    m = compute_metrics(segs, Roster(20, 10, 10), 600.0, completeness=0.3)
    ins = build_insight(m, activity="Shadow Art")
    assert any(w in ins.suggestion.lower() for w in ("phone", "record", "closer", "noisy"))


def test_a_usable_session_lists_the_metrics_it_showed():
    ins = build_insight(metrics_for(balanced()), activity="Trumpet")
    assert "M1_teacher_talk_ratio" in ins.metrics_shown


def test_withheld_metrics_are_never_listed_as_shown():
    weak = [speech(0, 100, Role.TEACHER, conf=0.02), speech(100, 150, Role.STUDENT, conf=0.02)]
    ins = build_insight(metrics_for(weak), activity="Trumpet")
    assert "M1_teacher_talk_ratio" not in ins.metrics_shown


# ================================================================ WhatsApp
def test_the_whatsapp_message_fits_a_phone_screen():
    """Her real channel is a WhatsApp bot, not a browser."""
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    assert len(whatsapp_message(ins)) <= config.WHATSAPP_MAX_CHARS


def test_the_whatsapp_message_carries_the_headline_and_the_suggestion():
    ins = build_insight(metrics_for(teacher_heavy()), activity="Trumpet")
    msg = whatsapp_message(ins)
    assert ins.headline[:30] in msg
    assert ins.suggestion[:30] in msg


def test_the_whatsapp_message_greets_by_alias_not_by_name():
    ins = build_insight(metrics_for(balanced()), activity="Trumpet", alias="Teacher B")
    assert "Teacher B" in whatsapp_message(ins)


def test_an_unreliable_session_sends_no_statistics_to_the_teacher():
    segs = [speech(0, 30, Role.TEACHER, conf=0.02), handson(30, 600)]
    m = compute_metrics(segs, Roster(20, 10, 10), 600.0, completeness=0.3)
    assert "%" not in whatsapp_message(build_insight(m, activity="Shadow Art"))


# ================================================================ output shape
def test_insight_is_json_serialisable():
    import json
    json.loads(json.dumps(build_insight(metrics_for(balanced()),
                                        activity="Trumpet").to_dict()))


def test_insight_records_how_it_was_generated():
    """A reviewer must be able to tell a rule from a generated sentence."""
    ins = build_insight(metrics_for(balanced()), activity="Trumpet")
    assert ins.generated_by.startswith("rules")


def test_insight_is_deterministic():
    a = build_insight(metrics_for(balanced()), activity="Trumpet")
    b = build_insight(metrics_for(balanced()), activity="Trumpet")
    assert a.to_dict() == b.to_dict()


def test_the_summary_mentions_hands_on_time_when_there_is_some():
    segs = balanced() + [handson(300, 580)]
    ins = build_insight(metrics_for(segs), activity="Trumpet")
    assert "hands-on" in ins.summary.lower() or "building" in ins.summary.lower()
