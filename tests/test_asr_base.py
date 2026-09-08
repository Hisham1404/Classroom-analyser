"""The shared transcript shape. Every backend must produce exactly this, or the
downstream metrics engine would have to care which model ran — which is the whole
thing we are trying to avoid.
"""

from __future__ import annotations

import pytest

from src.asr.base import Transcript, TranscriptSegment, Word


def seg(start=0.0, end=1.0, text="नमस्ते", logprob=-0.5, no_speech=0.1, words=None):
    return TranscriptSegment(start=start, end=end, text=text,
                             avg_logprob=logprob, no_speech_prob=no_speech, words=words)


# ---------------------------------------------------------------- segment
def test_segment_duration():
    assert seg(2.0, 5.5).duration == pytest.approx(3.5)


def test_segment_rejects_reversed_times():
    with pytest.raises(ValueError):
        seg(5.0, 2.0)


def test_segment_text_is_stripped():
    assert seg(text="  hello  ").text == "hello"


def test_segment_offset_shifts_both_ends_and_words():
    s = seg(1.0, 2.0, words=[Word(1.2, 1.6, "क्या")])
    moved = s.offset(600.0)
    assert moved.start == pytest.approx(601.0)
    assert moved.end == pytest.approx(602.0)
    assert moved.words[0].start == pytest.approx(601.2)


def test_offset_leaves_the_original_untouched():
    s = seg(1.0, 2.0)
    s.offset(10.0)
    assert s.start == 1.0


# ---------------------------------------------------------------- transcript
def make_transcript(**kw):
    defaults = dict(
        segments=[seg(0, 2, "एक", -0.4), seg(2, 5, "दो", -1.0)],
        language="hi", language_prob=0.95,
        backend="stub", model="stub-1", audio_sec=5.0,
    )
    return Transcript(**(defaults | kw))


def test_transcript_joins_segment_text():
    assert make_transcript().text == "एक दो"


def test_transcript_mean_logprob_is_duration_weighted():
    """A 3 s bad segment must outweigh a 2 s good one."""
    t = make_transcript()
    assert t.mean_logprob == pytest.approx((-0.4 * 2 + -1.0 * 3) / 5)


def test_mean_logprob_is_none_when_no_backend_reported_one():
    t = make_transcript(segments=[seg(0, 2, "एक", logprob=None)])
    assert t.mean_logprob is None


def test_speech_sec_sums_segment_durations():
    assert make_transcript().speech_sec == pytest.approx(5.0)


def test_speech_density_is_speech_over_audio():
    t = make_transcript(audio_sec=10.0)
    assert t.speech_density == pytest.approx(0.5)


def test_speech_density_never_exceeds_one():
    t = make_transcript(audio_sec=1.0)
    assert t.speech_density == pytest.approx(1.0)


def test_empty_transcript_is_valid_not_an_error():
    """A recording can legitimately contain no detectable speech."""
    t = make_transcript(segments=[])
    assert t.text == ""
    assert t.speech_sec == 0.0
    assert t.mean_logprob is None


def test_transcript_rejects_non_positive_audio_length():
    with pytest.raises(ValueError):
        make_transcript(audio_sec=0.0)


def test_transcript_segments_are_sorted_by_start():
    t = make_transcript(segments=[seg(5, 7, "b"), seg(0, 2, "a")])
    assert [s.text for s in t.segments] == ["a", "b"]


# ---------------------------------------------------------------- round trip
def test_transcript_survives_a_json_round_trip():
    t = make_transcript(segments=[seg(0, 2, "एक", words=[Word(0.1, 0.5, "एक")])])
    back = Transcript.from_dict(t.to_dict())
    assert back.text == t.text
    assert back.backend == t.backend
    assert back.model == t.model
    assert back.segments[0].words[0].word == "एक"
    assert back.mean_logprob == pytest.approx(t.mean_logprob)


def test_to_dict_is_json_serialisable():
    import json
    json.loads(json.dumps(make_transcript().to_dict()))
