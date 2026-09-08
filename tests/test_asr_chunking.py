"""Chunk planning and re-assembly.

Pure arithmetic, no network. Groq caps uploads at 25 MB and 4 of the 5 real recordings
exceed it, so long audio must be split, sent separately, and stitched back with the
segment times shifted into whole-recording time.
"""

from __future__ import annotations

import pytest

from src.asr.base import Transcript, TranscriptSegment
from src.asr.chunking import merge_chunk_transcripts, plan_chunks


def seg(start, end, text, logprob=-0.5):
    return TranscriptSegment(start=start, end=end, text=text,
                             avg_logprob=logprob, no_speech_prob=0.1, words=None)


def chunk_transcript(segments, audio_sec):
    return Transcript(segments=segments, language="hi", language_prob=0.9,
                      backend="stub", model="stub", audio_sec=audio_sec)


# ---------------------------------------------------------------- plan_chunks
def test_short_audio_is_a_single_chunk():
    c = plan_chunks(total_sec=120.0, chunk_sec=600.0)
    assert len(c) == 1
    assert c[0].start == 0.0
    assert c[0].duration == pytest.approx(120.0)


def test_exact_multiple_splits_evenly():
    c = plan_chunks(total_sec=1800.0, chunk_sec=600.0)
    assert [x.start for x in c] == [0.0, 600.0, 1200.0]
    assert all(x.duration == pytest.approx(600.0) for x in c)


def test_remainder_becomes_a_final_short_chunk():
    c = plan_chunks(total_sec=1450.0, chunk_sec=600.0)
    assert len(c) == 3
    assert c[-1].duration == pytest.approx(250.0)


def test_chunks_are_indexed_in_order():
    assert [x.index for x in plan_chunks(1450.0, 600.0)] == [0, 1, 2]


def test_chunks_cover_the_whole_recording():
    c = plan_chunks(total_sec=4062.0, chunk_sec=600.0)
    assert c[-1].start + c[-1].duration == pytest.approx(4062.0)


def test_overlap_makes_chunks_start_earlier():
    c = plan_chunks(total_sec=1800.0, chunk_sec=600.0, overlap_sec=5.0)
    assert c[1].start == pytest.approx(595.0)
    assert c[2].start == pytest.approx(1195.0)


def test_overlap_never_pushes_the_first_chunk_negative():
    assert plan_chunks(1800.0, 600.0, overlap_sec=30.0)[0].start == 0.0


def test_the_real_worst_case_file_splits_into_seven():
    """The 67.7-minute session at 10-minute chunks."""
    assert len(plan_chunks(total_sec=4062.0, chunk_sec=600.0)) == 7


def test_invalid_chunk_length_raises():
    with pytest.raises(ValueError):
        plan_chunks(total_sec=100.0, chunk_sec=0.0)


def test_overlap_larger_than_the_chunk_raises():
    with pytest.raises(ValueError):
        plan_chunks(total_sec=100.0, chunk_sec=10.0, overlap_sec=20.0)


def test_non_positive_total_raises():
    with pytest.raises(ValueError):
        plan_chunks(total_sec=0.0, chunk_sec=600.0)


# ---------------------------------------------------------------- merge
def test_merge_shifts_segment_times_into_whole_recording_time():
    chunks = plan_chunks(total_sec=1200.0, chunk_sec=600.0)
    parts = [
        chunk_transcript([seg(10.0, 20.0, "पहला")], 600.0),
        chunk_transcript([seg(5.0, 15.0, "दूसरा")], 600.0),
    ]
    merged = merge_chunk_transcripts(parts, chunks, audio_sec=1200.0)
    assert [round(s.start, 1) for s in merged.segments] == [10.0, 605.0]


def test_merge_concatenates_the_text_in_order():
    chunks = plan_chunks(1200.0, 600.0)
    parts = [chunk_transcript([seg(0, 5, "एक")], 600.0),
             chunk_transcript([seg(0, 5, "दो")], 600.0)]
    assert merge_chunk_transcripts(parts, chunks, 1200.0).text == "एक दो"


def test_merge_reports_the_whole_recording_length():
    chunks = plan_chunks(1200.0, 600.0)
    parts = [chunk_transcript([seg(0, 5, "a")], 600.0)] * 2
    assert merge_chunk_transcripts(parts, chunks, 1200.0).audio_sec == pytest.approx(1200.0)


def test_merge_drops_duplicates_from_the_overlap_region():
    """With overlap, the same words get transcribed twice — keep one copy."""
    chunks = plan_chunks(total_sec=1200.0, chunk_sec=600.0, overlap_sec=10.0)
    parts = [
        chunk_transcript([seg(595.0, 600.0, "दोहराव")], 600.0),   # -> 595–600
        chunk_transcript([seg(5.0, 10.0, "दोहराव")], 600.0),      # -> 595–600 again
    ]
    merged = merge_chunk_transcripts(parts, chunks, audio_sec=1200.0)
    assert len(merged.segments) == 1


def test_merge_keeps_distinct_text_that_merely_overlaps_in_time():
    chunks = plan_chunks(total_sec=1200.0, chunk_sec=600.0, overlap_sec=10.0)
    parts = [
        chunk_transcript([seg(595.0, 600.0, "पहला")], 600.0),
        chunk_transcript([seg(5.0, 10.0, "अलग")], 600.0),
    ]
    assert len(merge_chunk_transcripts(parts, chunks, 1200.0).segments) == 2


def test_merge_takes_the_most_confident_language_vote():
    chunks = plan_chunks(1200.0, 600.0)
    parts = [
        Transcript([seg(0, 5, "a")], "mr", 0.40, "stub", "stub", 600.0),
        Transcript([seg(0, 5, "b")], "hi", 0.95, "stub", "stub", 600.0),
    ]
    assert merge_chunk_transcripts(parts, chunks, 1200.0).language == "hi"


def test_merge_of_a_single_chunk_is_a_passthrough():
    chunks = plan_chunks(300.0, 600.0)
    part = chunk_transcript([seg(1.0, 2.0, "solo")], 300.0)
    merged = merge_chunk_transcripts([part], chunks, 300.0)
    assert merged.segments[0].start == pytest.approx(1.0)


def test_merge_rejects_a_count_mismatch():
    with pytest.raises(ValueError):
        merge_chunk_transcripts([chunk_transcript([], 600.0)], plan_chunks(1200.0, 600.0), 1200.0)


def test_merge_of_empty_chunks_gives_an_empty_transcript():
    chunks = plan_chunks(1200.0, 600.0)
    parts = [chunk_transcript([], 600.0), chunk_transcript([], 600.0)]
    assert merge_chunk_transcripts(parts, chunks, 1200.0).text == ""


# ---------------------------------------------------------------- tiny tails
def test_a_sliver_at_the_end_is_absorbed_not_sent_as_its_own_request():
    """6.03 s at 2 s chunks must be 3 pieces, not 3 plus a 0.03 s scrap."""
    c = plan_chunks(total_sec=6.03, chunk_sec=2.0)
    assert len(c) == 3
    assert c[-1].end == pytest.approx(6.03)


def test_absorbing_a_sliver_still_covers_everything():
    c = plan_chunks(total_sec=1802.0, chunk_sec=600.0)
    assert len(c) == 3
    assert c[-1].end == pytest.approx(1802.0)


def test_a_real_remainder_is_still_its_own_chunk():
    c = plan_chunks(total_sec=1450.0, chunk_sec=600.0)
    assert len(c) == 3
    assert c[-1].duration == pytest.approx(250.0)


def test_min_tail_is_configurable():
    assert len(plan_chunks(1000.0, 600.0, min_tail_sec=500.0)) == 1
    assert len(plan_chunks(1000.0, 600.0, min_tail_sec=100.0)) == 2


def test_absorbing_keeps_indexes_contiguous():
    assert [x.index for x in plan_chunks(1802.0, 600.0)] == [0, 1, 2]


def test_audio_shorter_than_the_minimum_tail_is_still_one_chunk():
    c = plan_chunks(total_sec=0.5, chunk_sec=600.0)
    assert len(c) == 1
    assert c[0].duration == pytest.approx(0.5)
