"""Choosing which slices of audio the bake-off runs on.

Comparing models is only meaningful if every model sees exactly the same audio, so
selection must be deterministic. Clips also skip the head and tail of a recording,
where the phone is usually being picked up or put down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.clips import ClipSpec, extract_clip, select_clips


# ---------------------------------------------------------------- selection
def test_returns_the_requested_number_of_clips():
    assert len(select_clips(total_sec=3600.0, n_clips=5, clip_sec=180.0)) == 5


def test_clips_are_in_chronological_order():
    starts = [c.start for c in select_clips(3600.0, 5, 180.0)]
    assert starts == sorted(starts)


def test_clips_do_not_overlap():
    clips = select_clips(3600.0, 5, 180.0)
    for a, b in zip(clips, clips[1:]):
        assert a.start + a.duration <= b.start + 1e-9


def test_clips_stay_inside_the_recording():
    for c in select_clips(3600.0, 5, 180.0):
        assert c.start >= 0.0
        assert c.start + c.duration <= 3600.0


def test_head_and_tail_are_skipped():
    """The first and last moments are usually the phone being handled."""
    clips = select_clips(3600.0, 5, 180.0, margin=0.05)
    assert clips[0].start >= 3600.0 * 0.05
    assert clips[-1].start + clips[-1].duration <= 3600.0 * 0.95


def test_selection_is_deterministic():
    a = select_clips(4062.0, 4, 180.0)
    b = select_clips(4062.0, 4, 180.0)
    assert [(c.start, c.duration) for c in a] == [(c.start, c.duration) for c in b]


def test_clips_are_indexed_from_zero():
    assert [c.index for c in select_clips(3600.0, 3, 180.0)] == [0, 1, 2]


def test_short_recording_yields_fewer_clips_rather_than_overlapping_ones():
    clips = select_clips(total_sec=300.0, n_clips=5, clip_sec=180.0)
    assert len(clips) < 5
    for a, b in zip(clips, clips[1:]):
        assert a.start + a.duration <= b.start + 1e-9


def test_recording_shorter_than_one_clip_gives_a_single_shortened_clip():
    clips = select_clips(total_sec=60.0, n_clips=3, clip_sec=180.0)
    assert len(clips) == 1
    assert clips[0].duration <= 60.0


def test_zero_clips_requested_raises():
    with pytest.raises(ValueError):
        select_clips(3600.0, 0, 180.0)


def test_non_positive_clip_length_raises():
    with pytest.raises(ValueError):
        select_clips(3600.0, 3, 0.0)


def test_the_real_session_split_into_three_minute_clips():
    """67.7 min -> 4 clips of 3 min, well spaced."""
    clips = select_clips(total_sec=4062.0, n_clips=4, clip_sec=180.0)
    assert len(clips) == 4
    assert all(c.duration == pytest.approx(180.0) for c in clips)
    gaps = [b.start - (a.start + a.duration) for a, b in zip(clips, clips[1:])]
    assert all(g > 0 for g in gaps)


# ---------------------------------------------------------------- ids
def test_clip_id_is_stable_and_readable():
    c = ClipSpec(session_id="OD11165_2026-01-06-114155", index=2, start=600.0, duration=180.0)
    assert c.clip_id == "OD11165_2026-01-06-114155_c02"


def test_clip_id_survives_a_different_start_time():
    a = ClipSpec("s", 1, 10.0, 180.0)
    b = ClipSpec("s", 1, 99.0, 180.0)
    assert a.clip_id == b.clip_id


# ---------------------------------------------------------------- extraction
def test_extract_writes_a_real_audio_file(intact_session: Path, tmp_path: Path):
    src = intact_session / "OD90001_2026-01-06-114155.mp3"
    spec = ClipSpec("OD90001_2026-01-06-114155", 0, 1.0, 2.0)
    out = extract_clip(src, spec, tmp_path)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_extracted_clip_has_the_requested_duration(intact_session: Path, tmp_path: Path):
    from src.ingest import probe_audio

    src = intact_session / "OD90001_2026-01-06-114155.mp3"
    out = extract_clip(src, ClipSpec("s", 0, 1.0, 2.0), tmp_path)
    assert probe_audio(out).actual_sec == pytest.approx(2.0, abs=0.2)


def test_extracted_clip_is_16k_mono(intact_session: Path, tmp_path: Path):
    """Every model gets identically prepared audio, so the comparison is fair."""
    from src.ingest import probe_audio

    src = intact_session / "OD90001_2026-01-06-114155.mp3"
    probe = probe_audio(extract_clip(src, ClipSpec("s", 0, 0.0, 2.0), tmp_path))
    assert probe.sample_rate == 16000
    assert probe.channels == 1


def test_extract_is_idempotent(intact_session: Path, tmp_path: Path):
    src = intact_session / "OD90001_2026-01-06-114155.mp3"
    spec = ClipSpec("s", 0, 0.0, 2.0)
    first = extract_clip(src, spec, tmp_path)
    second = extract_clip(src, spec, tmp_path)
    assert first == second


def test_extract_from_a_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        extract_clip(tmp_path / "ghost.mp3", ClipSpec("s", 0, 0.0, 2.0), tmp_path)
