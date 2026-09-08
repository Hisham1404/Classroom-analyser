"""Stage 02 — the acoustic layer.

This is the half of the pipeline that survives when the transcript does not. M1 talk-time,
M2 participation, M3 interaction density and M4 longest stretch are all built on these
segments, so a mistake here is silent and poisons every "robust" metric.

The maker-session distinction lives here too: non-speech that is *loud* is children
building things, not dead air. Conflating them would score the best lesson as the worst.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.signal_layer import (
    SegmentKind,
    build_timeline,
    classify_non_speech,
    rms_db,
    spectral_flatness,
)

SR = 16000


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3, sr: int = SR):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, amp: float = 0.3, sr: int = SR, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def silence(seconds: float, sr: int = SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ================================================================ rms
def test_rms_of_silence_is_very_low():
    assert rms_db(silence(1.0)) < -80


def test_rms_of_a_loud_tone_is_high():
    assert rms_db(tone(1.0, amp=0.5)) > -20


def test_rms_is_monotonic_in_amplitude():
    assert rms_db(tone(1.0, amp=0.5)) > rms_db(tone(1.0, amp=0.05))


def test_rms_of_an_empty_array_does_not_crash():
    assert rms_db(np.array([], dtype=np.float32)) < -80


# ================================================================ flatness
def test_a_pure_tone_has_low_spectral_flatness():
    """Near-field speech is more tonal; far-field babble is closer to noise."""
    assert spectral_flatness(tone(1.0)) < 0.1


def test_white_noise_has_high_spectral_flatness():
    assert spectral_flatness(noise(1.0)) > 0.3


def test_flatness_separates_tone_from_noise():
    assert spectral_flatness(noise(1.0)) > spectral_flatness(tone(1.0))


def test_flatness_of_silence_is_defined_not_nan():
    v = spectral_flatness(silence(1.0))
    assert v == v          # not NaN
    assert 0.0 <= v <= 1.0


# ================================================================ non-speech
def test_loud_non_speech_is_hands_on():
    """Children building things. The metric that stops a maker lesson scoring as dead air."""
    assert classify_non_speech(rms_db(noise(1.0, amp=0.3))) is SegmentKind.HANDSON


def test_quiet_non_speech_is_dead_time():
    assert classify_non_speech(rms_db(noise(1.0, amp=0.0005))) is SegmentKind.DEAD


def test_the_boundary_uses_the_configured_threshold():
    assert classify_non_speech(config.HANDSON_RMS_DB + 1) is SegmentKind.HANDSON
    assert classify_non_speech(config.HANDSON_RMS_DB - 1) is SegmentKind.DEAD


# ================================================================ timeline
def test_timeline_covers_the_whole_recording_with_no_gaps():
    wave = np.concatenate([tone(2.0), silence(2.0), tone(2.0)])
    segs = build_timeline(wave, SR)

    assert segs[0].start == pytest.approx(0.0)
    assert segs[-1].end == pytest.approx(len(wave) / SR, abs=0.05)
    for a, b in zip(segs, segs[1:]):
        assert a.end == pytest.approx(b.start, abs=1e-6)


def test_timeline_segments_are_sorted():
    segs = build_timeline(np.concatenate([tone(1.0), silence(1.0), tone(1.0)]), SR)
    assert [s.start for s in segs] == sorted(s.start for s in segs)


def test_every_segment_has_positive_duration():
    segs = build_timeline(np.concatenate([tone(1.0), silence(1.5), tone(1.0)]), SR)
    assert all(s.duration > 0 for s in segs)


def test_pure_silence_yields_one_dead_segment():
    segs = build_timeline(silence(4.0), SR)
    assert len(segs) == 1
    assert segs[0].kind is SegmentKind.DEAD


def test_loud_noise_with_no_speech_is_all_hands_on():
    """The Shadow Art case: 20% speech density but the loudest file of the five."""
    segs = build_timeline(noise(4.0, amp=0.3), SR)
    assert all(s.kind is not SegmentKind.SPEECH for s in segs)
    assert any(s.kind is SegmentKind.HANDSON for s in segs)


def test_segments_carry_their_acoustic_features():
    segs = build_timeline(np.concatenate([tone(2.0), silence(2.0)]), SR)
    for s in segs:
        assert s.rms_db < 0
        assert 0.0 <= s.flatness <= 1.0


def test_empty_audio_yields_no_segments():
    assert build_timeline(np.array([], dtype=np.float32), SR) == []


def test_timeline_is_deterministic():
    wave = np.concatenate([tone(1.5), silence(2.5), noise(1.5)])
    a = build_timeline(wave, SR)
    b = build_timeline(wave, SR)
    assert [(s.start, s.end, s.kind) for s in a] == [(s.start, s.end, s.kind) for s in b]


def test_speech_and_non_speech_durations_sum_to_the_recording():
    wave = np.concatenate([tone(2.0), silence(2.0), tone(1.0)])
    segs = build_timeline(wave, SR)
    assert sum(s.duration for s in segs) == pytest.approx(len(wave) / SR, abs=0.05)


# ================================================================ integration
@pytest.mark.integration
def test_real_session_timeline_is_wellformed(real_data_dir):
    """A 30-second slice of a real recording, end to end."""
    import subprocess

    from src.signal_layer import load_waveform

    session = sorted(p for p in real_data_dir.iterdir() if p.is_dir())[0]
    mp3 = session / f"{session.name}.mp3"

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "slice.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
                        "-ss", "600", "-t", "30", "-i", str(mp3),
                        "-ac", "1", "-ar", str(SR), str(clip)], check=True)
        segs = build_timeline(load_waveform(clip), SR)

    assert segs
    assert sum(s.duration for s in segs) == pytest.approx(30.0, abs=0.5)
    assert any(s.kind is SegmentKind.SPEECH for s in segs), "VAD found no speech in real audio"


# ================================ speech spans supplied from outside
#
# Where diarization ran, its turns are a better speech map than Silero — measured on
# OD11166_2026-01-12, where 142 s of 161 s of real speech turns were being classified
# as hands-on activity noise.

def test_a_caller_supplied_span_replaces_the_vad(monkeypatch):
    def explode(*args, **kwargs):                    # pragma: no cover - must not run
        raise AssertionError("Silero was called even though spans were supplied")

    monkeypatch.setattr("src.signal_layer.detect_speech", explode)

    wave = np.zeros(16000 * 10, dtype=np.float32)
    timeline = build_timeline(wave, speech_spans=[(2.0, 6.0)])
    speech = [s for s in timeline if s.kind is SegmentKind.SPEECH]
    assert len(speech) == 1
    assert (speech[0].start, speech[0].end) == pytest.approx((2.0, 6.0))


def test_supplied_spans_still_tile_the_whole_recording():
    wave = np.zeros(16000 * 10, dtype=np.float32)
    timeline = build_timeline(wave, speech_spans=[(2.0, 6.0)])
    assert timeline[0].start == pytest.approx(0.0)
    assert timeline[-1].end == pytest.approx(10.0)
    for a, b in zip(timeline, timeline[1:]):
        assert a.end == pytest.approx(b.start)


def test_supplied_spans_are_sorted_and_merged_before_use():
    """Diarization turns overlap and arrive in speaker order, not clock order."""
    wave = np.zeros(16000 * 20, dtype=np.float32)
    timeline = build_timeline(wave, speech_spans=[(12.0, 15.0), (2.0, 6.0), (5.0, 8.0)])
    speech = [(s.start, s.end) for s in timeline if s.kind is SegmentKind.SPEECH]
    assert speech == pytest.approx([(2.0, 8.0), (12.0, 15.0)])


def test_a_span_running_past_the_audio_is_clipped():
    wave = np.zeros(16000 * 5, dtype=np.float32)
    timeline = build_timeline(wave, speech_spans=[(3.0, 99.0)])
    assert timeline[-1].end == pytest.approx(5.0)
    assert timeline[-1].kind is SegmentKind.SPEECH


def test_no_supplied_spans_means_no_speech_not_fall_back_to_the_vad(monkeypatch):
    """An empty list is an answer - diarization heard nothing. Falling back to Silero
    there would silently reintroduce the detector we just replaced."""
    def explode(*args, **kwargs):                    # pragma: no cover - must not run
        raise AssertionError("Silero was called for an explicit empty span list")

    monkeypatch.setattr("src.signal_layer.detect_speech", explode)
    wave = np.zeros(16000 * 5, dtype=np.float32)
    timeline = build_timeline(wave, speech_spans=[])
    assert not [s for s in timeline if s.kind is SegmentKind.SPEECH]
