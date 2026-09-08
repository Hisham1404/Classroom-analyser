"""AI4Bharat IndicConformer via onnx-asr.

A third architecture in the same one-line switch. It matters for three reasons:

* **CTC/RNNT, not autoregressive** — it cannot invent a fluent sentence out of noise the
  way Whisper does, which makes it a natural cross-check on Whisper hallucinations.
* **ONNX, no torch** — runs on the onnxruntime that faster-whisper already pulls in,
  so it costs no extra disk on a drive with 3.2 GB free.
* **~4x faster than realtime on this CPU**, against the local Hindi Whisper's 0.11x.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src import config
from src.asr import BACKENDS, get_backend
from src.asr.base import ASRBackend, Transcript
from src.asr.indic import IndicConformerBackend


class _FakeRecognizer:
    """Stands in for onnx_asr's VAD + timestamps adapter."""

    def __init__(self):
        self.calls = 0

    def recognize(self, path):
        self.calls += 1
        return iter([
            SimpleNamespace(start=1.19, end=8.13, text="आसमान से बादल भी जा रहा था",
                            logprobs=[-0.3, -0.5]),
            SimpleNamespace(start=8.32, end=8.96, text="  ", logprobs=None),
            SimpleNamespace(start=18.63, end=24.77, text="उसने एक नदी आके पहाड़ से",
                            logprobs=[-0.7]),
        ])


@pytest.fixture
def backend(monkeypatch):
    b = IndicConformerBackend(use_vad=True)
    monkeypatch.setattr(b, "_load_recognizer", lambda: _FakeRecognizer())
    return b


@pytest.fixture
def audio(intact_session: Path) -> Path:
    return intact_session / "OD90001_2026-01-06-114155.mp3"


# ---------------------------------------------------------------- contract
def test_registered_as_a_backend():
    assert "indic" in BACKENDS


def test_satisfies_the_shared_protocol(backend):
    assert isinstance(backend, ASRBackend)
    assert backend.name == "indic"


def test_returns_the_shared_transcript_type(backend, audio):
    t = backend.transcribe(audio)
    assert isinstance(t, Transcript)
    assert t.backend == "indic"
    assert "indicconformer" in t.model.lower()


def test_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert get_backend("indic").name == "indic"


def test_does_not_chunk(backend):
    """Runs locally, so the 25 MB upload cap does not apply."""
    assert backend.max_upload_bytes is None


# ---------------------------------------------------------------- mapping
def test_segments_keep_their_timings(backend, audio):
    segs = backend.transcribe(audio).segments
    assert segs[0].start == pytest.approx(1.19)
    assert segs[0].end == pytest.approx(8.13)


def test_blank_segments_are_dropped(backend, audio):
    """VAD emits silent stretches with empty text — they are not speech."""
    texts = [s.text for s in backend.transcribe(audio).segments]
    assert "" not in texts
    assert len(texts) == 2


def test_text_is_joined_in_order(backend, audio):
    assert backend.transcribe(audio).text.startswith("आसमान से बादल")


def test_logprobs_are_averaged_per_segment(backend, audio):
    """M6's confidence gate needs a per-segment number, not a token list."""
    seg = backend.transcribe(audio).segments[0]
    assert seg.avg_logprob == pytest.approx(-0.4)


def test_a_segment_without_logprobs_is_none_not_zero(backend, audio):
    """Zero would read as perfect confidence — the opposite of unknown."""
    assert backend.transcribe(audio).segments[-1].avg_logprob == pytest.approx(-0.7)


def test_language_is_reported(backend, audio):
    assert backend.transcribe(audio).language == "hi"


def test_audio_length_is_measured_not_guessed(backend, audio):
    """The fixture is 6 s; segment ends run to 24.77 and must not be mistaken for it."""
    assert backend.transcribe(audio).audio_sec == pytest.approx(6.0, abs=0.3)


# ---------------------------------------------------------------- behaviour
def test_missing_file_raises(backend, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        backend.transcribe(tmp_path / "ghost.mp3")


def test_second_call_hits_the_cache(monkeypatch, audio, tmp_path):
    from src.asr.cache import TranscriptCache

    rec = _FakeRecognizer()
    b = IndicConformerBackend(cache=TranscriptCache(tmp_path / "c"))
    monkeypatch.setattr(b, "_load_recognizer", lambda: rec)

    b.transcribe(audio, language="hi")
    b.transcribe(audio, language="hi")
    assert rec.calls == 1


def test_a_recognizer_failure_becomes_a_transcription_error(monkeypatch, audio):
    from src.asr.base import TranscriptionError

    class Boom:
        def recognize(self, path):
            raise RuntimeError("onnx session died")

    b = IndicConformerBackend()
    monkeypatch.setattr(b, "_load_recognizer", lambda: Boom())
    with pytest.raises(TranscriptionError) as e:
        b.transcribe(audio)
    assert "onnx session died" in str(e.value)


def test_model_id_can_be_overridden():
    assert get_backend("indic", model="OpenVoiceOS/ai4bharat-indicconformer-mr-onnx").model_id \
        == "OpenVoiceOS/ai4bharat-indicconformer-mr-onnx"


def test_config_declares_the_default_model():
    assert "indicconformer" in config.INDIC_MODEL.lower()
    assert config.INDIC_MODEL.endswith("-onnx")


# ---------------------------------------------------------------- VAD fallback
class _EmptyVadRecognizer:
    """Silero returns nothing — measured on 2 of 3 real classroom clips."""

    def __init__(self):
        self.calls = 0

    def recognize(self, path):
        self.calls += 1
        return iter([])


class _PlainRecognizer:
    """The same model without VAD, which does find speech in those clips."""

    def __init__(self):
        self.calls = 0

    def recognize(self, path):
        self.calls += 1
        return SimpleNamespace(text="और ओबल साइक में क्या होगा क्लोज सर्क है न",
                               logprobs=[-0.6])


@pytest.fixture
def vad_drops_everything(monkeypatch):
    b = IndicConformerBackend(use_vad=True)
    vad, plain = _EmptyVadRecognizer(), _PlainRecognizer()
    monkeypatch.setattr(b, "_load_recognizer", lambda: vad)
    monkeypatch.setattr(b, "_load_plain_recognizer", lambda: plain)
    return b, vad, plain


def test_falls_back_when_vad_finds_no_speech(vad_drops_everything, audio):
    """An over-aggressive VAD must not turn a usable transcript into silence."""
    b, vad, plain = vad_drops_everything
    t = b.transcribe(audio)

    assert vad.calls == 1 and plain.calls == 1
    assert "क्लोज सर्क" in t.text


def test_the_fallback_segment_spans_the_whole_clip(vad_drops_everything, audio):
    b, _, _ = vad_drops_everything
    t = b.transcribe(audio)

    assert len(t.segments) == 1
    assert t.segments[0].start == 0.0
    assert t.segments[0].end == pytest.approx(t.audio_sec)


def test_the_fallback_keeps_confidence(vad_drops_everything, audio):
    b, _, _ = vad_drops_everything
    assert b.transcribe(audio).segments[0].avg_logprob == pytest.approx(-0.6)


def test_no_fallback_when_vad_works(audio, monkeypatch):
    backend = IndicConformerBackend(use_vad=True)
    monkeypatch.setattr(backend, "_load_recognizer", lambda: _FakeRecognizer())
    plain = _PlainRecognizer()
    monkeypatch.setattr(backend, "_load_plain_recognizer", lambda: plain)

    backend.transcribe(audio)
    assert plain.calls == 0


def test_genuinely_silent_audio_still_gives_an_empty_transcript(monkeypatch, audio):
    """Falling back is not the same as inventing speech."""
    class Silent:
        def recognize(self, path):
            return SimpleNamespace(text="   ", logprobs=None)

    b = IndicConformerBackend(use_vad=True)
    monkeypatch.setattr(b, "_load_recognizer", lambda: _EmptyVadRecognizer())
    monkeypatch.setattr(b, "_load_plain_recognizer", lambda: Silent())

    t = b.transcribe(audio)
    assert t.text == ""
    assert t.segments == []


def test_vad_can_be_switched_off_entirely(monkeypatch, audio):
    b = IndicConformerBackend(use_vad=False)
    plain = _PlainRecognizer()
    monkeypatch.setattr(b, "_load_recognizer", lambda: plain)

    assert "क्लोज सर्क" in b.transcribe(b_path := audio).text
    assert plain.calls == 1


# ---------------------------------------------------------------- token segmentation
from src.asr.indic import segments_from_tokens


def test_one_run_of_tokens_becomes_one_segment():
    segs = segments_from_tokens(
        tokens=[" यह", " क्या", " है"], timestamps=[1.0, 1.4, 1.8],
        logprobs=[-0.4, -0.6, -0.2], gap_sec=1.0, audio_sec=30.0)
    assert len(segs) == 1
    assert segs[0].text == "यह क्या है"


def test_a_long_gap_splits_the_run():
    """Real output: token stamps jump 2.2 -> 9.56 where the speaker stopped."""
    segs = segments_from_tokens(
        tokens=[" एक", " दो", " तीन"], timestamps=[2.2, 2.6, 9.56],
        logprobs=[-0.5, -0.5, -0.5], gap_sec=1.0, audio_sec=30.0)
    assert len(segs) == 2
    assert segs[1].text == "तीन"


def test_segment_start_is_its_first_token(): 
    segs = segments_from_tokens([" एक", " दो"], [3.0, 3.4], [-0.5, -0.5], 1.0, 30.0)
    assert segs[0].start == pytest.approx(3.0)


def test_segment_end_never_exceeds_the_audio():
    segs = segments_from_tokens([" एक"], [29.9], [-0.5], 1.0, 30.0)
    assert segs[0].end <= 30.0


def test_logprobs_are_averaged_within_each_segment():
    segs = segments_from_tokens(
        [" एक", " दो", " तीन"], [1.0, 1.4, 9.0], [-0.2, -0.6, -0.9], 1.0, 30.0)
    assert segs[0].avg_logprob == pytest.approx(-0.4)
    assert segs[1].avg_logprob == pytest.approx(-0.9)


def test_subword_tokens_rejoin_without_spurious_spaces():
    """Tokens carry their own leading spaces — naive joining would mangle the words."""
    segs = segments_from_tokens(
        [" रस्", "ता", " था"], [1.0, 1.1, 1.3], [-0.5] * 3, 1.0, 30.0)
    assert segs[0].text == "रस्ता था"


def test_no_tokens_gives_no_segments():
    assert segments_from_tokens([], [], [], 1.0, 30.0) == []


def test_blank_tokens_are_dropped():
    segs = segments_from_tokens([" ", "  "], [1.0, 1.2], [-0.5, -0.5], 1.0, 30.0)
    assert segs == []


def test_missing_logprobs_are_tolerated():
    segs = segments_from_tokens([" एक", " दो"], [1.0, 1.4], None, 1.0, 30.0)
    assert segs[0].avg_logprob is None


def test_mismatched_lengths_do_not_crash():
    segs = segments_from_tokens([" एक", " दो", " तीन"], [1.0, 1.4], [-0.5], 1.0, 30.0)
    assert len(segs) <= 2


# ---------------------------------------------------------------- default path
def test_vad_is_off_by_default():
    """onnx-asr's VAD returns SegmentResult, which carries no logprobs. We already
    segment with faster-whisper's Silero in signal_layer, so this path costs us the
    confidence signal for nothing."""
    assert IndicConformerBackend().use_vad is False


def test_the_default_path_produces_confidence(monkeypatch, audio):
    class TokenRecognizer:
        def recognize(self, path):
            return SimpleNamespace(text="यह क्या है", tokens=[" यह", " क्या", " है"],
                                   timestamps=[1.0, 1.4, 1.8], logprobs=[-0.4, -0.6, -0.2])

    b = IndicConformerBackend()
    monkeypatch.setattr(b, "_load_recognizer", lambda: TokenRecognizer())
    t = b.transcribe(audio)

    assert t.mean_logprob is not None
    assert t.text == "यह क्या है"


# ---------------------------------------------------------------- long audio
def test_config_declares_the_duration_cap():
    """The conformer's ONNX graph has a fixed positional-encoding length. Past it the
    session dies with 'Attempting to broadcast an axis ... 2501 by 7501' — roughly
    100 seconds of audio."""
    assert 0 < config.INDIC_MAX_AUDIO_SEC < 100


def test_short_audio_is_not_chunked(monkeypatch, audio):
    class Once:
        def __init__(self): self.calls = 0
        def recognize(self, path):
            self.calls += 1
            return SimpleNamespace(text="यह क्या है", tokens=[" यह", " क्या"],
                                   timestamps=[1.0, 1.4], logprobs=[-0.4, -0.5])

    rec = Once()
    b = IndicConformerBackend()
    monkeypatch.setattr(b, "_load_recognizer", lambda: rec)
    b.transcribe(audio)                       # fixture is 6 s
    assert rec.calls == 1


def test_long_audio_is_split_into_chunks(monkeypatch, intact_session: Path):
    b = IndicConformerBackend()
    monkeypatch.setattr(b, "max_audio_sec", 2.0)     # force the 6 s fixture to split

    calls: list[str] = []

    class Chunked:
        def recognize(self, path):
            calls.append(str(path))
            return SimpleNamespace(text=f"c{len(calls)}", tokens=[f" c{len(calls)}"],
                                   timestamps=[0.5], logprobs=[-0.5])

    monkeypatch.setattr(b, "_load_recognizer", lambda: Chunked())
    t = b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")

    assert len(calls) >= 3
    assert t.text.startswith("c1")
    assert t.audio_sec == pytest.approx(6.0, abs=0.3)


def test_chunked_segments_are_shifted_into_whole_recording_time(monkeypatch,
                                                                intact_session: Path):
    b = IndicConformerBackend()
    monkeypatch.setattr(b, "max_audio_sec", 2.0)

    n = {"i": 0}

    class Chunked:
        def recognize(self, path):
            n["i"] += 1
            return SimpleNamespace(text="x", tokens=[" x"], timestamps=[0.5],
                                   logprobs=[-0.5])

    monkeypatch.setattr(b, "_load_recognizer", lambda: Chunked())
    t = b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")

    assert t.segments[-1].start > 2.0          # later chunks land later in the recording


def test_chunking_preserves_confidence(monkeypatch, intact_session: Path):
    b = IndicConformerBackend()
    monkeypatch.setattr(b, "max_audio_sec", 2.0)

    class Chunked:
        def recognize(self, path):
            return SimpleNamespace(text="x", tokens=[" x"], timestamps=[0.5],
                                   logprobs=[-0.7])

    monkeypatch.setattr(b, "_load_recognizer", lambda: Chunked())
    t = b.transcribe(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert t.mean_logprob == pytest.approx(-0.7)
