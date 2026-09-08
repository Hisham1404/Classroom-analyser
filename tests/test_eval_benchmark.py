"""Scoring models against a reference dataset — real WER, not proxies.

Everything else in `eval/` measures models against each other because our own recordings
have no ground truth. This module is the one place we get an actual error rate, by using
a public Hindi benchmark that ships audio and transcripts together.

The catch to keep in mind: FLEURS is clean, read, single-speaker news prose. A model that
wins here has not been shown to win on 27 children in an Igatpuri classroom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.asr.base import Transcript, TranscriptSegment
from src.evaluation.benchmark import (
    BenchmarkResult,
    BenchmarkSample,
    corpus_cer,
    corpus_wer,
    score_backend,
)


def sample(ref: str, name: str = "s0") -> BenchmarkSample:
    return BenchmarkSample(sample_id=name, audio_path=Path(f"{name}.wav"), reference=ref)


class FakeBackend:
    """Returns a canned transcript per sample id."""

    name, model_id = "fake", "fake-model"

    def __init__(self, answers: dict[str, str], fail_on: set[str] | None = None):
        self.answers, self.fail_on = answers, fail_on or set()
        self.seen: list[Path] = []

    def transcribe(self, path, language=None, use_cache=True):
        key = Path(path).stem
        if key in self.fail_on:
            raise RuntimeError("backend blew up")
        self.seen.append(Path(path))
        text = self.answers.get(key, "")
        return Transcript(
            segments=[TranscriptSegment(0.0, 5.0, text, -0.4, 0.1, None)] if text else [],
            language="hi", language_prob=0.9, backend=self.name,
            model=self.model_id, audio_sec=5.0,
        )


# ================================================================ corpus WER
def test_corpus_wer_is_edits_over_total_words_not_a_mean_of_rates():
    """The standard definition. Averaging per-sample rates lets one short sample
    dominate — a 1-word reference scored wrong would count as much as a 50-word one."""
    pairs = [("एक दो तीन चार पाँच छह सात आठ नौ दस", "एक दो तीन चार पाँच छह सात आठ नौ दस"),  # 10 words, 0 errors
             ("ग", "क")]                                                                    # 1 word, 1 error
    assert corpus_wer(pairs) == pytest.approx(1 / 11)

    naive_mean = (0.0 + 1.0) / 2
    assert corpus_wer(pairs) < naive_mean


def test_corpus_wer_of_perfect_output_is_zero():
    assert corpus_wer([("यह क्या है", "यह क्या है")]) == 0.0


def test_corpus_wer_of_empty_output_is_one():
    assert corpus_wer([("एक दो तीन", "")]) == pytest.approx(1.0)


def test_corpus_wer_can_exceed_one_when_a_model_rambles():
    assert corpus_wer([("एक", "एक दो तीन चार")]) > 1.0


def test_corpus_wer_needs_at_least_one_reference():
    with pytest.raises(ValueError):
        corpus_wer([])


def test_corpus_cer_is_gentler_than_wer_on_near_misses():
    pairs = [("कागज", "कागच")]
    assert corpus_cer(pairs) < corpus_wer(pairs)


# ================================================================ scoring a backend
def test_scores_every_sample():
    samples = [sample("एक दो तीन", "a"), sample("चार पाँच", "b")]
    backend = FakeBackend({"a": "एक दो तीन", "b": "चार पाँच"})
    result = score_backend(backend, samples)

    assert result.scored == 2
    assert result.wer == 0.0
    assert result.cer == 0.0


def test_reports_per_sample_rows_for_inspection():
    samples = [sample("एक दो तीन", "a")]
    result = score_backend(FakeBackend({"a": "एक दो चार"}), samples)

    assert len(result.rows) == 1
    assert result.rows[0].hypothesis == "एक दो चार"
    assert result.rows[0].wer == pytest.approx(1 / 3)


def test_a_failing_sample_is_recorded_not_raised():
    samples = [sample("एक दो तीन", "a"), sample("चार पाँच", "b")]
    result = score_backend(FakeBackend({"a": "एक दो तीन"}, fail_on={"b"}), samples)

    assert result.scored == 1
    assert result.failed == 1
    assert "b" in result.failures[0]


def test_a_failed_sample_does_not_silently_count_as_correct():
    """The dangerous version of this bug: a crash reads as a perfect transcript."""
    samples = [sample("एक दो तीन", "a")]
    result = score_backend(FakeBackend({}, fail_on={"a"}), samples)

    assert result.scored == 0
    assert result.wer is None


def test_an_empty_transcript_scores_as_fully_wrong_not_skipped():
    samples = [sample("एक दो तीन", "a")]
    result = score_backend(FakeBackend({"a": ""}), samples)

    assert result.scored == 1
    assert result.wer == pytest.approx(1.0)


def test_backend_identity_is_carried_through():
    result = score_backend(FakeBackend({"a": "एक"}), [sample("एक", "a")])
    assert result.backend == "fake"
    assert result.model == "fake-model"


def test_timing_is_recorded():
    result = score_backend(FakeBackend({"a": "एक"}), [sample("एक", "a")])
    assert result.elapsed_sec >= 0.0


def test_result_is_json_serialisable():
    import json
    result = score_backend(FakeBackend({"a": "एक दो"}), [sample("एक दो तीन", "a")])
    json.loads(json.dumps(result.to_dict()))


def test_scoring_no_samples_raises():
    with pytest.raises(ValueError):
        score_backend(FakeBackend({}), [])


# ================================================================ comparison
def test_a_better_model_gets_a_lower_wer():
    samples = [sample("यह कागज़ है इसे मोड़ो", "a")]
    good = score_backend(FakeBackend({"a": "यह कागज़ है इसे मोड़ो"}), samples)
    poor = score_backend(FakeBackend({"a": "यह कागच हइ इसो मोड"}), samples)
    assert good.wer < poor.wer


def test_results_sort_by_wer_ascending():
    samples = [sample("एक दो तीन", "a")]
    results = [
        score_backend(FakeBackend({"a": "एक दो चार"}), samples),
        score_backend(FakeBackend({"a": "एक दो तीन"}), samples),
    ]
    ranked = sorted(results, key=lambda r: r.wer)
    assert ranked[0].wer == 0.0


def test_empty_benchmark_result_renders_without_crashing():
    r = BenchmarkResult(backend="x", model="y", rows=[], failures=["all died"],
                        elapsed_sec=0.0)
    assert r.wer is None
    assert r.scored == 0
    assert r.failed == 1


# ================================================================ audio normalisation
def test_normalised_audio_is_16k_mono(intact_session: Path, tmp_path: Path):
    """Backends differ in what wav encodings they accept — onnx-asr rejected FLEURS's
    files outright ("unknown format: 3"). Normalising also makes the comparison fair,
    since every model then sees byte-identical audio."""
    from src.evaluation.benchmark import normalize_audio
    from src.ingest import probe_audio

    raw = (intact_session / "OD90001_2026-01-06-114155.mp3").read_bytes()
    out = normalize_audio(raw, tmp_path / "x.wav")

    probe = probe_audio(out)
    assert probe.sample_rate == 16000
    assert probe.channels == 1
    assert probe.actual_sec == pytest.approx(6.0, abs=0.3)


def test_normalisation_returns_the_requested_path(intact_session: Path, tmp_path: Path):
    from src.evaluation.benchmark import normalize_audio

    raw = (intact_session / "OD90001_2026-01-06-114155.mp3").read_bytes()
    dst = tmp_path / "deep" / "y.wav"
    assert normalize_audio(raw, dst) == dst
    assert dst.is_file()


def test_normalisation_of_junk_bytes_raises(tmp_path: Path):
    from src.evaluation.benchmark import normalize_audio

    with pytest.raises(ValueError):
        normalize_audio(b"not audio at all", tmp_path / "z.wav")
