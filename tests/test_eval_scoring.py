"""Scoring transcripts two ways.

1. Against a human reference, when one exists — WER and CER.
2. Without any reference — proxy signals that catch the specific ways Whisper fails on
   noisy Hindi classroom audio: looping, silence-filling, and drifting into the wrong script.

The proxies matter because we have no ground truth for these recordings, and hand-writing
five hours of reference transcript is not on the table.
"""

from __future__ import annotations

import pytest

from src.evaluation.scoring import cer, normalize_for_scoring, wer
from src.evaluation.signals import (
    compression_ratio,
    devanagari_ratio,
    quality_signals,
    repetition_rate,
)
from src.asr.base import Transcript, TranscriptSegment


# ================================================================ normalisation
def test_normalisation_collapses_whitespace():
    assert normalize_for_scoring("यह   क्या  है") == "यह क्या है"


def test_normalisation_strips_the_devanagari_full_stop():
    assert normalize_for_scoring("यह क्या है।") == "यह क्या है"


def test_normalisation_strips_latin_punctuation():
    assert normalize_for_scoring("What is this?") == "what is this"


def test_normalisation_lowercases_latin():
    assert normalize_for_scoring("Okay SIZE") == "okay size"


def test_normalisation_leaves_devanagari_letters_alone():
    assert normalize_for_scoring("कागज़") == "कागज़"


def test_normalisation_of_empty_text():
    assert normalize_for_scoring("   ") == ""


# ================================================================ WER
def test_identical_text_scores_zero():
    assert wer("यह क्या है", "यह क्या है") == 0.0


def test_completely_different_text_scores_one():
    assert wer("एक दो तीन", "क ख ग") == pytest.approx(1.0)


def test_one_substitution_in_three_words():
    assert wer("एक दो तीन", "एक दो चार") == pytest.approx(1 / 3)


def test_one_deletion():
    assert wer("एक दो तीन", "एक दो") == pytest.approx(1 / 3)


def test_one_insertion():
    assert wer("एक दो", "एक दो तीन") == pytest.approx(1 / 2)


def test_wer_ignores_punctuation_and_case():
    assert wer("What is this?", "what is this") == 0.0


def test_empty_hypothesis_against_a_real_reference_scores_one():
    assert wer("एक दो तीन", "") == pytest.approx(1.0)


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        wer("", "कुछ")


def test_wer_can_exceed_one_when_the_model_rambles():
    """Hallucinated padding is worse than saying nothing."""
    assert wer("एक", "एक दो तीन चार पाँच") > 1.0


# ================================================================ CER
def test_cer_identical_is_zero():
    assert cer("कागज़", "कागज़") == 0.0


def test_cer_is_gentler_than_wer_on_a_near_miss():
    """One wrong letter should not count as a whole wrong word."""
    ref, hyp = "कागज", "कागच"
    assert cer(ref, hyp) < wer(ref, hyp)


def test_cer_empty_reference_raises():
    with pytest.raises(ValueError):
        cer("", "कुछ")


# ================================================================ repetition
def test_no_repetition_scores_zero():
    assert repetition_rate("एक दो तीन चार पाँच छह सात") == 0.0


def test_a_looping_hallucination_scores_high():
    """Whisper's classic failure on noise: the same phrase over and over."""
    text = "और लगता आँ उसकना " * 6
    assert repetition_rate(text) > 0.7


def test_text_shorter_than_the_window_scores_zero():
    assert repetition_rate("एक दो") == 0.0


def test_empty_text_scores_zero():
    assert repetition_rate("") == 0.0


def test_a_natural_repeat_barely_registers():
    text = "यह कागज़ है और यह भी कागज़ है लेकिन वह प्लास्टिक है और वह भी"
    assert repetition_rate(text) < 0.3


# ================================================================ compression
def test_compression_ratio_is_higher_for_repetitive_text():
    repetitive = "और लगता आँ " * 20
    varied = "यह कागज़ है वह प्लास्टिक है इसे मोड़कर देखो अब क्या हुआ बताओ"
    assert compression_ratio(repetitive) > compression_ratio(varied)


def test_compression_ratio_of_empty_text_is_zero():
    assert compression_ratio("") == 0.0


# ================================================================ script purity
def test_pure_devanagari_scores_one():
    assert devanagari_ratio("यह क्या है") == pytest.approx(1.0)


def test_pure_latin_scores_zero():
    assert devanagari_ratio("what is this") == 0.0


def test_code_mixed_text_scores_in_between():
    r = devanagari_ratio("यह okay है")
    assert 0.0 < r < 1.0


def test_digits_and_spaces_are_ignored():
    assert devanagari_ratio("यह 123 है") == pytest.approx(1.0)


def test_empty_text_scores_zero():
    assert devanagari_ratio("") == 0.0


def test_the_real_garbled_output_is_still_devanagari():
    """Gibberish in the right script is a different failure from the wrong script —
    the proxies have to tell them apart."""
    assert devanagari_ratio("अदिस गाम बाना ब्या बाखु") == pytest.approx(1.0)


# ================================================================ bundle
def build(text: str, logprob: float = -0.5, no_speech: float = 0.1,
          audio_sec: float = 10.0) -> Transcript:
    return Transcript(
        segments=[TranscriptSegment(0.0, 5.0, text, logprob, no_speech, None)],
        language="hi", language_prob=0.9, backend="stub", model="m", audio_sec=audio_sec,
    )


def test_signals_carry_the_model_confidence():
    s = quality_signals(build("यह क्या है", logprob=-0.35))
    assert s.mean_logprob == pytest.approx(-0.35)
    assert s.mean_no_speech == pytest.approx(0.1)


def test_signals_measure_coverage():
    assert quality_signals(build("यह क्या है", audio_sec=10.0)).speech_density == pytest.approx(0.5)


def test_signals_flag_an_empty_transcript():
    t = Transcript([], "hi", 0.9, "stub", "m", 10.0)
    s = quality_signals(t)
    assert s.is_empty
    assert s.chars == 0
    assert s.repetition_rate == 0.0


def test_signals_spot_a_looping_hallucination():
    s = quality_signals(build("और लगता आँ उसकना " * 6))
    assert s.repetition_rate > 0.7
    assert s.compression_ratio > 2.4          # Whisper's own hallucination threshold


def test_signals_spot_the_wrong_script():
    s = quality_signals(build("apa kabar ini bagus sekali"))
    assert s.devanagari_ratio == 0.0


def test_signals_are_json_serialisable():
    import json
    json.loads(json.dumps(quality_signals(build("यह क्या है")).to_dict()))


def test_good_transcript_beats_a_garbled_one_on_the_composite():
    good = quality_signals(build("यह कागज़ है इसे मोड़कर देखो अब बताओ क्या हुआ", logprob=-0.3))
    bad = quality_signals(build("और लगता आँ " * 12, logprob=-1.6))
    assert good.composite > bad.composite


def test_composite_stays_inside_zero_to_one():
    for t in (build("यह क्या है", logprob=-0.1), build("x" * 200, logprob=-5.0),
              Transcript([], "hi", 0.9, "s", "m", 10.0)):
        assert 0.0 <= quality_signals(t).composite <= 1.0


# ================================================================ known artefacts
from src.evaluation.signals import artefact_hits, cross_model_agreement


def test_the_subscribe_hallucination_is_caught():
    """Groq produced this on 30s of classroom noise — Whisper's YouTube-caption artefact."""
    assert "subscribe" in artefact_hits("सब्सक्राइब करो कुछ भी करो, जो लगे की ये")


def test_thanks_for_watching_is_caught():
    assert artefact_hits("देखने के लिए धन्यवाद")


def test_news_bulletin_boilerplate_is_caught():
    """The Hindi fine-tune drifted into its news training corpus."""
    assert artefact_hits("इसके अलावा प्रदेश के अन्य जिलों में भी इस अभियान के तहत")


def test_ordinary_classroom_hindi_is_not_flagged():
    assert artefact_hits("मतलब आसमान से बादल भी जा रहा था इतना भारी वर्षा हो रहा था") == []


def test_empty_text_is_not_flagged():
    assert artefact_hits("") == []


def test_signals_expose_the_artefact_count():
    s = quality_signals(build("सब्सक्राइब करो और चैनल को लाइक करें"))
    assert s.artefacts
    assert s.composite < quality_signals(build("यह कागज़ है इसे मोड़कर देखो")).composite


# ================================================================ agreement
def test_identical_transcripts_agree_completely():
    assert cross_model_agreement(["यह कागज़ है", "यह कागज़ है"]) == pytest.approx(1.0)


def test_completely_different_transcripts_do_not_agree():
    assert cross_model_agreement(["एक दो तीन", "क ख ग"]) == pytest.approx(0.0)


def test_partial_overlap_scores_in_between():
    a = cross_model_agreement(["और एक बार आगे पूरा चारों भाग पानी",
                               "और एक बार आखे पूरा चारों धार पानी"])
    assert 0.3 < a < 1.0


def test_agreement_needs_at_least_two_transcripts():
    assert cross_model_agreement(["अकेला"]) is None
    assert cross_model_agreement([]) is None


def test_agreement_ignores_an_empty_transcript():
    assert cross_model_agreement(["यह कागज़ है", ""]) == pytest.approx(0.0)


def test_agreement_is_symmetric():
    a = cross_model_agreement(["एक दो तीन चार", "एक दो पाँच"])
    b = cross_model_agreement(["एक दो पाँच", "एक दो तीन चार"])
    assert a == pytest.approx(b)


# ================================================================ unknown confidence
def test_unknown_confidence_is_not_treated_as_certainty():
    """A model that reports no logprob must not get full confidence marks.

    Measured: `theainerd/Wav2Vec2-large-xlsr-hindi` produced Devanagari-shaped noise and
    scored a perfect 1.000, purely because it reported no confidence at all.
    """
    known_good = quality_signals(build("यह कागज़ है इसे मोड़कर देखो", logprob=-0.2))
    unknown = quality_signals(build("यह कागज़ है इसे मोड़कर देखो", logprob=None))
    assert unknown.composite < known_good.composite


def test_unknown_confidence_still_beats_measured_garbage():
    unknown = quality_signals(build("यह कागज़ है इसे मोड़कर देखो", logprob=None))
    bad = quality_signals(build("यह कागज़ है इसे मोड़कर देखो", logprob=-2.6))
    assert unknown.composite > bad.composite


def test_nothing_can_reach_a_perfect_score_without_reporting_confidence():
    assert quality_signals(build("यह कागज़ है", logprob=None)).composite < 1.0


# ================================================================ speech rate
from src.evaluation.signals import words_per_second


def test_natural_speech_rate_is_around_two_to_three_words_per_second():
    t = build("यह कागज़ है इसे मोड़कर देखो अब बताओ क्या हुआ", audio_sec=10.0)
    assert 0.5 < words_per_second(t) < 4.0


def test_a_rambling_model_exceeds_any_plausible_rate():
    """Measured on FLEURS: the Hindi fine-tune transcribes the audio correctly, then
    keeps generating news prose past the end of it."""
    t = build(" ".join(["शब्द"] * 120), audio_sec=15.0)
    assert words_per_second(t) > 4.0


def test_speech_rate_of_an_empty_transcript_is_zero():
    t = Transcript([], "hi", 0.9, "stub", "m", 10.0)
    assert words_per_second(t) == 0.0


def test_signals_expose_the_speech_rate():
    s = quality_signals(build("यह कागज़ है", audio_sec=10.0))
    assert s.words_per_sec == pytest.approx(0.3)


def test_an_implausible_speech_rate_is_penalised():
    normal = quality_signals(build("यह कागज़ है इसे मोड़कर देखो अब बताओ", audio_sec=10.0))
    rambling = quality_signals(build(" ".join(["शब्द"] * 120), audio_sec=10.0))
    assert rambling.composite < normal.composite


def test_the_rate_penalty_does_not_punish_ordinary_transcripts():
    """A dense but real 30-second turn must not be flagged."""
    dense = quality_signals(build(" ".join(["शब्द"] * 80), audio_sec=30.0))
    assert dense.composite > 0.5
