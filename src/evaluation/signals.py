"""Reference-free quality signals.

We have no human transcript for these five hours, so the bake-off leans on proxies that
catch the specific ways Whisper fails on noisy Hindi classroom audio:

* **looping** — the same phrase repeated, its commonest hallucination on noise
* **silence-filling** — confident text over what is really classroom din
* **script drift** — asked for Hindi, produced Latin or Javanese, which is what the model
  emits when it cannot find speech at all

None of these prove correctness. They reliably catch garbage, which is enough to rank
contenders before a human listens to the shortlist.
"""

from __future__ import annotations

import gzip
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from src.asr.base import Transcript

_REPEAT_WINDOW = 4          # n-gram length; Whisper loops are usually longer than this
_COMPRESSION_ALARM = 2.4    # Whisper's own hallucination threshold

# Fluent, confident, well-formed text that is nonetheless invented. Each model drifts
# toward its own training data, so the tells differ by model. All of these were observed
# in our own bake-off run - see docs/asr-options.md section 1.
_ARTEFACTS: dict[str, tuple[str, ...]] = {
    # Whisper trained on YouTube captions
    "subscribe":      ("सब्सक्राइब", "सबस्क्राइब", "subscribe to", "चैनल को लाइक"),
    "thanks_watching": ("देखने के लिए धन्यवाद", "thanks for watching", "धन्यवाद देखने"),
    # the Hindi fine-tune drifting into its news corpus
    "news_bulletin":  ("प्रदेश के अन्य जिलों", "इस अभियान के तहत", "जागरूकता कार्यक्रम",
                       "संवाददाता", "समाचार"),
}


def repetition_rate(text: str, n: int = _REPEAT_WINDOW) -> float:
    """Fraction of n-gram positions whose n-gram already appeared earlier."""
    tokens = (text or "").split()
    if len(tokens) < n * 2:
        return 0.0

    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    seen: set[tuple] = set()
    duplicates = 0
    for g in grams:
        if g in seen:
            duplicates += 1
        else:
            seen.add(g)
    return duplicates / len(grams)


def compression_ratio(text: str) -> float:
    """Raw length over gzipped length. Repetitive text compresses hard."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(gzip.compress(raw)))


def devanagari_ratio(text: str) -> float:
    """Share of letters written in Devanagari, ignoring digits, spaces and punctuation."""
    letters = [ch for ch in (text or "") if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "ऀ" <= ch <= "ॿ") / len(letters)


# Hindi runs about 2-3 words/second. Anything far above that is a model generating past
# the end of the audio, which is how `vasista22-whisper-hindi-large-v2` scored 84% WER on
# FLEURS while transcribing the actual speech near-perfectly.
_MAX_PLAUSIBLE_WPS = 4.0


def words_per_second(transcript: Transcript) -> float:
    """Output length against audio length. Catches rambling that keyword lists miss."""
    words = len(transcript.text.split())
    return words / transcript.audio_sec if transcript.audio_sec > 0 else 0.0


def artefact_hits(text: str) -> list[str]:
    """Names of known hallucination patterns present in `text`.

    These are the failures the other proxies miss: fluent Devanagari, no looping, high
    model confidence - and completely invented. Catching them needs a list of tells,
    not a statistic.
    """
    lowered = (text or "").lower()
    return sorted(name for name, needles in _ARTEFACTS.items()
                  if any(n.lower() in lowered for n in needles))


def cross_model_agreement(texts: list[str]) -> float | None:
    """Mean pairwise word overlap (Jaccard) between independent transcripts.

    Where two models that share no weights produce the same words, those words are very
    likely real. Where they agree on nothing, at least one is inventing. Returns None if
    there is nothing to compare.
    """
    if len(texts) < 2:
        return None

    bags = [set((t or "").split()) for t in texts]
    scores: list[float] = []
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            union = bags[i] | bags[j]
            scores.append(len(bags[i] & bags[j]) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else None


@dataclass(frozen=True)
class QualitySignals:
    mean_logprob: float | None
    mean_no_speech: float | None
    speech_density: float
    repetition_rate: float
    compression_ratio: float
    devanagari_ratio: float
    segments: int
    chars: int
    artefacts: tuple[str, ...] = ()
    words_per_sec: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.chars == 0

    @property
    def composite(self) -> float:
        """A single 0–1 number for ranking. Documented, not authoritative.

        Confidence and coverage count for it; looping and script drift count against.
        Deliberately simple — a fancier score would imply an accuracy we cannot claim
        without ground truth.
        """
        if self.is_empty:
            return 0.0

        # -3.0 is roughly "certainly wrong", 0.0 is "certain"; clamp and rescale to 0-1.
        # A missing logprob means UNKNOWN, not certain. Scoring it 1.0 handed
        # `theainerd/Wav2Vec2-large-xlsr-hindi` a perfect 1.000 for Devanagari-shaped
        # noise, purely because it reports no confidence at all.
        confidence = 0.5 if self.mean_logprob is None else \
            max(0.0, min(1.0, (self.mean_logprob + 3.0) / 3.0))
        coverage = max(0.0, min(1.0, self.speech_density))
        looping = max(0.0, min(1.0, self.repetition_rate))
        drift = 1.0 - max(0.0, min(1.0, self.devanagari_ratio))

        score = (0.40 * confidence
                 + 0.20 * coverage
                 + 0.25 * (1.0 - looping)
                 + 0.15 * (1.0 - drift))

        # A known artefact means the model was fluent and confident about something it
        # invented. Confidence is then evidence against it, so penalise hard.
        if self.artefacts:
            score *= 0.5

        # More words than the audio could physically contain means the model kept going
        # after the speech stopped. Scale down in proportion to how implausible it is.
        if self.words_per_sec > _MAX_PLAUSIBLE_WPS:
            score *= float(max(0.2, _MAX_PLAUSIBLE_WPS / self.words_per_sec))
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data |= {"is_empty": self.is_empty, "composite": round(self.composite, 4),
                 "artefacts": list(self.artefacts),
                 "words_per_sec": round(self.words_per_sec, 3)}
        return data


def quality_signals(transcript: Transcript) -> QualitySignals:
    text = transcript.text
    no_speech = [s.no_speech_prob for s in transcript.segments if s.no_speech_prob is not None]

    return QualitySignals(
        mean_logprob=transcript.mean_logprob,
        mean_no_speech=sum(no_speech) / len(no_speech) if no_speech else None,
        speech_density=transcript.speech_density,
        repetition_rate=repetition_rate(text),
        compression_ratio=compression_ratio(text),
        devanagari_ratio=devanagari_ratio(text),
        segments=len(transcript.segments),
        chars=len(text),
        artefacts=tuple(artefact_hits(text)),
        words_per_sec=words_per_second(transcript),
    )


__all__ = [
    "QualitySignals",
    "artefact_hits",
    "cross_model_agreement",
    "words_per_second",
    "compression_ratio",
    "devanagari_ratio",
    "quality_signals",
    "repetition_rate",
    "_COMPRESSION_ALARM",
]
