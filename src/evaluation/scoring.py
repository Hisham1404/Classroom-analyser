"""Word and character error rates, for the clips that get a human reference.

Devanagari has its own sentence terminator (danda, U+0964) and its own digits, so the
normaliser cannot just be `str.lower()` plus `string.punctuation`.
"""

from __future__ import annotations

import re
import unicodedata

_DANDA = "।॥"
_WHITESPACE = re.compile(r"\s+")


def normalize_for_scoring(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace.

    Punctuation is dropped by Unicode category so Devanagari danda and Latin marks are
    handled by the same rule, and Devanagari letters — which have no case — pass through
    `casefold` untouched.
    """
    normalized = unicodedata.normalize("NFC", text or "").casefold()
    kept = [
        ch for ch in normalized
        if not unicodedata.category(ch).startswith("P")   # all punctuation, danda included
        and unicodedata.category(ch) != "So"              # stray symbols/emoji
    ]
    return _WHITESPACE.sub(" ", "".join(kept)).strip()


def _levenshtein(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (r != h),  # substitution
            ))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate. Can exceed 1.0 when a model pads with hallucinated words."""
    ref = normalize_for_scoring(reference).split()
    if not ref:
        raise ValueError("reference is empty after normalisation")
    return _levenshtein(ref, normalize_for_scoring(hypothesis).split()) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate — fairer than WER for a language whose ASR output is
    usually *near* the right word rather than a different word entirely."""
    ref = normalize_for_scoring(reference).replace(" ", "")
    if not ref:
        raise ValueError("reference is empty after normalisation")
    hyp = normalize_for_scoring(hypothesis).replace(" ", "")
    return _levenshtein(list(ref), list(hyp)) / len(ref)
