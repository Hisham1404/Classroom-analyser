"""Stage 05 — the engagement metrics, M1 to M6.

Each carries its own formula, explanation and interpretation, because that is exactly what
the assignment grades. Each also carries a confidence, so a metric that cannot be trusted
can be suppressed individually rather than dragging the whole report down with it.

The split that matters:

* **M1-M4 are acoustic.** They come from VAD, loudness and turn order, so they keep working
  when the transcript is unusable — which on this audio it sometimes is.
* **M5 is lexical.** It needs question detection, so it needs a transcript, so it is the
  first thing to go when the room gets loud.
* **M6 governs.** Below the floor, nothing lexical is reported at all.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from src import config
from src.asr.base import Transcript
from src.attribution import Role, talk_time
from src.models import Roster
from src.signal_layer import Segment, SegmentKind


class Verdict(str, Enum):
    USABLE = "usable"
    PARTIAL = "partial"
    UNRELIABLE = "unreliable"


# Hindi interrogatives. Matched as whole words so "कयामत" does not trigger on "क्या".
_QUESTION_WORDS = (
    "क्या", "कौन", "कौनसा", "कहाँ", "कहां", "कब", "क्यों", "कैसे", "कितना", "कितने",
    "कितनी", "किसने", "किसका", "किसकी", "किसे", "बताओ", "बताइए",
)
_QUESTION_RE = re.compile(r"(?:^|\s)(" + "|".join(map(re.escape, _QUESTION_WORDS)) + r")(?=\s|$)")


@dataclass(frozen=True)
class Metric:
    key: str
    value: float | None
    unit: str | None
    band: str
    formula: str
    explanation: str
    interpretation: str
    confidence: float
    benchmark: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else round(self.value, 4),
            "unit": self.unit,
            "band": self.band,
            "benchmark": self.benchmark,
            "confidence": round(self.confidence, 3),
            "formula": self.formula,
            "explanation": self.explanation,
            "interpretation": self.interpretation,
        }


@dataclass
class MetricSet:
    metrics: dict[str, Metric]
    totals: dict[str, float]
    confidence: float
    verdict: Verdict
    confidence_parts: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            # `str(Role.TEACHER)` is "Role.TEACHER" - the enum repr, not a name anyone
            # reading the JSON would look for. The timeline uses "teacher"/"student".
            "totals_sec": {(k.value if isinstance(k, Role) else str(k)): round(v, 2)
                           for k, v in self.totals.items()},
            "confidence": {
                "overall": round(self.confidence, 3),
                "components": {k: round(v, 3) for k, v in self.confidence_parts.items()},
                "verdict": self.verdict.value,
            },
        }


# --------------------------------------------------------------------------- helpers
def is_question(text: str) -> bool:
    """Lexical question detection. Deliberately simple and deliberately fragile.

    Hindi has no inverted word order for questions, so an interrogative word or a question
    mark is most of what there is to go on without prosody.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    return "?" in cleaned or bool(_QUESTION_RE.search(cleaned))


def text_for_span(transcript: Transcript | None, start: float, end: float) -> str:
    """Whatever the ASR heard between two timestamps."""
    if transcript is None:
        return ""
    parts = [s.text for s in transcript.segments
             if s.end > start and s.start < end and s.text]
    return " ".join(parts).strip()


def _speech(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if s.kind is SegmentKind.SPEECH and s.role]


def longest_stretch(segments: list[Segment], role: Role) -> float:
    """Longest unbroken run by one speaker.

    Short pauses stay inside the stretch — a two-second breath is the same piece of talking.
    Another speaker, or a long gap, ends it.
    """
    best = current = 0.0
    run_start = run_end = None

    for s in segments:
        is_mine = s.kind is SegmentKind.SPEECH and s.role == role
        if is_mine:
            if run_end is not None and (s.start - run_end) <= config.STRETCH_MERGE_GAP_SEC:
                run_end = s.end
            else:
                run_start, run_end = s.start, s.end
            current = run_end - run_start
            best = max(best, current)
        elif s.kind is SegmentKind.SPEECH:
            run_start = run_end = None          # someone else took the floor
        elif run_end is not None and (s.end - run_end) > config.STRETCH_MERGE_GAP_SEC:
            run_start = run_end = None          # too long a gap to still be one stretch
    return best


class _Turn(NamedTuple):
    """The minimum a wait-time calculation needs, so the diarization turns and the
    energy-path segments can go through one code path."""

    start: float
    end: float
    speaker: str


# Two turns reported as sharing a boundary to the microsecond did not have a silence
# between them that anything measured; the boundary was snapped.
_SNAPPED_SEC = 1e-6


def wait_times_from_turns(turns, teacher_label: str,
                          transcript: Transcript | None) -> list[float]:
    """Silences between a teacher question and the next student voice.

    Measured from raw diarization turns, not from the timeline. Splitting segments at
    speaker changes makes them tile exactly, so a student turn begins the instant the
    teacher's ends and every gap computes to 0.00. The turns keep the real silences.

    Turns overlap, so this cannot walk the list in pairs the way a timeline can. Doing
    that on the real sessions threw away the questions the teacher followed with more
    talking - two thirds of them on one session - and paired long teacher turns with
    student turns nested inside them, producing gaps as negative as -11 s.
    """
    if transcript is None or not turns or teacher_label is None:
        return []

    ordered = sorted(turns, key=lambda t: t.start)
    others = [t for t in ordered if t.speaker != teacher_label]
    waits: list[float] = []

    for question in ordered:
        if question.speaker != teacher_label:
            continue
        if not is_question(text_for_span(transcript, question.start, question.end)):
            continue

        # A student already talking as the question lands did not wait at all. That is
        # an observation of zero, unlike a snapped boundary, so it counts.
        if any(o.start < question.end < o.end for o in others):
            waits.append(0.0)
            continue

        after = [o.start - question.end for o in others if o.start >= question.end]
        if not after:
            continue
        gap = min(after)
        if _SNAPPED_SEC < gap <= config.WAIT_TIME_MAX_SEC:
            waits.append(gap)
    return waits


def _band(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    return "high" if value > high else "normal"


# --------------------------------------------------------------------------- M6
def _confidence(segments: list[Segment], audio_sec: float, completeness: float | None,
                transcript: Transcript | None) -> tuple[float, dict[str, float]]:
    speech = _speech(segments)
    spoken = sum(s.duration for s in speech)

    density = min(1.0, spoken / audio_sec) if audio_sec > 0 else 0.0
    # Density is scaled: a classroom at 50% speech is a healthy recording, not half-bad.
    density_score = min(1.0, density / 0.5)

    confs = [s.role_conf for s in speech if s.role_conf is not None]
    attribution = sum(confs) / len(confs) if confs else 0.0

    complete = 1.0 if completeness is None else max(0.0, min(1.0, completeness))

    if transcript is None or transcript.mean_logprob is None:
        asr = 0.5                                # unknown, not perfect
    else:
        asr = max(0.0, min(1.0, (transcript.mean_logprob + 3.0) / 3.0))

    parts = {"speech_density": density_score, "attribution": attribution,
             "completeness": complete, "asr": asr}
    overall = sum(config.CONF_WEIGHTS[k] * v for k, v in parts.items())
    return max(0.0, min(1.0, overall)), parts


# --------------------------------------------------------------------------- main
def compute_metrics(segments: list[Segment], roster: Roster, audio_sec: float,
                    transcript: Transcript | None = None,
                    completeness: float | None = None,
                    speaker_turns=None) -> MetricSet:
    """M1-M6 for one session."""
    totals = talk_time(segments)
    speech = _speech(segments)
    teacher_sec = totals[Role.TEACHER]
    student_sec = totals[Role.STUDENT]
    spoken = teacher_sec + student_sec
    minutes = audio_sec / 60.0 if audio_sec > 0 else 0.0

    confidence, parts = _confidence(segments, audio_sec, completeness, transcript)

    # M1-M4 all rest on knowing who was speaking. Measured on OD11163_2025-12-23: every
    # segment fell between -27.1 and -29.7 dB, a 2.6 dB spread with nothing to cluster on.
    # The split was noise, every role confidence came back under the floor - and M1 still
    # reported a confident-looking 45%. Below the floor these are not weak numbers, they
    # are no number at all.
    attribution_conf = parts["attribution"]
    split_usable = attribution_conf >= config.ROLE_CONF_FLOOR
    split_conf = min(confidence, attribution_conf)
    verdict = (Verdict.USABLE if confidence >= config.CONF_USABLE
               else Verdict.PARTIAL if confidence >= config.CONF_UNRELIABLE
               else Verdict.UNRELIABLE)

    # ---- M1 ----------------------------------------------------------------
    ttr = (teacher_sec / spoken) if (spoken > 0 and split_usable) else None
    m1 = Metric(
        key="M1_teacher_talk_ratio", value=ttr, unit="ratio",
        band=_band(ttr, config.TEACHER_TALK_BENCHMARK, config.TEACHER_TALK_HIGH),
        benchmark=config.TEACHER_TALK_BENCHMARK, confidence=split_conf,
        formula="teacher_speech_sec / (teacher_speech_sec + student_speech_sec)",
        explanation=("Share of *spoken* time held by the teacher. Non-speech is excluded "
                     "on purpose, so hands-on noise does not dilute it."),
        interpretation=("Reported only when the teacher/student split is separable; below that it is withheld rather than guessed. Observed average in conventional classrooms is about 61% teacher "
                        "talk. A hands-on maker session should sit below 40%; above 70% the "
                        "activity has become a lecture."),
    )

    # ---- M2 ----------------------------------------------------------------
    student_turns = sum(1 for s in speech if s.role == Role.STUDENT)
    spr = ((student_turns / roster.total) / minutes * 60.0
           if roster.total > 0 and minutes > 0 and split_usable else None)
    m2 = Metric(
        key="M2_student_participation", value=spr, unit="turns/child/hour",
        band=_band(spr, 1.0, 6.0), benchmark=None, confidence=split_conf,
        formula="(student_turns / roster_total) / lesson_minutes * 60",
        explanation=("Student turns per child per hour, normalised by the headcount their "
                     "own app recorded. 8 students and 27 students cannot be compared on "
                     "raw turn counts."),
        interpretation=("Depends on the speaker split, so it is withheld when that split is a guess. High student talk time with a low value here is the classic sign "
                        "that two or three confident children are doing all the talking."),
    )

    # ---- M3 ----------------------------------------------------------------
    switches = sum(1 for a, b in zip(speech, speech[1:]) if a.role != b.role)
    density = switches / minutes if (minutes > 0 and split_usable) else None
    m3 = Metric(
        key="M3_interaction_density", value=density, unit="switches/min",
        band=_band(density, 0.5, 4.0), benchmark=None, confidence=split_conf,
        formula="speaker_switches(teacher <-> student) / lesson_minutes",
        explanation=("How often the floor changes hands. Needs no transcript at all, so it "
                     "survives audio the ASR cannot read."),
        interpretation=("Depends on the speaker split. High means genuine back-and-forth. Low with a high talk ratio is a "
                        "monologue; low with a low talk ratio is unstructured activity with "
                        "no teacher check-ins."),
    )

    # ---- M4 ----------------------------------------------------------------
    stretch = longest_stretch(segments, Role.TEACHER) if split_usable else None
    m4 = Metric(
        key="M4_longest_teacher_stretch", value=stretch, unit="sec",
        band=_band(stretch, 60.0, 300.0), benchmark=None, confidence=split_conf,
        formula="max(continuous teacher speech, merging pauses under 3s)",
        explanation=("The longest the teacher talked without stopping. The most concrete "
                     "number on the page, and the easiest to picture changing."),
        interpretation=("Depends on the speaker split. In a 60-minute maker session anything over about five minutes "
                        "means the making stopped."),
    )

    # ---- M5 ----------------------------------------------------------------
    waits: list[float] = []
    if transcript is not None and verdict is not Verdict.UNRELIABLE and split_usable:
        if speaker_turns:
            from src.diarization import teacher_speaker

            label, _ = teacher_speaker(speaker_turns)
            waits = wait_times_from_turns(speaker_turns, label, transcript)
        else:
            # The energy path leaves real gaps between segments, so it can still be
            # measured there. Same rules, so it goes through the same function.
            pseudo = [_Turn(seg.start, seg.end,
                            "teacher" if seg.role is Role.TEACHER else "student")
                      for seg in speech if seg.role is not None]
            waits = wait_times_from_turns(pseudo, "teacher", transcript)

    # A median over one pause is not a median. The first real run published 0.00 s off
    # a single sample, which is the kind of number this pipeline exists to refuse.
    enough = len(waits) >= config.WAIT_TIME_MIN_SAMPLES
    wait = statistics.median(waits) if enough else None
    # Full marks at twice the floor: three pairs is publishable, not authoritative.
    sample_factor = min(1.0, len(waits) / (2 * config.WAIT_TIME_MIN_SAMPLES)) if enough else 0.0
    m5_conf = split_conf * (0.6 if transcript is not None else 0.0) * sample_factor
    m5 = Metric(
        key="M5_wait_time_1", value=wait, unit="sec",
        band=_band(wait, config.WAIT_TIME_LOW_SEC, 8.0),
        benchmark=config.WAIT_TIME_TARGET_SEC, confidence=m5_conf,
        formula="median(start_of_next_student_turn - end_of_teacher_question)",
        explanation=("Mary Budd Rowe's wait time: the pause after a question before a "
                     "student answers. Measured from raw diarization turns, which keep the "
                     "real silences - the segment timeline tiles exactly, so gaps there are "
                     "always zero. Depends on reading questions out of the transcript, so "
                     "it is the first metric to become unreliable in a noisy room."),
        interpretation=("Below 1s, answers are short and students say 'I don't know'. At 3s "
                        "or more, responses get longer, more correct, and more students "
                        "join in. It is the single most replicated finding in this field."
                        + (f" Measured from {len(waits)} question-answer pairs." if enough
                           else f" Withheld: only {len(waits)} question-answer "
                                f"{'pair was' if len(waits) == 1 else 'pairs were'} found, "
                                f"below the {config.WAIT_TIME_MIN_SAMPLES} needed.")),
    )

    return MetricSet(
        metrics={m.key: m for m in (m1, m2, m3, m4, m5)},
        totals=totals, confidence=confidence, verdict=verdict, confidence_parts=parts,
    )
