"""Teacher/student from real speaker diarization.

Replaces the energy heuristic, which was measured at **41.7% accuracy** against
transcript-register labels — worse than chance, and beaten 100% to 41.7% by a constant
"always teacher" (`docs/data-notes.md` §8). Confidence did not discriminate either, so
there was no threshold to tune. The feature itself was wrong.

Diarization brings a much stronger prior. Energy assumed *the teacher is nearest the mic*,
which failed because the phone hears one room at one level — 2.6 dB end to end on one
session. This assumes instead:

    the teacher is one person who talks a lot;
    students are many people who each talk little.

That holds regardless of where the phone is sitting, which is exactly where energy broke.

Per-child identity is still not claimed. Every non-teacher speaker collapses to `student`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.attribution import Role
from src.signal_layer import Segment, SegmentKind

# pyannote 4.x redirects the old "speaker-diarization-3.1" id to this pipeline, and it is
# separately gated - accepting the 3.1 conditions is not enough.
PIPELINE_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def teacher_speaker(turns: list[SpeakerTurn]) -> tuple[str | None, float]:
    """Which diarized speaker is the teacher, and how clear-cut it is.

    Confidence is the share of speaking time between the top speaker and the runner-up:
    one voice holding twice the floor of the next is a clear teacher, two voices splitting
    it evenly is not.
    """
    if not turns:
        return None, 0.0

    totals: dict[str, float] = {}
    for t in turns:
        totals[t.speaker] = totals.get(t.speaker, 0.0) + t.duration

    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    top_label, top_time = ranked[0]
    if len(ranked) == 1:
        return top_label, 1.0

    runner_up = ranked[1][1]
    denominator = top_time + runner_up
    margin = (top_time - runner_up) / denominator if denominator > 0 else 0.0
    return top_label, max(0.0, min(1.0, margin))


# Two overlaps this close are the same overlap; float arithmetic on turn boundaries
# will not reproduce them bit-for-bit.
_TIE_SEC = 1e-6


def speaker_for_span(turns: list[SpeakerTurn], start: float, end: float) -> str | None:
    """Whoever spoke for most of this span. None if diarization never covered it.

    Turns overlap, and a teacher's turn routinely *encloses* a student's - on
    OD11163_2026-01-28 all 25 student turns did, on OD11165_2026-01-06 all 38. Both
    then cover the student's span identically, so taking the first strict maximum
    handed it to the enclosing turn every single time: 100% of student turns were
    labelled teacher, M1 reported 100% teacher talk, and M3 reported zero exchanges on
    sessions where M5 had just found 28 answered questions.

    Ties therefore go to the speaker whose turn is *most contained* in the span. A
    student turn matching the span scores 1.0; a teacher turn blanketing twenty
    seconds of it scores 0.05. Containment is only ever a tie-break, so a speaker who
    genuinely holds more of the span still wins - a student with the floor is not
    displaced by a teacher chipping in for a second.
    """
    best_label, best_overlap, best_containment = None, 0.0, 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap <= 0:
            continue
        duration = t.end - t.start
        containment = overlap / duration if duration > 0 else 1.0

        if (overlap > best_overlap + _TIE_SEC
                or (abs(overlap - best_overlap) <= _TIE_SEC
                    and containment > best_containment)):
            best_label, best_overlap, best_containment = t.speaker, overlap, containment
    return best_label


def speech_spans(turns: list[SpeakerTurn]) -> list[tuple[float, float]]:
    """Where anyone was speaking: the union of every speaker's turns.

    pyannote's segmentation is a speech detector before it is a speaker detector, and
    it is a much better one than Silero on this audio. Measured on OD11166_2026-01-12,
    142 s of the 161 s of diarized speech fell inside segments Silero had left out, and
    the "loud but not speech" rule then filed them as hands-on activity - so 67 s of the
    70 s of student speech was published as 1.6 s.

    Turns overlap and are not in clock order, so this is a proper interval union rather
    than a sort.
    """
    if not turns:
        return []

    ordered = sorted(((t.start, t.end) for t in turns if t.end > t.start))
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:                       # overlapping, or merely touching
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def assign_roles_from_turns(segments: list[Segment],
                            turns: list[SpeakerTurn]) -> list[Segment]:
    """Label speech segments teacher/student using diarized turns. Returns new segments.

    Confidence is *not* the airtime margin alone. Measured on OD11166_2026-01-20: a 0.06
    margin still produced 75% correct labels, because the margin says how dominant the
    teacher is, not how trustworthy the split is. What does carry reliability is
    **coverage** - how much of the speech diarization actually spoke for. So:

        confidence = coverage x (0.5 + 0.5 x margin)

    An evenly split room still clears the floor; a room diarization barely covered does not.
    """
    teacher, margin = teacher_speaker(turns)

    speech_segments = [s for s in segments if s.kind is SegmentKind.SPEECH]
    covered = sum(1 for s in speech_segments
                  if speaker_for_span(turns, s.start, s.end) is not None)
    coverage = covered / len(speech_segments) if speech_segments else 0.0
    confidence = coverage * (0.5 + 0.5 * margin)

    out: list[Segment] = []

    for s in segments:
        if s.kind is not SegmentKind.SPEECH or teacher is None:
            out.append(s.with_role(None, None))
            continue

        speaker = speaker_for_span(turns, s.start, s.end)
        if speaker is None:
            # Diarization found no voice here. Saying nothing beats guessing — guessing is
            # what produced 41.7%.
            out.append(s.with_role(None, None))
            continue

        role = Role.TEACHER if speaker == teacher else Role.STUDENT
        out.append(s.with_role(role, confidence))

    return out


# Shortest thing we will call a turn. Measured: 68 of 185 pieces came back under 0.5 s on
# one 4-minute window. At that resolution, on overlapping classroom audio, a "turn" is a
# diarization boundary artefact rather than someone speaking - and each one manufactures
# two spurious speaker switches. Anything shorter is absorbed into its neighbour.
MIN_PIECE_SEC = 0.30
# A hole shorter than this between two turns by the same speaker is a breath, not a
# speaker change. pyannote fragments heavily on overlapping classroom audio - 185 pieces
# in four minutes, median 0.68 s - and leaving that unmerged inflated M2 and M3 and
# collapsed M4 from 241 s to 16 s.
MERGE_GAP_SEC = 1.0


def _merge_adjacent(pieces: list[Segment]) -> list[Segment]:
    """Join neighbouring pieces that share a role, absorbing short unattributed holes."""
    if not pieces:
        return []

    merged = [pieces[0]]
    for piece in pieces[1:]:
        prev = merged[-1]

        same_role = piece.role == prev.role and piece.kind is prev.kind
        # An unattributed sliver flanked by the same speaker is a pause inside their turn.
        bridgeable = (piece.role is None and piece.kind is prev.kind
                      and prev.role is not None and piece.duration <= MERGE_GAP_SEC)

        if same_role or bridgeable:
            merged[-1] = Segment(prev.start, piece.end, prev.kind, prev.rms_db,
                                 prev.flatness, prev.role, prev.role_conf)
        else:
            merged.append(piece)
    return merged


def split_segments_by_speaker(segments: list[Segment],
                              turns: list[SpeakerTurn]) -> list[Segment]:
    """Cut speech segments at speaker changes instead of labelling them wholesale.

    VAD segments run up to 63 s on this audio while diarization turns are seconds long.
    Assigning each segment to whoever dominated it erased every student turn and made M1
    report **100% teacher talk** on three sessions - a real number, confidently wrong.

    Splitting keeps the timeline tiling: the pieces of a segment cover exactly the same
    span, so durations still sum to the recording length. Sub-segments inherit the
    parent's loudness and flatness, which is an approximation - those were measured over
    the whole segment - but nothing downstream re-derives speech/non-speech from them.
    """
    teacher, margin = teacher_speaker(turns)

    speech_segments = [s for s in segments if s.kind is SegmentKind.SPEECH]
    covered = sum(1 for s in speech_segments
                  if speaker_for_span(turns, s.start, s.end) is not None)
    coverage = covered / len(speech_segments) if speech_segments else 0.0
    confidence = coverage * (0.5 + 0.5 * margin)

    out: list[Segment] = []
    for seg in segments:
        if seg.kind is not SegmentKind.SPEECH or teacher is None or not turns:
            out.append(seg.with_role(None, None))
            continue

        # Boundaries where the speaker could change inside this segment.
        edges = {seg.start, seg.end}
        for t in turns:
            for edge in (t.start, t.end):
                if seg.start < edge < seg.end:
                    edges.add(edge)
        ordered = sorted(edges)

        pieces: list[Segment] = []
        for start, end in zip(ordered, ordered[1:]):
            if end - start < MIN_PIECE_SEC:
                continue
            speaker = speaker_for_span(turns, start, end)
            role = (None if speaker is None
                    else Role.TEACHER if speaker == teacher else Role.STUDENT)
            piece = Segment(start=start, end=end, kind=seg.kind,
                            rms_db=seg.rms_db, flatness=seg.flatness)
            pieces.append(piece.with_role(role, confidence if role else None))

        if not pieces:
            out.append(seg.with_role(None, None))
            continue

        pieces = _merge_adjacent(pieces)

        # Absorbing slivers left gaps; stretch the neighbours so the timeline still tiles.
        pieces[0] = Segment(seg.start, pieces[0].end, pieces[0].kind, pieces[0].rms_db,
                            pieces[0].flatness, pieces[0].role, pieces[0].role_conf)
        pieces[-1] = Segment(pieces[-1].start, seg.end, pieces[-1].kind, pieces[-1].rms_db,
                             pieces[-1].flatness, pieces[-1].role, pieces[-1].role_conf)
        for i in range(len(pieces) - 1):
            if pieces[i].end != pieces[i + 1].start:
                pieces[i] = Segment(pieces[i].start, pieces[i + 1].start, pieces[i].kind,
                                    pieces[i].rms_db, pieces[i].flatness,
                                    pieces[i].role, pieces[i].role_conf)
        out.extend(_merge_adjacent(pieces))

    return out


# --------------------------------------------------------------------------- pyannote
def _load_pipeline(token: str):
    """Build the pyannote pipeline. Separated so tests can stand in for it."""
    from pyannote.audio import Pipeline

    config.configure_hf_cache()
    return Pipeline.from_pretrained(PIPELINE_ID, token=token)


def diarize(path: Path, token: str | None = None,
            num_speakers: int | None = None) -> list[SpeakerTurn]:
    """Run pyannote over one audio file.

    Needs an HF token and accepted conditions on **three** repos, not the one the model
    card mentions: `speaker-diarization-community-1` (what 4.x actually loads),
    `segmentation-3.0`, and `wespeaker-voxceleb-resnet34-LM`.

    The audio is decoded here rather than handed over as a path. pyannote 4.x decodes via
    torchcodec, whose DLL needs FFmpeg's *shared libraries* on the search path - absent on
    a normal Windows box even with ffmpeg.exe installed. We already decode with ffmpeg, so
    passing a tensor sidesteps the whole problem.
    """
    import numpy as np
    import torch

    from src.signal_layer import SAMPLE_RATE, load_waveform

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. pyannote is gated: accept the conditions on "
            "pyannote/speaker-diarization-community-1, pyannote/segmentation-3.0 and "
            "pyannote/wespeaker-voxceleb-resnet34-LM, then export a read token."
        )

    wave = load_waveform(Path(path), SAMPLE_RATE)
    # pyannote wants (channel, time); a bare 1-D array is silently misread as one sample
    # per channel.
    tensor = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)

    pipeline = _load_pipeline(token)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipeline({"waveform": tensor, "sample_rate": SAMPLE_RATE}, **kwargs)

    # 4.x wraps the Annotation in a DiarizeOutput; older versions return it directly.
    annotation = getattr(result, "speaker_diarization", result)

    return [SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)]


def assign_roles_by_diarization(segments: list[Segment], path: Path,
                                token: str | None = None) -> list[Segment]:
    """Convenience: diarize the file and label the segments in one call."""
    return assign_roles_from_turns(segments, diarize(path, token=token))
