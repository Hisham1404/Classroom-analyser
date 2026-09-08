"""Stage 04 — teacher or student.

**Binary, not per-child.** 8-27 children on one phone mic is not a solvable speaker
identification problem, and a demo that claims otherwise falls apart the moment someone
asks how. What is solvable is the split between the person holding the phone and everyone
else.

The prior doing the work: the recording is made on the **teacher's own device**, so she is
consistently the nearest, loudest, highest-SNR voice in the room. Two clusters on
per-segment loudness recover that; the louder cluster is her.

Nothing here is validated yet. `scripts/make_labelling_sheet.py` produces the sheet that
turns this from an assertion into a measured accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src import config
from src.signal_layer import Segment, SegmentKind


class Role(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


@dataclass(frozen=True)
class _Cluster:
    centre: float
    members: list[int]


def _two_means(values: np.ndarray, iterations: int = 40) -> tuple[np.ndarray, float, float]:
    """1-D k-means with k=2 and a deterministic start.

    Seeded from the quietest and loudest segment rather than at random, which makes the
    result reproducible *and* encodes the prior: one cluster is near the mic, one is not.
    """
    low, high = float(values.min()), float(values.max())
    centres = np.array([low, high], dtype=np.float64)

    labels = np.zeros(values.size, dtype=np.int64)
    for _ in range(iterations):
        labels = (np.abs(values[:, None] - centres[None, :])).argmin(axis=1)
        moved = False
        for k in (0, 1):
            members = values[labels == k]
            if members.size:
                new = float(members.mean())
                if not np.isclose(new, centres[k]):
                    centres[k], moved = new, True
        if not moved:
            break
    return labels, float(centres[0]), float(centres[1])


def assign_roles(segments: list[Segment]) -> list[Segment]:
    """Label every speech segment teacher or student. Returns new segments.

    Non-speech is left unassigned — hands-on noise has no speaker.
    """
    speech_idx = [i for i, s in enumerate(segments) if s.kind is SegmentKind.SPEECH]
    out = list(segments)

    if not speech_idx:
        return out

    if len(speech_idx) == 1:
        # Nothing to compare against. The phone is the teacher's, so assume it is her —
        # but this is an assumption, and the confidence has to say so.
        i = speech_idx[0]
        out[i] = segments[i].with_role(Role.TEACHER, 0.0)
        return out

    loudness = np.array([segments[i].rms_db for i in speech_idx], dtype=np.float64)
    spread = float(loudness.max() - loudness.min())

    if spread < 1e-6:
        # Every voice identical: no split exists. Say so rather than inventing one.
        for i in speech_idx:
            out[i] = segments[i].with_role(Role.TEACHER, 0.0)
        return out

    labels, centre_a, centre_b = _two_means(loudness)
    teacher_label = 0 if centre_a > centre_b else 1
    separation = abs(centre_a - centre_b)

    for position, i in enumerate(speech_idx):
        is_teacher = labels[position] == teacher_label
        role = Role.TEACHER if is_teacher else Role.STUDENT

        # How far inside its own cluster this segment sits, relative to the gap between
        # clusters. Segments near the boundary are the ones we are least sure about.
        own = centre_a if labels[position] == 0 else centre_b
        other = centre_b if labels[position] == 0 else centre_a
        margin = (abs(loudness[position] - other) - abs(loudness[position] - own))
        confidence = float(np.clip(margin / max(separation, 1e-9), 0.0, 1.0))

        # A tight split is a weak split however cleanly the points fall either side of it.
        confidence *= float(np.clip(separation / 12.0, 0.0, 1.0))
        out[i] = segments[i].with_role(role, confidence)

    return out


def talk_time(segments: list[Segment]) -> dict:
    """Seconds per role, plus hands-on and dead time.

    Every second of the recording lands in exactly one bucket, so these sum to the
    session length and can be used as metric denominators directly.
    """
    totals = {Role.TEACHER: 0.0, Role.STUDENT: 0.0,
              "handson": 0.0, "dead": 0.0, "unattributed": 0.0}

    for s in segments:
        if s.kind is SegmentKind.HANDSON:
            totals["handson"] += s.duration
        elif s.kind is SegmentKind.DEAD:
            totals["dead"] += s.duration
        elif s.role == Role.TEACHER:
            totals[Role.TEACHER] += s.duration
        elif s.role == Role.STUDENT:
            totals[Role.STUDENT] += s.duration
        else:
            totals["unattributed"] += s.duration

    return totals
