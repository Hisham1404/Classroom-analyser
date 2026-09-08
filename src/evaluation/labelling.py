"""Turn "the classifier works" into a number.

The teacher/student split rests on one assumption — that the phone belongs to the teacher,
so she is the loudest voice. Plausible, and completely unverified until somebody listens.
This builds the sheet they listen with, reads it back, and scores it.

The rule that shapes the design: **an unlabelled row is not a correct row.** The tempting
bug is to treat a blank truth column as agreement, which reports 100% accuracy on a sheet
nobody has filled in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from src.attribution import Role
from src.signal_layer import Segment, SegmentKind

# Row ids look like SESSION_A#0007 — the '#' has to be in the class or nothing matches.
# The transcript column is last and may contain anything, so it is not captured here.
_ROW = re.compile(
    r"^\|\s*`?([\w.#\-]+)`?\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|")


@dataclass(frozen=True)
class LabelRow:
    row_id: str
    start: float
    end: float
    predicted: Role
    confidence: float
    truth: Role | None = None
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_sheet(session_id: str, segments: list[Segment],
                transcript=None) -> list[LabelRow]:
    """One row per speech segment. Hands-on noise has no speaker to label.

    The transcript is included when available because linguistic register is independent
    evidence: "आज हमारे क्लास में पहले आपको बताना है" is a teacher whatever the loudness
    said. It is a cross-check on an energy-based split, not a replacement for listening.
    """
    from src.metrics import text_for_span

    rows: list[LabelRow] = []
    for i, s in enumerate(segments):
        if s.kind is not SegmentKind.SPEECH or s.role is None:
            continue
        rows.append(LabelRow(
            row_id=f"{session_id}#{i:04d}",
            start=s.start,
            end=s.end,
            predicted=Role(s.role),
            confidence=float(s.role_conf or 0.0),
            text=text_for_span(transcript, s.start, s.end) if transcript else "",
        ))
    return rows


def render_sheet(rows: list[LabelRow]) -> str:
    """Markdown with a blank Truth column, ordered so it can be worked through linearly."""
    lines = [
        "# Teacher / student — hand labelling",
        "",
        "Play each span and write `teacher` or `student` in the **Truth** column.",
        "Leave a row blank if you genuinely cannot tell — a blank is excluded from the",
        "score, which is the honest treatment. Do **not** guess to fill the sheet.",
        "",
        "`Pred` and `Conf` are the model's answer. Try not to read them before deciding.",
        "",
        "| Row | Start | End | Pred | Conf | Truth | Transcript |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for r in rows:
        truth = r.truth.value if r.truth else ""
        # A pipe in the transcript would split the table cell in two.
        text = " ".join((r.text or "").replace("|", "/").split())[:110]
        lines.append(f"| `{r.row_id}` | {r.start:.1f} | {r.end:.1f} | "
                     f"{r.predicted.value} | {r.confidence:.2f} | {truth} | {text} |")
    lines += ["", f"_{len(rows)} rows · "
                  f"{sum(r.duration for r in rows) / 60:.1f} min of speech to review_", ""]
    return "\n".join(lines)


def _parse_role(raw: str) -> Role | None:
    cleaned = (raw or "").strip().lower()
    for role in Role:
        if cleaned == role.value:
            return role
    return None


def parse_sheet(markdown: str) -> list[LabelRow]:
    """Read a sheet back, filled in or not. Unparseable truth values stay None."""
    rows: list[LabelRow] = []
    for line in markdown.splitlines():
        match = _ROW.match(line.strip())
        if not match:
            continue
        row_id, start, end, predicted, confidence, truth = (g.strip() for g in match.groups())
        text = line.strip().split("|")[7].strip() if line.count("|") >= 8 else ""
        if row_id.lower() in {"row", "---"} or not row_id:
            continue

        pred = _parse_role(predicted)
        if pred is None:
            continue
        try:
            rows.append(LabelRow(
                row_id=row_id,
                start=float(start),
                end=float(end),
                predicted=pred,
                confidence=float(confidence),
                truth=_parse_role(truth),
                text=text,
            ))
        except ValueError:
            continue
    return rows


@dataclass
class LabelScore:
    total: int
    labelled: int
    correct: int
    accuracy: float | None
    confident_accuracy: float | None
    recall: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "labelled": self.labelled,
            "correct": self.correct,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "confident_accuracy": (None if self.confident_accuracy is None
                                   else round(self.confident_accuracy, 4)),
            "recall": {k.value: round(v, 4) for k, v in self.recall.items()},
            "confusion": {f"{a.value}->{b.value}": n for (a, b), n in self.confusion.items()},
        }


def score_labels(rows: list[LabelRow], confidence_floor: float = 0.15) -> LabelScore:
    """Accuracy over labelled rows only, plus accuracy restricted to confident rows.

    The second number is the one that matters for the design: if the classifier is right
    whenever it is sure, then gating on confidence is a working strategy even when overall
    accuracy is mediocre.
    """
    labelled = [r for r in rows if r.truth is not None]
    correct = sum(1 for r in labelled if r.predicted is r.truth)

    recall: dict[Role, float] = {}
    for role in Role:
        actual = [r for r in labelled if r.truth is role]
        if actual:
            recall[role] = sum(1 for r in actual if r.predicted is role) / len(actual)

    confusion: dict[tuple[Role, Role], int] = {}
    for r in labelled:
        key = (r.truth, r.predicted)
        confusion[key] = confusion.get(key, 0) + 1

    confident = [r for r in labelled if r.confidence >= confidence_floor]

    return LabelScore(
        total=len(rows),
        labelled=len(labelled),
        correct=correct,
        accuracy=(correct / len(labelled)) if labelled else None,
        confident_accuracy=(
            sum(1 for r in confident if r.predicted is r.truth) / len(confident)
            if confident else None),
        recall=recall,
        confusion=confusion,
    )


def with_truth(row: LabelRow, truth: Role | None) -> LabelRow:
    return replace(row, truth=truth)
