"""Stage 07 — one session in, one committable JSON out.

`results/<session_id>.json` is the boundary between the offline batch pass and the
dashboard. Everything upstream is free to change as long as this shape holds, which is
what lets the ASR backend be swapped in one line without touching the app.

It is also the only artefact from this project that may be committed. The recordings
themselves are of identifiable children, so nothing here may carry a teacher's name or a
path off their phone — `tests/test_pipeline.py` fails if either leaks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import config
from src.asr.base import Transcript
from src.attribution import Role
from src.insight import build_insight, whatsapp_message
from src.metrics import compute_metrics, text_for_span
from src.models import Session
from src.signal_layer import Segment, SegmentKind

SCHEMA_VERSION = "1.0"


# Diarization hearing less of a second speaker than this is noise, not a child.
MIN_OTHER_SPEAKER_SEC = 2.0


def _lost_students(segments: list[Segment], speaker_turns) -> bool:
    """Did the split drop students that diarization clearly heard?

    Two views of the same recording have to agree about whether anyone but the teacher
    spoke. They did not: M1 reported 100% teacher talk on two sessions while M5, reading
    the same turns, found 28 and 11 answered student questions. The hand labels could
    not catch it - all 36 of them are "teacher", so never saying "student" scores 100%.
    """
    if not speaker_turns:
        return False

    from src.diarization import teacher_speaker

    teacher, _ = teacher_speaker(speaker_turns)
    if teacher is None:
        return False

    other_sec = sum(t.end - t.start for t in speaker_turns if t.speaker != teacher)
    if other_sec < MIN_OTHER_SPEAKER_SEC:
        return False

    student_sec = sum(s.duration for s in segments if s.role is Role.STUDENT)
    return student_sec <= 0.0


def build_result(session: Session, segments: list[Segment], audio_sec: float,
                 transcript: Transcript | None = None,
                 attribution: str = "energy-kmeans-v1",
                 speaker_turns=None,
                 vad: str | None = None) -> dict[str, Any]:
    """Assemble everything known about one session into the published shape."""
    metrics = compute_metrics(
        segments=segments,
        roster=session.meta.roster,
        audio_sec=audio_sec,
        transcript=transcript,
        completeness=session.completeness,
        speaker_turns=speaker_turns,
    )

    timeline: list[dict[str, Any]] = []
    for seg in segments:
        entry = seg.to_dict()
        if transcript is not None and seg.kind is SegmentKind.SPEECH:
            text = text_for_span(transcript, seg.start, seg.end)
            if text:
                entry["text"] = text
        timeline.append(entry)

    payload = metrics.to_dict()

    flags = list(session.flags)
    if _lost_students(segments, speaker_turns):
        flags.append("attribution_lost_students")

    insight = build_insight(metrics, activity=session.meta.activity, alias=session.alias)
    insight_payload = insight.to_dict()
    insight_payload["whatsapp"] = whatsapp_message(insight)

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session.meta.session_id,
        # alias, never the real name — this file is the one that gets committed
        "teacher": {"id": session.meta.teacher_id, "alias": session.alias},
        "activity": session.meta.activity,
        "recorded_at": (session.meta.recorded_at.isoformat()
                        if session.meta.recorded_at else None),
        "roster": {
            "total": session.meta.roster.total,
            "boys": session.meta.roster.boys,
            "girls": session.meta.roster.girls,
        },
        "audio": {
            "declared_sec": (round(session.meta.declared_sec, 1)
                             if session.meta.declared_sec else None),
            "actual_sec": round(session.duration_sec, 1),
            # How much was actually processed. A partial run analyses a window, and the
            # metrics divide by *this*, not by the whole file.
            "analysed_sec": round(audio_sec, 1),
            "completeness": (round(session.completeness, 3)
                             if session.completeness is not None else None),
            "sample_rate": session.probe.sample_rate,
            "channels": session.probe.channels,
            "has_gps": session.meta.photo.has_gps,
        },
        "models": {
            "asr": transcript.model if transcript else None,
            "asr_backend": transcript.backend if transcript else None,
            "attribution": attribution,
            "vad": vad or f"silero@{config.VAD_THRESHOLD}",
        },
        "timeline": timeline,
        "totals_sec": payload["totals_sec"],
        "metrics": payload["metrics"],
        "confidence": payload["confidence"],
        "insight": insight_payload,
        "flags": flags,
    }


def write_index(out_dir: Path) -> Path:
    """List every result file so the dashboard can find them without a directory listing.

    A static site cannot enumerate a folder, so this index is what makes the app work as
    plain files on any host - no backend, no cold start.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in out_dir.glob("*.json") if p.stem != "index")
    index = out_dir / "index.json"
    index.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    return index


def write_result(result: dict[str, Any], out_dir: Path) -> Path:
    """Write one result file. `ensure_ascii=False` so the Devanagari stays readable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['session_id']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
