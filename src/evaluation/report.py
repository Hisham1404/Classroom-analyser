"""Turning a bake-off into markdown.

Two documents come out of a run: a summary a reviewer can read, and a scoring sheet
Mohammed fills in after listening. The summary must never let its proxy numbers be
mistaken for a word error rate — that overclaim is exactly what a technical reviewer
would catch.
"""

from __future__ import annotations

from src.evaluation.bakeoff import BakeoffResult, rank_contenders
from src.evaluation.signals import _COMPRESSION_ALARM

_DISCLAIMER = (
    "> **These are proxy signals, not ground truth.** There is no human reference "
    "transcript for this audio, so nothing here is a word error rate. The proxies "
    "reliably catch *garbage* — looping, script drift, confident text over silence — "
    "which is enough to shortlist models before hand-scoring. The verdict column in "
    "`asr_handscoring.md` is the one that decides."
)


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "_(no speech detected)_"
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_report(result: BakeoffResult) -> str:
    lines: list[str] = [
        "# ASR bake-off",
        "",
        "Same clips, same preprocessing, every contender.",
        "",
        _DISCLAIMER,
        "",
    ]

    ranked = rank_contenders(result)
    if ranked:
        lines += [
            "## Ranking",
            "",
            "| # | Contender | Backend | Model | Clips | Composite | Agreement | Mean logprob | Looping | Devanagari | Sec/clip |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for position, s in enumerate(ranked, start=1):
            lines.append(
                f"| {position} | **{s.label}** | {s.backend_name} | `{s.model_id}` | "
                f"{s.clips_scored} | {s.composite:.3f} | {_fmt(s.agreement, '.3f')} | {_fmt(s.mean_logprob)} | "
                f"{s.repetition_rate:.2f} | {s.devanagari_ratio:.2f} | {s.mean_elapsed_sec:.1f} |"
            )
        lines += [
            "",
            "**Agreement** is mean word overlap with the other contenders on the same clip. "
            "A contender that agrees with nobody is inventing - that is the only signal that "
            "catches fluent-looking nonsense. "
            "**Composite** weights model confidence (0.40), speech coverage (0.20), "
            "absence of looping (0.25) and staying in Devanagari (0.15).",
            f"**Looping** is the share of repeated 4-grams. **Compression ratio above "
            f"{_COMPRESSION_ALARM}** is Whisper's own hallucination alarm.",
            "",
        ]
    else:
        lines += ["## Ranking", "", "_No contender produced a transcript._", ""]

    if result.rows:
        lines += ["## Transcripts", ""]
        by_clip: dict[str, list] = {}
        for row in result.rows:
            by_clip.setdefault(row.clip.clip_id, []).append(row)

        for clip_id, rows in by_clip.items():
            spec = rows[0].clip
            lines += [
                f"### `{clip_id}`",
                f"_{spec.start / 60:.1f}–{spec.end / 60:.1f} min_",
                "",
                "| Contender | Lang | Composite | Logprob | Compression | Text |",
                "|---|---|---:|---:|---:|---|",
            ]
            for row in sorted(rows, key=lambda r: -r.signals.composite):
                sig = row.signals
                alarm = " ⚠️" if sig.compression_ratio > _COMPRESSION_ALARM else ""
                lines.append(
                    f"| {row.label} | {row.transcript.language} | {sig.composite:.3f} | "
                    f"{_fmt(sig.mean_logprob)} | {sig.compression_ratio:.2f}{alarm} | "
                    f"{_truncate(row.transcript.text)} |"
                )
            lines.append("")

    if result.failures:
        lines += ["## Failures", ""]
        lines += [f"- {f}" for f in result.failures]
        lines.append("")

    return "\n".join(lines)


def hand_scoring_sheet(result: BakeoffResult) -> str:
    """A blank sheet to fill in while listening. This is the real evaluation."""
    lines: list[str] = [
        "# ASR hand-scoring",
        "",
        "Listen to each clip, read the text, fill in the last three columns.",
        "",
        "- **Usable?** — y / n / partial: could a teacher recognise their own lesson in this?",
        "- **Words right** — rough fraction of words that are actually correct.",
        "- **Notes** — what it got wrong: names, numbers, code-mixed English, invented text.",
        "",
        "| Clip | Contender | Transcript | Usable? | Words right | Notes |",
        "|---|---|---|:-:|:-:|---|",
    ]

    for row in sorted(result.rows, key=lambda r: (r.clip.clip_id, r.label)):
        lines.append(
            f"| `{row.clip.clip_id}` | {row.label} | {_truncate(row.transcript.text, 120)} "
            f"| _ | _ | |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        "_Winner:_ ",
        "_Why:_ ",
        "_Good enough to build metrics on?_ ",
        "",
    ]
    return "\n".join(lines)
