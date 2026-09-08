"""Stage 06 — one sentence a teacher can act on.

Santana teaches 22 children in an Igatpuri government school. She has a phone, patchy
data, five minutes, and no interest in machine learning. MakerGhat already reaches her
through MakerDost on WhatsApp, so the dashboard is the M&E team's surface — hers is a
message, and the design goal is that **the message alone is enough**.

Rule-based rather than generated. Deterministic output is explainable, runs offline, and —
the reason that decides it — **structurally cannot invent a statistic**. On a project whose
whole architecture exists to stop a confident wrong number reaching a teacher, letting a
language model write the headline would undo the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src import config
from src.attribution import Role
from src.metrics import MetricSet, Verdict


@dataclass
class Insight:
    headline: str
    suggestion: str
    summary: str
    metrics_shown: list[str] = field(default_factory=list)
    verdict: str = "usable"
    alias: str | None = None
    generated_by: str = "rules-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "suggestion": self.suggestion,
            "summary": self.summary,
            "metrics_shown": list(self.metrics_shown),
            "verdict": self.verdict,
            "generated_by": self.generated_by,
        }


def _value(metrics: MetricSet, key: str) -> float | None:
    metric = metrics.metrics.get(key)
    return metric.value if metric else None


def _minutes(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


# --------------------------------------------------------------------------- headline
def _headline(metrics: MetricSet, activity: str | None) -> str:
    ttr = _value(metrics, "M1_teacher_talk_ratio")
    what = activity or "this session"

    if ttr is None:
        spoken = metrics.totals[Role.TEACHER] + metrics.totals[Role.STUDENT]
        return (f"{what}: {_minutes(spoken)} of talking, but the recording could not tell "
                f"your voice from the children's.")

    if ttr > config.TEACHER_TALK_HIGH:
        return (f"You spoke {ttr:.0%} of the talking time in {what} — a hands-on session "
                f"usually works better below {config.TEACHER_TALK_BENCHMARK:.0%}.")
    if ttr > config.TEACHER_TALK_BENCHMARK:
        return f"You spoke {ttr:.0%} of the talking time in {what} — close to the target."
    return (f"You spoke {ttr:.0%} of the talking time in {what} — the children did most "
            f"of the talking.")


# --------------------------------------------------------------------------- suggestion
def _suggestion(metrics: MetricSet) -> str:
    """One thing to try. Picked from the weakest metric that was actually reported —
    never from one we withheld."""
    stretch = _value(metrics, "M4_longest_teacher_stretch")
    wait = _value(metrics, "M5_wait_time_1")
    ttr = _value(metrics, "M1_teacher_talk_ratio")
    participation = _value(metrics, "M2_student_participation")

    if stretch is not None and stretch > 300:
        return (f"Your longest unbroken stretch was {_minutes(stretch)}. Try breaking it "
                f"with a question at the halfway point.")

    if wait is not None and wait < config.WAIT_TIME_LOW_SEC:
        return (f"After you asked a question you moved on in {wait:.1f}s. Count to three "
                f"before taking an answer — research finds responses get longer and more "
                f"children join in.")

    if ttr is not None and ttr > config.TEACHER_TALK_HIGH:
        return ("Try handing one explanation to a group to give back to the class — it "
                "moves talking time to them without losing the content.")

    if participation is not None and participation < 1.0:
        return ("Only a few children spoke. Try asking a question to a named group rather "
                "than the whole room.")

    if stretch is not None and stretch > 120:
        return (f"Your longest unbroken stretch was {_minutes(stretch)} — about right, but "
                f"worth watching if it grows.")

    return ("Nothing stands out to change. Keep the phone where it is; the recording came "
            "through clearly.")


def _unreliable_suggestion(metrics: MetricSet) -> str:
    parts = metrics.confidence_parts
    if parts.get("speech_density", 1.0) < 0.4:
        return ("This recording was mostly room noise. Try keeping the phone on the front "
                "desk rather than in a pocket or bag.")
    if parts.get("completeness", 1.0) < 0.8:
        return ("The recording stopped early — only part of the lesson was saved. Check "
                "the app is still recording before you put the phone down.")
    return ("This recording was too noisy to read. Try keeping the phone closer to where "
            "you are standing.")


# --------------------------------------------------------------------------- summary
def _summary(metrics: MetricSet, activity: str | None) -> str:
    totals = metrics.totals
    bits: list[str] = []

    spoken = totals[Role.TEACHER] + totals[Role.STUDENT]
    if spoken > 0:
        bits.append(f"{_minutes(spoken)} of talking")
    if totals["handson"] > 0:
        bits.append(f"{_minutes(totals['handson'])} hands-on")
    if totals["dead"] > 30:
        bits.append(f"{_minutes(totals['dead'])} quiet")

    density = _value(metrics, "M3_interaction_density")
    if density is not None and density > 0:
        bits.append(f"{density:.1f} exchanges a minute")

    what = activity or "This session"
    return f"{what}: " + ", ".join(bits) + "." if bits else f"{what}: nothing measurable."


# --------------------------------------------------------------------------- main
def build_insight(metrics: MetricSet, activity: str | None = None,
                  alias: str | None = None) -> Insight:
    """Headline, one suggestion, and a plain summary — gated by the confidence verdict."""
    if metrics.verdict is Verdict.UNRELIABLE:
        what = activity or "This session"
        return Insight(
            headline=f"{what} was too noisy to analyse reliably.",
            suggestion=_unreliable_suggestion(metrics),
            summary=("No metrics are shown for this recording. A number here would be a "
                     "guess, and a guess is worse than nothing."),
            metrics_shown=[],
            verdict=metrics.verdict.value,
            alias=alias,
        )

    shown = [key for key, m in metrics.metrics.items() if m.value is not None]

    return Insight(
        headline=_headline(metrics, activity),
        suggestion=_suggestion(metrics),
        summary=_summary(metrics, activity),
        metrics_shown=shown,
        verdict=metrics.verdict.value,
        alias=alias,
    )


def whatsapp_message(insight: Insight) -> str:
    """The message that actually reaches the teacher. Everything else is optional."""
    greeting = f"नमस्ते {insight.alias}!" if insight.alias else "नमस्ते!"
    body = f"{greeting}\n\n{insight.headline}\n\n💡 {insight.suggestion}"

    if len(body) > config.WHATSAPP_MAX_CHARS:
        body = body[:config.WHATSAPP_MAX_CHARS - 1].rstrip() + "…"
    return body
