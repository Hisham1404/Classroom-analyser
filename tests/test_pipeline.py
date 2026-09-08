"""The orchestration — one session in, one `results/<id>.json` out.

That file is the contract between the offline batch pass and the dashboard. Everything
upstream can be rewritten freely as long as its shape holds, which is what makes the ASR
backend swappable without touching the app.

The rule it must never break: **no teacher name, no device paths.** The raw recordings are
of identifiable children and can never be committed; this file is what gets committed
instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.asr.base import Transcript, TranscriptSegment
from src import config
from src.attribution import Role
from src.pipeline import build_result, write_result
from src.signal_layer import Segment, SegmentKind


def speech(start, end, role=Role.TEACHER, conf=0.7, rms=-18.0):
    return Segment(start, end, SegmentKind.SPEECH, rms, 0.2, role, conf)


def handson(start, end):
    return Segment(start, end, SegmentKind.HANDSON, -30.0, 0.6)


@pytest.fixture
def session(corpus: Path):
    from src.ingest import build_corpus
    return build_corpus(corpus)[0]


@pytest.fixture
def segments():
    return [speech(0, 60, Role.TEACHER), speech(60, 90, Role.STUDENT, rms=-35.0),
            handson(90, 200), speech(200, 240, Role.TEACHER)]


@pytest.fixture
def transcript():
    return Transcript(
        segments=[TranscriptSegment(0, 60, "यह क्या है", -0.4, 0.1, None),
                  TranscriptSegment(60, 90, "कागज़ से", -0.5, 0.1, None)],
        language="hi", language_prob=0.95, backend="indic", model="m", audio_sec=240.0,
    )


# ================================================================ shape
def test_result_carries_the_session_identity(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["session_id"] == session.meta.session_id
    assert r["teacher"]["alias"].startswith("Teacher ")


def test_result_carries_the_roster(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["roster"]["total"] == session.meta.roster.total


def test_result_carries_audio_facts_including_completeness(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert "actual_sec" in r["audio"]
    assert "completeness" in r["audio"]


def test_result_carries_every_metric(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert set(r["metrics"]) == {
        "M1_teacher_talk_ratio", "M2_student_participation", "M3_interaction_density",
        "M4_longest_teacher_stretch", "M5_wait_time_1",
    }


def test_each_metric_keeps_its_formula_for_the_readme(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    for m in r["metrics"].values():
        assert m["formula"] and m["explanation"] and m["interpretation"]


def test_result_carries_the_confidence_verdict(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["confidence"]["verdict"] in {"usable", "partial", "unreliable"}


def test_result_carries_the_timeline(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert len(r["timeline"]) == len(segments)
    assert r["timeline"][0]["kind"] == "speech"


def test_timeline_carries_transcript_text_when_available(session, segments, transcript):
    r = build_result(session, segments, audio_sec=240.0, transcript=transcript)
    assert any(seg.get("text") for seg in r["timeline"])


def test_timeline_has_no_text_without_a_transcript(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert all("text" not in seg for seg in r["timeline"])


def test_result_records_which_models_ran(session, segments, transcript):
    r = build_result(session, segments, audio_sec=240.0, transcript=transcript)
    assert r["models"]["asr"] == "m"
    assert r["models"]["attribution"]


def test_result_carries_ingest_flags(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert isinstance(r["flags"], list)


# ================================================================ privacy
def test_no_teacher_name_anywhere_in_the_result(session, segments, transcript):
    blob = json.dumps(build_result(session, segments, audio_sec=240.0,
                                   transcript=transcript), ensure_ascii=False)
    assert "Testteacher" not in blob
    assert "teacher_name" not in blob


def test_no_device_paths_in_the_result(session, segments):
    blob = json.dumps(build_result(session, segments, audio_sec=240.0))
    assert "/storage/emulated" not in blob


def test_result_is_json_serialisable(session, segments, transcript):
    json.loads(json.dumps(build_result(session, segments, audio_sec=240.0,
                                       transcript=transcript), ensure_ascii=False))


# ================================================================ writing
def test_write_creates_one_file_named_for_the_session(session, segments, tmp_path: Path):
    out = write_result(build_result(session, segments, audio_sec=240.0), tmp_path)
    assert out.name == f"{session.meta.session_id}.json"
    assert out.is_file()


def test_written_file_round_trips(session, segments, tmp_path: Path):
    result = build_result(session, segments, audio_sec=240.0)
    out = write_result(result, tmp_path)
    assert json.loads(out.read_text(encoding="utf-8"))["session_id"] == result["session_id"]


def test_written_file_keeps_devanagari_readable(session, segments, transcript, tmp_path: Path):
    """Escaped \\u sequences would make the committed results unreviewable by eye."""
    out = write_result(build_result(session, segments, audio_sec=240.0,
                                    transcript=transcript), tmp_path)
    assert "यह क्या है" in out.read_text(encoding="utf-8")


def test_write_creates_the_directory(session, segments, tmp_path: Path):
    out = write_result(build_result(session, segments, audio_sec=240.0),
                       tmp_path / "deep" / "nested")
    assert out.is_file()


# ================================================================ schema stability
def test_schema_version_is_declared(session, segments):
    """The dashboard reads these files; a silent shape change would break it."""
    assert build_result(session, segments, audio_sec=240.0)["schema_version"]


def test_top_level_keys_are_stable(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert set(r) == {
        "schema_version", "session_id", "teacher", "activity", "recorded_at", "roster",
        "audio", "models", "timeline", "totals_sec", "metrics", "confidence", "insight",
        "flags",
    }


# ================================================================ attribution method
def test_result_records_which_attribution_method_ran(session, segments):
    r = build_result(session, segments, audio_sec=240.0, attribution="pyannote-community-1")
    assert r["models"]["attribution"] == "pyannote-community-1"


def test_attribution_method_defaults_to_the_energy_heuristic(session, segments):
    """Recording which one ran matters: they measured 88.9% and 41.7%, and a results file
    that does not say which produced it cannot be interpreted."""
    r = build_result(session, segments, audio_sec=240.0)
    assert r["models"]["attribution"] == "energy-kmeans-v1"


# ================================================================ insight
def test_result_carries_the_insight(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["insight"]["headline"]
    assert r["insight"]["suggestion"]


def test_the_insight_carries_the_whatsapp_message(session, segments):
    """Her real channel is the bot, not the browser — the message ships in the file."""
    r = build_result(session, segments, audio_sec=240.0)
    assert r["insight"]["whatsapp"]
    assert len(r["insight"]["whatsapp"]) <= 400


def test_the_whatsapp_message_uses_the_alias(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["teacher"]["alias"] in r["insight"]["whatsapp"]


def test_the_insight_says_it_came_from_rules(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["insight"]["generated_by"].startswith("rules")


def test_no_teacher_name_reaches_the_whatsapp_message(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert "Testteacher" not in json.dumps(r["insight"], ensure_ascii=False)


def test_insight_is_in_the_stable_key_set(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert set(r) == {
        "schema_version", "session_id", "teacher", "activity", "recorded_at", "roster",
        "audio", "models", "timeline", "totals_sec", "metrics", "confidence", "insight",
        "flags",
    }


# ================================================================ index
def test_index_lists_every_written_result(session, segments, tmp_path: Path):
    from src.pipeline import write_index

    write_result(build_result(session, segments, audio_sec=240.0), tmp_path)
    index = write_index(tmp_path)

    assert index.name == "index.json"
    assert json.loads(index.read_text(encoding="utf-8")) == [session.meta.session_id]


def test_index_is_sorted_for_a_stable_ui_order(tmp_path: Path):
    from src.pipeline import write_index

    for name in ("c", "a", "b"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
    assert json.loads(write_index(tmp_path).read_text(encoding="utf-8")) == ["a", "b", "c"]


def test_the_index_never_lists_itself(tmp_path: Path):
    from src.pipeline import write_index

    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    write_index(tmp_path)
    assert "index" not in json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))


def test_an_empty_results_directory_gives_an_empty_index(tmp_path: Path):
    from src.pipeline import write_index

    assert json.loads(write_index(tmp_path).read_text(encoding="utf-8")) == []


# ================================================================ analysed window
def test_result_says_how_much_audio_was_actually_analysed(session, segments):
    """A partial run must not present the whole file's duration as the analysed span —
    the metrics divide by the window, so a reader could not reproduce them."""
    r = build_result(session, segments, audio_sec=240.0)
    assert r["audio"]["analysed_sec"] == pytest.approx(240.0)


def test_analysed_and_actual_differ_on_a_partial_run(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert r["audio"]["actual_sec"] != r["audio"]["analysed_sec"]


def test_a_full_run_has_them_equal(session, segments):
    full = session.duration_sec
    r = build_result(session, segments, audio_sec=full)
    assert r["audio"]["analysed_sec"] == pytest.approx(r["audio"]["actual_sec"], abs=0.1)


def test_the_timeline_never_exceeds_the_analysed_span(session, segments):
    r = build_result(session, segments, audio_sec=240.0)
    assert max(s["t1"] for s in r["timeline"]) <= r["audio"]["analysed_sec"] + 0.1


def test_the_runner_regenerates_the_index(tmp_path: Path, monkeypatch):
    """A session written without refreshing the index is invisible to the dashboard —
    a static site cannot list a directory."""
    import src.run_pipeline as runner

    assert "write_index(out_dir)" in Path(runner.__file__).read_text(encoding="utf-8")


# ================================ cross-checking the two views of who spoke
#
# The bug that made this necessary: M1 reported 100% teacher talk and M3 zero exchanges
# on two sessions while M5, reading the same recording, found 28 and 11 answered student
# questions. Both numbers were printed on the same card and nothing noticed. The hand
# labels could not catch it either - all 36 of them are "teacher", so a classifier that
# never says "student" scores 100% on that set.

def test_a_timeline_with_no_students_contradicts_turns_that_have_them(session):
    from src.diarization import SpeakerTurn

    # Diarization heard a second speaker for 12 seconds...
    turns = [SpeakerTurn(0, 60, "T"), SpeakerTurn(10, 16, "S1"), SpeakerTurn(30, 36, "S1")]
    # ...but the timeline that reached the metrics has only teacher speech.
    segments = [speech(0, 60, Role.TEACHER)]

    result = build_result(session, segments, audio_sec=60.0, speaker_turns=turns,
                          attribution="pyannote-community-1")
    assert "attribution_lost_students" in result["flags"]


def test_no_contradiction_flag_when_the_students_survive(session):
    from src.diarization import SpeakerTurn

    turns = [SpeakerTurn(0, 60, "T"), SpeakerTurn(10, 16, "S1"), SpeakerTurn(30, 36, "S1")]
    segments = [speech(0, 10, Role.TEACHER), speech(10, 16, Role.STUDENT),
                speech(16, 60, Role.TEACHER)]

    result = build_result(session, segments, audio_sec=60.0, speaker_turns=turns,
                          attribution="pyannote-community-1")
    assert "attribution_lost_students" not in result["flags"]


def test_no_contradiction_flag_when_diarization_heard_one_speaker(session):
    from src.diarization import SpeakerTurn

    turns = [SpeakerTurn(0, 60, "T")]
    result = build_result(session, [speech(0, 60, Role.TEACHER)], audio_sec=60.0,
                          speaker_turns=turns, attribution="pyannote-community-1")
    assert "attribution_lost_students" not in result["flags"]


def test_a_trace_of_other_speech_is_not_treated_as_a_lost_student(session):
    """Half a second of someone else is diarization noise, not a missing child."""
    from src.diarization import SpeakerTurn

    turns = [SpeakerTurn(0, 60, "T"), SpeakerTurn(10, 10.4, "S1")]
    result = build_result(session, [speech(0, 60, Role.TEACHER)], audio_sec=60.0,
                          speaker_turns=turns, attribution="pyannote-community-1")
    assert "attribution_lost_students" not in result["flags"]


def test_the_dashboard_reads_the_pipeline_output_directly():
    """There must be exactly one copy of the results.

    A hand-copied duplicate under `app/public/results` went stale and the dashboard
    spent a day serving TTR 100% and M5 0.00 from before the fixes, with nothing to
    indicate the numbers were old. The build now maps `results/` in, so a stale
    dashboard is not a state the project can reach.
    """
    import json

    root = Path(__file__).resolve().parent.parent
    config_path = root / "app" / "angular.json"
    if not config_path.exists():                       # the app is optional tooling
        pytest.skip("angular.json not present")

    assets = (json.loads(config_path.read_text(encoding="utf-8"))
              ["projects"]["dashboard"]["architect"]["build"]["options"]["assets"])
    mapped = [a for a in assets
              if isinstance(a, dict) and a.get("input") == "../results"]
    assert mapped, "angular.json does not map the pipeline's results/ directory"
    assert mapped[0].get("output") == "results"
    assert not (root / "app" / "public" / "results").exists(), (
        "app/public/results is back — that duplicate is what went stale")


def test_the_result_names_the_speech_detector_that_actually_ran(session, segments):
    """`models.vad` said "silero@0.5" unconditionally. Once diarization can supply the
    speech map, a hardcoded label is a lie about how the numbers were produced — and
    provenance is the one thing a withheld-rather-than-guessed pipeline cannot fudge."""
    r = build_result(session, segments, audio_sec=240.0, vad=config.DIARIZATION_VAD_LABEL)
    assert r["models"]["vad"] == config.DIARIZATION_VAD_LABEL

    fallback = build_result(session, segments, audio_sec=240.0)
    assert "silero" in fallback["models"]["vad"]


def test_the_runner_feeds_diarization_spans_into_the_timeline():
    """Ordering guard. If `build_timeline` is called before `diarize`, the speech map
    silently reverts to Silero and the hands-on sessions lose their students again —
    with every published number still looking perfectly plausible."""
    import src.run_pipeline as runner

    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "build_timeline(wave, speech_spans=speech_spans(speaker_turns))" in source
    assert source.index("speaker_turns = diarize(wav)") < source.index(
        "build_timeline(wave, speech_spans="), "diarization must run before the timeline"
    assert "vad=vad_label" in source
