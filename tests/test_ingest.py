"""Ingest: find sessions, parse their metadata, measure the audio, and catch the truncation bug.

The rule this module exists to enforce: the session JSON's `duration` is untrusted.
Only the decoded audio length may be used as a denominator anywhere downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest import (
    build_corpus,
    build_session,
    discover_session_dirs,
    load_session_meta,
    probe_audio,
)


# ---------------------------------------------------------------- discovery
def test_discovers_every_complete_session(corpus: Path):
    found = discover_session_dirs(corpus)
    assert len(found) == 3


def test_discovery_is_sorted_for_reproducible_runs(corpus: Path):
    names = [d.name for d in discover_session_dirs(corpus)]
    assert names == sorted(names)


def test_a_folder_without_audio_is_skipped(corpus: Path):
    names = [d.name for d in discover_session_dirs(corpus)]
    assert "OD90003_2026-02-01-101010" not in names


def test_loose_files_are_ignored(corpus: Path):
    assert all(d.is_dir() for d in discover_session_dirs(corpus))


def test_missing_root_raises_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_session_dirs(tmp_path / "nope")


# ---------------------------------------------------------------- metadata
def test_loads_metadata_from_a_session_dir(intact_session: Path):
    m = load_session_meta(intact_session)
    assert m.session_id == "OD90001_2026-01-06-114155"
    assert m.roster.total == 8
    assert m.photo.has_gps


def test_loads_a_session_that_has_no_gps(truncated_session: Path):
    m = load_session_meta(truncated_session)
    assert not m.photo.has_gps
    assert m.activity == "Shadow Art"


def test_unreadable_metadata_raises_with_the_path_named(tmp_path: Path):
    d = tmp_path / "BAD_2026-01-01-000000"
    d.mkdir()
    (d / "BAD_2026-01-01-000000.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_session_meta(d)
    assert "BAD_2026-01-01-000000" in str(e.value)


# ---------------------------------------------------------------- audio probe
def test_probe_measures_the_real_duration(intact_session: Path):
    p = probe_audio(intact_session / "OD90001_2026-01-06-114155.mp3")
    assert p.actual_sec == pytest.approx(6.0, abs=0.25)
    assert p.channels == 1
    assert p.sample_rate == 44100
    assert p.size_bytes > 0


def test_probe_on_a_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        probe_audio(tmp_path / "ghost.mp3")


# ---------------------------------------------------------------- the truncation rule
def test_intact_session_is_not_flagged(intact_session: Path):
    s = build_session(intact_session)
    assert s.completeness == pytest.approx(1.0, abs=0.05)
    assert not s.is_truncated
    assert "audio_truncated" not in s.flags


def test_truncated_session_is_caught(truncated_session: Path):
    """Declared 12 s, 4 s present — the shape of the real 64-min/22-min file."""
    s = build_session(truncated_session)
    assert s.completeness == pytest.approx(4 / 12, abs=0.05)
    assert s.is_truncated
    assert "audio_truncated" in s.flags


def test_duration_used_downstream_is_the_measured_one(truncated_session: Path):
    s = build_session(truncated_session)
    assert s.duration_sec == s.probe.actual_sec
    assert s.duration_sec != s.meta.declared_sec


def test_completeness_is_none_when_they_declared_nothing(tmp_path: Path, intact_session: Path):
    j = intact_session / "OD90001_2026-01-06-114155.json"
    data = json.loads(j.read_text(encoding="utf-8"))
    del data["duration"]
    j.write_text(json.dumps(data), encoding="utf-8")

    s = build_session(intact_session)
    assert s.completeness is None
    assert not s.is_truncated
    assert s.duration_sec == pytest.approx(6.0, abs=0.25)


def test_roster_mismatch_raises_a_flag(tmp_path: Path, intact_session: Path):
    j = intact_session / "OD90001_2026-01-06-114155.json"
    data = json.loads(j.read_text(encoding="utf-8"))
    data["totalStudents"] = 99
    j.write_text(json.dumps(data), encoding="utf-8")

    assert "roster_mismatch" in build_session(intact_session).flags


def test_missing_gps_raises_a_flag(truncated_session: Path):
    assert "no_gps" in build_session(truncated_session).flags


# ---------------------------------------------------------------- privacy
def test_public_dict_never_carries_the_teacher_name(corpus: Path):
    for s in build_corpus(corpus):
        blob = json.dumps(s.to_public_dict())
        assert "Testteacher" not in blob
        assert "teacher_name" not in blob


def test_public_dict_uses_an_alias_instead(corpus: Path):
    s = build_corpus(corpus)[0]
    assert s.to_public_dict()["teacher"]["alias"].startswith("Teacher ")


def test_public_dict_drops_device_file_paths(corpus: Path):
    for s in build_corpus(corpus):
        assert "/storage/emulated" not in json.dumps(s.to_public_dict())


def test_public_dict_is_json_serialisable(corpus: Path):
    for s in build_corpus(corpus):
        json.loads(json.dumps(s.to_public_dict()))


# ---------------------------------------------------------------- corpus
def test_corpus_assigns_one_alias_per_teacher(corpus: Path):
    sessions = build_corpus(corpus)
    by_teacher = {s.meta.teacher_id: s.alias for s in sessions}
    assert by_teacher == {"OD90001": "Teacher A", "OD90002": "Teacher B"}


def test_corpus_finds_the_truncated_member(corpus: Path):
    truncated = [s for s in build_corpus(corpus) if s.is_truncated]
    assert [s.meta.session_id for s in truncated] == ["OD90001_2026-01-28-121933"]


def test_corpus_total_audio_uses_measured_seconds(corpus: Path):
    total = sum(s.duration_sec for s in build_corpus(corpus))
    assert total == pytest.approx(13.0, abs=0.75)   # 5 + 3 + 5, not 5 + 9 + 5


# ---------------------------------------------------------------- real dataset
@pytest.mark.integration
def test_real_corpus_has_five_sessions(real_data_dir: Path):
    assert len(build_corpus(real_data_dir)) == 5


@pytest.mark.integration
def test_real_corpus_reproduces_the_measured_findings(real_data_dir: Path):
    sessions = build_corpus(real_data_dir)

    total_min = sum(s.duration_sec for s in sessions) / 60
    assert total_min == pytest.approx(251.7, abs=1.0)

    assert sum(s.is_truncated for s in sessions) == 3
    assert sum(1 for s in sessions if "no_gps" in s.flags) == 3
    assert len({s.meta.teacher_id for s in sessions}) == 3


@pytest.mark.integration
def test_real_corpus_worst_truncation_is_the_january_session(real_data_dir: Path):
    worst = min((s for s in build_corpus(real_data_dir) if s.completeness is not None),
                key=lambda s: s.completeness)
    assert worst.meta.session_id == "OD11163_2026-01-28-121933"
    assert worst.completeness == pytest.approx(0.344, abs=0.02)


@pytest.mark.integration
def test_no_real_teacher_name_escapes_into_public_output(real_data_dir: Path):
    names = {"Santana", "Aayushi", "Sudipto"}
    for s in build_corpus(real_data_dir):
        blob = json.dumps(s.to_public_dict())
        assert not (names & set(blob.split('"')))
