"""Typed structures for a session, plus the privacy rule the whole project hangs on."""

from __future__ import annotations

import pytest

from src.models import AliasRegistry, AudioProbe, PhotoMeta, Roster, SessionMeta


# ---------------------------------------------------------------- Roster
def test_roster_totals_agree():
    r = Roster(total=8, boys=5, girls=3)
    assert r.is_consistent


def test_roster_flags_a_headcount_mismatch():
    r = Roster(total=10, boys=5, girls=3)
    assert not r.is_consistent


def test_roster_rejects_negative_counts():
    with pytest.raises(ValueError):
        Roster(total=-1, boys=0, girls=0)


# ---------------------------------------------------------------- PhotoMeta
def test_photo_meta_reports_gps_present():
    p = PhotoMeta(teacher_name="Santana", latitude=19.69, longitude=73.57)
    assert p.has_gps


def test_photo_meta_without_gps():
    """3 of 5 real sessions have no coordinates — this must not be an error."""
    p = PhotoMeta(teacher_name="Aayushi")
    assert not p.has_gps
    assert p.latitude is None


# ---------------------------------------------------------------- SessionMeta
def test_session_meta_parses_their_json_shape():
    m = SessionMeta.from_dict("OD11165_2026-01-06-114155", {
        "activityType": "Trumpet",
        "boys": 5, "girls": 3, "totalStudents": 8,
        "duration": 4134562,
        "teacherId": "OD11165",
        "timestamp": "2026-01-06 12:51:13",
        "photoMetadata": {"teacherName": "Aayushi", "capturedAt": "2026-01-06 12:51:13"},
    })
    assert m.teacher_id == "OD11165"
    assert m.activity == "Trumpet"
    assert m.roster.total == 8
    assert m.declared_sec == pytest.approx(4134.562)
    assert m.photo.teacher_name == "Aayushi"
    assert not m.photo.has_gps


def test_session_meta_survives_missing_optional_fields():
    m = SessionMeta.from_dict("OD00000_2026-01-01-000000", {"teacherId": "OD00000"})
    assert m.activity is None
    assert m.declared_sec is None
    assert m.roster.total == 0


def test_declared_duration_is_converted_from_milliseconds():
    m = SessionMeta.from_dict("s", {"duration": 3838906})
    assert m.declared_sec == pytest.approx(3838.906)


# ---------------------------------------------------------------- AudioProbe
def test_audio_probe_holds_what_ffprobe_returns():
    a = AudioProbe(path=None, actual_sec=3858.0, sample_rate=44100,
                   channels=1, bit_rate=130000, size_bytes=62532374)
    assert a.actual_sec == 3858.0
    assert a.channels == 1


def test_audio_probe_rejects_a_zero_duration():
    with pytest.raises(ValueError):
        AudioProbe(path=None, actual_sec=0.0, sample_rate=16000,
                   channels=1, bit_rate=1, size_bytes=1)


# ---------------------------------------------------------------- AliasRegistry
def test_aliases_are_assigned_in_sorted_order():
    reg = AliasRegistry(["OD11166", "OD11163", "OD11165"])
    assert reg.alias("OD11163") == "Teacher A"
    assert reg.alias("OD11165") == "Teacher B"
    assert reg.alias("OD11166") == "Teacher C"


def test_aliases_are_stable_regardless_of_input_order():
    a = AliasRegistry(["OD3", "OD1", "OD2"])
    b = AliasRegistry(["OD1", "OD2", "OD3"])
    assert [a.alias(t) for t in ("OD1", "OD2", "OD3")] == \
           [b.alias(t) for t in ("OD1", "OD2", "OD3")]


def test_duplicate_teacher_ids_collapse_to_one_alias():
    reg = AliasRegistry(["OD1", "OD1", "OD2"])
    assert reg.alias("OD1") == "Teacher A"
    assert reg.alias("OD2") == "Teacher B"


def test_unknown_teacher_id_raises_rather_than_inventing_an_alias():
    reg = AliasRegistry(["OD1"])
    with pytest.raises(KeyError):
        reg.alias("OD999")


def test_registry_handles_more_teachers_than_letters():
    ids = [f"OD{i:03d}" for i in range(30)]
    reg = AliasRegistry(ids)
    seen = {reg.alias(t) for t in ids}
    assert len(seen) == 30
