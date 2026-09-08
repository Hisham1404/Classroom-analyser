"""Config must resolve paths correctly and keep model weights off the nearly-full D: drive."""

from __future__ import annotations

from pathlib import Path

from src import config


def test_project_root_is_the_repo_root():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src").is_dir()


def test_data_dirs_sit_under_project_root():
    for p in (config.DATA_DIR, config.RESULTS_DIR, config.EVAL_DIR):
        assert p.is_relative_to(config.PROJECT_ROOT)


def test_hf_cache_is_not_on_the_project_drive():
    """D: has ~3.7 GB free and the model set is ~5 GB — weights must live elsewhere."""
    cache = config.hf_cache_dir()
    assert cache.is_absolute()
    if config.PROJECT_ROOT.drive:
        assert cache.drive.upper() != config.PROJECT_ROOT.drive.upper()


def test_configure_hf_cache_sets_the_env_var(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    config.configure_hf_cache()
    import os
    assert Path(os.environ["HF_HOME"]) == config.hf_cache_dir()


def test_configure_hf_cache_respects_an_existing_override(monkeypatch):
    monkeypatch.setenv("HF_HOME", r"X:\somewhere\else")
    config.configure_hf_cache()
    import os
    assert os.environ["HF_HOME"] == r"X:\somewhere\else"


def test_confidence_bands_are_ordered_and_in_range():
    assert 0.0 < config.CONF_UNRELIABLE < config.CONF_USABLE < 1.0


def test_truncation_threshold_is_a_sensible_fraction():
    assert 0.9 <= config.COMPLETENESS_OK < 1.0


def test_asr_model_ids_are_declared():
    assert "small" in config.ASR_MODELS["baseline"]
    assert "whisper-hindi" in config.ASR_MODELS["primary"]
    assert config.ASR_LANGUAGE == "hi"


def test_local_models_are_ctranslate2_builds():
    """faster-whisper cannot load a raw transformers checkpoint.

    Pointing it at `openai/whisper-small` fails at load time with
    "Unable to open file 'model.bin'" - a real bug this guard now prevents.
    """
    for key, model in config.ASR_MODELS.items():
        lowered = model.lower()
        assert "faster-whisper" in lowered or "ct2" in lowered, (key, model)


def test_models_needing_another_runtime_are_kept_out_of_the_bakeoff():
    assert "indic-conformer" in config.ASR_MODELS_OTHER_RUNTIME
    assert not set(config.ASR_MODELS) & set(config.ASR_MODELS_OTHER_RUNTIME)
