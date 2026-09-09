"""Project-wide paths, model choices and thresholds.

Model weights deliberately do not live on the project drive: D: has ~3.7 GB free and the
model set is ~5 GB, so the Hugging Face cache is pinned to the user's home drive.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "audio"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "eval"

# --- ASR -------------------------------------------------------------------
# THE ONE-LINE SWITCH. "local" runs open weights on this machine; "groq" calls the
# hosted API (needs GROQ_API_KEY). Both return the identical Transcript shape, so
# nothing downstream changes.
ASR_BACKEND = "indic"          # "local" | "groq" | "indic"

# faster-whisper runs CTranslate2, NOT raw transformers checkpoints. Pointing it at
# `openai/whisper-small` fails with "Unable to open file 'model.bin'". Every id here must
# therefore be a CT2 conversion - see test_config.test_local_models_are_ctranslate2_builds.
ASR_MODELS = {
    "baseline":    "Systran/faster-whisper-small",                      # ~0.5 GB
    "vanilla-med": "Systran/faster-whisper-medium",                     # ~1.5 GB
    "vanilla-lg":  "Systran/faster-whisper-large-v3",                   # ~3.1 GB
    "primary":     "digikar/vasista22-whisper-hindi-large-v2-ct2-int8",  # ~1.6 GB, Hindi fine-tuned
}

# Needs torch + transformers (~1 GB installed) and a different runtime entirely, so it is
# out of the faster-whisper bake-off. Revisit if the Whisper family proves unusable.
ASR_MODELS_OTHER_RUNTIME = {
    "indic-conformer": "ai4bharat/indic-conformer-600m-multilingual",
}
ASR_LANGUAGE = "hi"
ASR_BEAM_SIZE = 1
ASR_CPU_THREADS = 8            # measured: 16 threads is slower on this i7-1360P
ASR_CACHE_DIR = PROJECT_ROOT / ".cache" / "asr"
# Diarization is the single most expensive step (1.24x realtime, ~3 h for the corpus)
# and its turns are what every lexical metric is measured from, so they are cached too.
DIARIZATION_CACHE_DIR = PROJECT_ROOT / ".cache" / "diarization"

# --- IndicConformer (AI4Bharat via onnx-asr) -------------------------------
# ONNX, so it needs onnxruntime but NOT torch. Measured ~4x faster than realtime on this
# CPU vs 0.11x for the local Hindi Whisper. CTC/RNNT, so it does not hallucinate fluently.
INDIC_MODEL = "OpenVoiceOS/ai4bharat-indicconformer-hi-onnx"
# Split the token stream into segments wherever the speaker paused this long. onnx-asr's
# own VAD would do this too, but it returns SegmentResult, which carries no logprobs -
# and M6 weights ASR confidence at 0.20, so that signal is not worth throwing away.
INDIC_SEGMENT_GAP_SEC = 1.0
# The conformer's ONNX export has a fixed positional-encoding length. Feeding it more than
# ~100 s fails with "Attempting to broadcast an axis ... 2501 by 7501", so longer audio is
# chunked. 90 s leaves headroom; 2 s of overlap catches sentences straddling a cut.
INDIC_MAX_AUDIO_SEC = 90.0
INDIC_CHUNK_OVERLAP_SEC = 2.0

# --- hosted backend --------------------------------------------------------
GROQ_MODEL = "whisper-large-v3"
GROQ_MAX_UPLOAD_BYTES = 25 * 1024 * 1024      # free tier cap
GROQ_CHUNK_SEC = 600.0                        # 10 min -> ~10 MB as 16 kHz mono FLAC
GROQ_OVERLAP_SEC = 5.0                        # catches sentences straddling a cut

# --- thresholds ------------------------------------------------------------
# Below CONF_UNRELIABLE the dashboard refuses to report at all; above CONF_USABLE
# every metric is shown. In between, only the acoustic metrics are.
CONF_UNRELIABLE = 0.40
CONF_USABLE = 0.65

# A session whose audio is shorter than this fraction of its declared duration is
# flagged as truncated. Three of the five real recordings fail this.
COMPLETENESS_OK = 0.98

# --- signal layer ----------------------------------------------------------
# Silero VAD. 0.50 is faster-whisper's default and the value `scripts/tune_vad.py`
# validated against the real clips (58.5% mean coverage, matching the 20-86% per-session
# density measured in Phase 0). Below ~0.25 it starts counting classroom din as speech,
# which would inflate every talk-time metric.
VAD_THRESHOLD = 0.50
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 2000

# Non-speech louder than this is children working, not dead air. The maker-session
# distinction that stops a hands-on lesson scoring as a failed one.
HANDSON_RMS_DB = -35.0

# Below this margin between the two speaker clusters, the teacher/student split is a
# guess and must be reported as such.
ROLE_CONF_FLOOR = 0.15

# --- attribution -----------------------------------------------------------
# Measured against 36 transcript-register labels (docs/data-notes.md section 8):
#   pyannote diarization  88.9%
#   energy k-means        41.7%   (worse than chance)
# Diarization is the default; energy is only the fallback when no HF token is present,
# and results say which one ran.
ATTRIBUTION_METHOD = "diarization"      # "diarization" | "energy"
DIARIZATION_LABEL = "pyannote-community-1"
# When diarization runs, its own segmentation is the speech detector - it found
# speech Silero missed on 142 s of one session, which `classify_non_speech` was then
# filing as hands-on activity. `models.vad` records which of the two actually ran.
DIARIZATION_VAD_LABEL = "pyannote-community-1-segmentation"
ENERGY_LABEL = "energy-kmeans-v1"

# --- metrics ---------------------------------------------------------------
# Observed average in conventional classrooms is ~61% teacher talk. A hands-on maker
# session should sit well below that, so this is the target, not the norm.
TEACHER_TALK_BENCHMARK = 0.40
TEACHER_TALK_HIGH = 0.70          # above this the activity has become a lecture

# A pause this short is the same stretch of talking, not two separate ones.
STRETCH_MERGE_GAP_SEC = 3.0

# Mary Budd Rowe's wait time. Below 1s answers are short and students say "I don't know";
# at 3s or more responses get longer, more correct, and more students join in.
WAIT_TIME_TARGET_SEC = 3.0
WAIT_TIME_LOW_SEC = 1.0
# A student turn starting later than this is a new exchange, not an answer to the question.
WAIT_TIME_MAX_SEC = 15.0
# A median over one question-answer pair is not a median. Measured on the real sessions,
# four minutes of audio yields 3 to 8 detected questions, so this keeps the good sessions
# and withholds the thin ones instead of publishing a number resting on a single pause.
WAIT_TIME_MIN_SAMPLES = 3

# The message that actually reaches the teacher, via MakerDost on WhatsApp. Long enough
# for a headline and one suggestion, short enough to read on a phone without scrolling.
WHATSAPP_MAX_CHARS = 400

# M6 weights. Speech coverage and the speaker split are what the acoustic metrics rest on,
# so they carry the most; ASR confidence only matters for the one lexical metric.
CONF_WEIGHTS = {
    "speech_density": 0.30,
    "attribution": 0.30,
    "completeness": 0.20,
    "asr": 0.20,
}


def hf_cache_dir() -> Path:
    """Where Hugging Face model weights are cached."""
    override = os.environ.get("HF_HOME")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "huggingface"


def configure_hf_cache() -> Path:
    """Point Hugging Face at `hf_cache_dir()`, leaving an existing HF_HOME alone."""
    cache = hf_cache_dir()
    if not os.environ.get("HF_HOME"):
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
    return Path(os.environ["HF_HOME"])
