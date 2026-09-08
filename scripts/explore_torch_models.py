"""Try the ASR models that need torch, on the same clips as the bake-off.

Exploration, not a graded contender list — anything that wins here gets wired in properly
as a backend afterwards. Scores use the same `quality_signals` as the bake-off so the
numbers are directly comparable.

    python scripts/explore_torch_models.py --only wav2vec2
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.base import Transcript, TranscriptSegment          # noqa: E402
from src.evaluation.signals import quality_signals               # noqa: E402

# (label, model_id, kind, note)
CANDIDATES = [
    ("indicwav2vec",  "ai4bharat/indicwav2vec-hindi",                  "ctc",     "AI4Bharat wav2vec2"),
    ("vakyansh",      "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200", "ctc",     "Vakyansh 4200h"),
    ("xlsr-hindi",    "theainerd/Wav2Vec2-large-xlsr-hindi",           "ctc",     "most-downloaded Hindi ASR"),
    ("conformer-600m", "ai4bharat/indic-conformer-600m-multilingual",  "conformer", "AI4Bharat 22-lang"),
    ("hinglish-swift", "Oriserve/Whisper-Hindi2Hinglish-Swift",       "whisper", "0.29 GB, Hinglish out"),
    ("hindi2hinglish", "Oriserve/Whisper-Hindi2Hinglish-Prime",        "whisper", "6.2 GB, 248K downloads"),
    ("vaani-large-v3", "ARTPARK-IISc/whisper-large-v3-vaani-hindi",    "whisper", "IISc, rural Vaani corpus"),
]


def load_audio(path: str, sr: int = 16000):
    import numpy as np
    import subprocess
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def run_ctc(model_id: str, clips: list[str]):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    # Some repos ship a ProcessorWithLM that needs kenlm + pyctcdecode. We only want the
    # acoustic model here, so ask for the plain processor and fall back if unavailable.
    try:
        from transformers import Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained(model_id)
    except Exception:
        proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).eval()

    for clip in clips:
        audio = load_audio(clip)
        t0 = time.perf_counter()
        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        decoder = getattr(proc, "tokenizer", proc)
        # skip_special_tokens matters: without it the CTC blank leaks into the output as
        # literal "<s>" between every character, and the quality signals happily score it.
        text = decoder.batch_decode(ids, skip_special_tokens=True)[0]
        yield clip, text, time.perf_counter() - t0


def run_whisper(model_id: str, clips: list[str]):
    import torch
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=model_id,
                    device="cpu", torch_dtype=torch.float32)
    for clip in clips:
        t0 = time.perf_counter()
        out = pipe(clip, generate_kwargs={"language": "hi", "task": "transcribe"},
                   chunk_length_s=30)
        yield clip, out.get("text", ""), time.perf_counter() - t0


def run_conformer(model_id: str, clips: list[str]):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
    for clip in clips:
        audio = load_audio(clip)
        t0 = time.perf_counter()
        with torch.no_grad():
            text = model(torch.tensor(audio).unsqueeze(0), "hi", "ctc")
        if isinstance(text, (list, tuple)):
            text = text[0]
        yield clip, str(text), time.perf_counter() - t0


RUNNERS = {"ctc": run_ctc, "whisper": run_whisper, "conformer": run_conformer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="labels to run")
    ap.add_argument("--clips", type=int, default=3)
    args = ap.parse_args()

    clips = sorted(glob.glob("eval/clips/*.wav"))[:args.clips]
    if not clips:
        print("no clips in eval/clips/ — run the bake-off first", file=sys.stderr)
        return 1

    print(f"{len(clips)} clips\n" + "=" * 96)
    summary = []

    for label, model_id, kind, note in CANDIDATES:
        if args.only and label not in args.only:
            continue
        print(f"\n### {label}  ({model_id})\n    {note}", flush=True)

        try:
            t0 = time.perf_counter()
            results = list(RUNNERS[kind](model_id, clips))
            load_and_run = time.perf_counter() - t0
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            summary.append((label, None, None, f"{type(exc).__name__}"))
            continue

        comps, secs = [], []
        for clip, text, elapsed in results:
            t = Transcript(
                segments=[TranscriptSegment(0.0, 30.0, text, None, None, None)] if text.strip() else [],
                language="hi", language_prob=None, backend=label, model=model_id, audio_sec=30.0,
            )
            sig = quality_signals(t)
            comps.append(sig.composite)
            secs.append(elapsed)
            print(f"    [{elapsed:6.1f}s] comp {sig.composite:.3f} deva {sig.devanagari_ratio:.2f} "
                  f"rep {sig.repetition_rate:.2f} {len(text):>4}ch", flush=True)
            print(f"        {(text or '(empty)')[:150]}", flush=True)

        avg = sum(comps) / len(comps) if comps else 0.0
        sec = sum(secs) / len(secs) if secs else 0.0
        summary.append((label, avg, sec, f"total {load_and_run:.0f}s incl. load"))

    print("\n" + "=" * 96 + "\nSUMMARY (composite, sec/30s clip)")
    for label, comp, sec, note in sorted(summary, key=lambda r: -(r[1] or -1)):
        if comp is None:
            print(f"  {label:<16} FAILED  {note}")
        else:
            print(f"  {label:<16} {comp:.3f}   {sec:6.1f}s/clip   ({30/sec if sec else 0:.2f}x realtime)  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
