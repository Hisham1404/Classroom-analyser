# Task 1 — Classroom Voice Analytics: Build Plan

Written 2026-09-05, grounded in the actual dataset (`data-notes.md`), current model availability on
Hugging Face, and classroom-discourse research. Everything here costs **₹0**.

---

## 0. The one-paragraph version

Transcribe noisy Hindi classroom audio with an Indic-tuned Whisper, split it into teacher and student
speech acoustically, compute a small set of research-backed engagement metrics, and show a teacher —
not a data scientist — what happened in their lesson and one thing to try next time. Because the audio
is bad and the recordings are an hour long, **all heavy processing runs offline in a batch pipeline and
the live demo serves precomputed results**. That is also, conveniently, exactly the offline-first
architecture MakerGhat says it is building.

---

## 1. Start from the teacher, not the model

The assignment says "insights for teachers." So the design question is not *what can we compute* but
**what does Santana need on Monday morning?**

She teaches 22–25 children a maker activity in a government school in Igatpuri. She has a phone, patchy
data, ~5 minutes, and no interest in machine learning. MakerGhat reaches her through **MakerDost, a
WhatsApp bot** (`company-brief.md` §4) — so the web dashboard is the M&E team's view, and *her* view is
ultimately a WhatsApp message.

### What she actually needs, in four layers

| Layer | Time | Content |
|---|---|---|
| **1. The headline** | 3 sec | One sentence. *"You spoke 58% of the time — down from 71% last session."* |
| **2. The shape of the lesson** | 30 sec | A timeline strip (teacher / student / hands-on / quiet) and 4 metric cards, each with a plain-language reading and a benchmark bar |
| **3. The one thing to change** | 2 min | A single concrete suggestion tied to the weakest metric. *"You moved on 0.8s after asking a question. Count to three next time — research says answers get longer and more correct."* |
| **4. The evidence** | on demand | Clickable timeline → transcript segment → audio. Plus an honest confidence note. |

### What she does **not** need
WER, DER, diarization plots, model names, confusion matrices. None of it goes on her screen.

### The M&E / admin view is a different screen
Cross-teacher comparison, trend over time, data-quality flags, coverage. Same data, different framing.
**We have two sessions each for Santana and Sudipto — so "am I improving?" is demonstrable, not hypothetical.**
That trend view is the most convincing thing we can build.

### The domain insight that most candidates will miss

These are **maker sessions** — Trumpet, Shadow Art, Wind Anemometer — not lectures. In a hands-on class:

- Low teacher talk is **good**, not a coverage failure.
- Loud non-speech noise is usually **children building things**, i.e. engagement — not disruption.

A naive metric would score the Shadow Art session (20% speech, loudest file) as the *worst* class when it
may well have been the best one. So we split non-speech into **hands-on time** (loud, non-speech) and
**dead time** (quiet, non-speech). That single distinction is what makes this a *MakerGhat* analytics tool
rather than a generic one, and it comes straight from reading their curriculum pages.

---

## 2. The metrics

The assignment demands 2–3 metrics, each with **formula / explanation / interpretation**. Six are specified
below; four are acoustic and survive bad transcripts, one is lexical and flagged as fragile, one governs
the rest. Benchmarks come from classroom-discourse research (see Sources).

### M1 — Teacher Talk Ratio · *acoustic, robust*
```
TTR = teacher_speech_sec / (teacher_speech_sec + student_speech_sec)
```
Share of *spoken* time held by the teacher. Deliberately excludes non-speech so activity noise doesn't
dilute it.
**Interpretation.** Observed average in conventional classrooms is ~61% teacher / ~39% student. For a
hands-on maker session the target is lower — under ~40%. Above 70% means the activity has become a lecture.

### M2 — Student Participation Rate · *acoustic, robust*
```
SPR = (student_turns / totalStudents) / lesson_minutes × 60
```
Student turns per child per hour, normalised by the **roster count from the session JSON**.
**Interpretation.** Normalising matters: 8 students and 27 students cannot be compared on raw turn counts.
High student talk time *with* low SPR is the classic signal that two or three confident children are
dominating and the rest are silent.

### M3 — Interaction Density · *acoustic, robust*
```
ID = speaker_switches_between(teacher, student) / lesson_minutes
```
How often the floor changes hands. Needs no transcript at all.
**Interpretation.** High = genuine back-and-forth dialogue. Low with high TTR = monologue. Low with low TTR
= unstructured activity with no teacher check-ins.

### M4 — Longest Teacher Stretch · *acoustic, robust*
```
LTS = max(continuous teacher speech duration)
```
**Interpretation.** The most actionable number on the page, because it is easy to picture and easy to change.
In a 60-minute maker session, any stretch over ~5 minutes means the making stopped.

### M5 — Wait Time (WT1) · *lexical, fragile — always shown with confidence*
```
WT1 = median(start_of_next_student_turn − end_of_teacher_question)
```
Mary Budd Rowe's wait time: the pause after a teacher's question before a student answers.
**Interpretation.** The single most-replicated finding in this literature. Below 1s, answers are short and
students say "I don't know." At **3s or more**, responses get longer, more correct, and more students join in.
**Caveat:** depends on detecting questions in the transcript, which our audio undermines. Gate it behind M6
and fall back to prosodic question detection (rising pitch at the end of a teacher turn) when ASR confidence
is low.
**Measured from raw diarization turns, never the timeline** — splitting segments at speaker changes makes them
tile exactly, so every gap there is 0.00 by construction. Turns overlap, so the answer is found by searching
forward from the question rather than taking the next turn: a student already speaking when the question ends
counts as a wait of zero, an exact `0.000` gap between two separate turns is a snapped boundary and is
discarded. **Withheld below `WAIT_TIME_MIN_SAMPLES = 3` pairs** — a median over one pause is not a median.

### M6 — Analysis Confidence · *the differentiator*
```
C = w1·speech_density + w2·mean_ASR_logprob_norm + w3·audio_completeness + w4·segmentation_margin
```
**Interpretation.** Governs the whole report. Below ~0.4 the dashboard greys the numbers and says
**"this recording is too noisy to analyse reliably"** instead of printing a confident, wrong statistic.

> On our own data, the Shadow Art session should self-report as low-confidence. Building a system that
> knows when to keep quiet is worth more than one extra point of accuracy — and it is precisely the
> "system design thinking" they listed as a grading criterion.

### Also reported (context, not scored)
Hands-on time vs dead time · total lesson duration (from decoded audio, **never** the JSON `duration` field,
which is wrong on 3 of 5 files) · student turn count · question count.

### Deliberately *not* faked
Gender-split participation would land perfectly on MakerGhat's priority on girls, and `boys`/`girls` are in
the JSON — but child voices cannot be reliably sexed from this audio. **List it as roadmap, don't fabricate it.**
Same for per-child identification: 27 children on one phone mic is not a solvable diarization problem, so we
do binary teacher/student classification and say so.

---

## 3. Architecture

```
                    OFFLINE (batch, heavy, runs once)
  data/audio/*.mp3
        │
        ├─ ffmpeg ─────────────► 16 kHz mono wav
        │
        ├─ Silero VAD ─────────► speech / non-speech segments
        │
        ├─ Whisper (Indic) ────► transcript + word timings + logprobs
        │
        ├─ IndicTrans2 ────────► English rendering (for reviewers + question cues)
        │
        ├─ Speaker split ──────► teacher vs student per segment
        │                        (energy + duration + optional pyannote)
        │
        ├─ Metrics engine ─────► M1–M6
        │
        └─ Insight generator ──► headline + suggestion (rule-based, LLM optional)
                │
                ▼
        results/<session>.json     ← small, safe, no raw audio, committable
                │
                ▼
      ┌─────────────────────────────────────────┐
      │  ONLINE (Streamlit, free tier, instant) │
      │  reads JSON → renders dashboard          │
      │  + optional live upload, capped at 2 min │
      └─────────────────────────────────────────┘
```

### Why precompute — three independent reasons

1. **The demo must not break.** A reviewer clicking the mandatory live link gets a full dashboard in under
   a second, not a spinner and a timeout.
2. **Free hosting cannot transcribe an hour of audio.** Streamlit Community Cloud is 1 GB RAM and CPU-only.
   HF ZeroGPU gives an *unauthenticated visitor* 2 minutes of GPU per day — one 60-minute file would blow
   through that instantly.
3. **It is honest about the real product.** A phone in an Indian government school records for an hour and
   syncs later. Batch-then-serve *is* the production shape.

The optional short-upload path keeps it feeling live without depending on it.

### The teacher/student split, concretely

The recording is made **on the teacher's own phone** (`com.example.mneapplication`, in her hand or on her
desk). So the teacher is consistently the nearest, loudest, highest-SNR voice in the room. That gives a
cheap and surprisingly strong feature set:

- per-segment RMS energy (teacher ≫ students)
- segment duration (teacher turns are longer)
- spectral flatness (near-field speech vs far-field babble)
- cumulative-time prior: the longest-talking cluster is the teacher

Start with a 2-means split on those features, validate by hand-labelling ~20 minutes, and add pyannote only
if it measurably beats the heuristic. **Hand-labelling a validation set and reporting real accuracy is the
part that demonstrates AI/ML understanding.** Don't skip it.

---

## 4. Tool stack — everything free

| Layer | Choice | Size | License | Cost |
|---|---|---|---|---|
| Audio decode | **ffmpeg** | — | LGPL | ₹0 (installed) |
| VAD | **Silero VAD** (bundled in faster-whisper) | ~2 MB | MIT | ₹0 |
| ASR runtime | **faster-whisper** / CTranslate2 | — | MIT | ₹0 (installed) |
| ASR baseline | `openai/whisper-small` | 484 MB | MIT | ₹0 (downloaded) |
| **ASR primary** | **`vasista22/whisper-hindi-medium`** | ~1.5 GB | Apache-2.0 | ₹0 |
| ASR stretch | `vasista22/whisper-hindi-large-v2` | ~3 GB | Apache-2.0 | ₹0 |
| ASR alternative | `ai4bharat/indic-conformer-600m-multilingual` | ~2.4 GB | MIT | ₹0 |
| Diarization (optional) | `pyannote/speaker-diarization-3.1` | ~30 MB | MIT, **gated** | ₹0 + HF token |
| Translation | `ai4bharat/indictrans2-indic-en-dist-200M` | ~800 MB | MIT | ₹0 |
| Metrics | numpy / pandas / scipy | — | BSD | ₹0 |
| Dashboard | **Streamlit** | — | Apache-2.0 | ₹0 |
| Charts | Plotly | — | MIT | ₹0 |
| Batch compute | local CPU, or **Google Colab free T4** | — | — | ₹0 |
| Hosting | **Streamlit Community Cloud** | 1 GB RAM, sleeps after 12 h idle | — | ₹0 |
| Optional narrative | Gemini / Claude free tier, or local Qwen2.5-3B | — | — | ₹0 |

**Total cash cost: ₹0.** The real cost is compute hours and your time.

### Why these picks

**`vasista22/whisper-hindi-medium` over vanilla Whisper.** IIT Madras Speech Lab, Bhashini-funded, trained on
GramVaani + ULCA + Shrutilipi + Fleurs. Reports **6.82 WER on Fleurs, 11.38 on Common Voice 11** — versus
vanilla Whisper `small`, which we already saw produce incoherent Devanagari on this audio. GramVaani in the
training mix matters: it is rural Indian telephone speech, far closer to our recordings than clean read speech.

> **Do not quote 6.82% as our expected accuracy.** Fleurs is clean, read, single-speaker audio. On far-field
> phone recordings of 25 shouting children, expect something in the **40–70% WER** range. Saying this
> explicitly in the README is a strength, not a weakness — overclaiming is how you lose a technical reviewer.

**`ai4bharat/indic-conformer-600m` as the hedge.** MIT, 600M params, all 22 scheduled languages, 13.2 WER on
Vaani. Covers Marathi too — and our language probe showed Marathi as the consistent runner-up in Igatpuri.
Running it as a second opinion is cheap and makes the ASR comparison genuinely multilingual.

**pyannote is optional, not core.** MIT-licensed but gated: you need an HF account, must accept conditions on
*both* `speaker-diarization-3.1` and `segmentation-3.0`, and pass a token. Its benchmark DER ranges 7.8%–50%
depending on the corpus; a 27-child classroom sits at the bad end. Treat it as a refinement over the energy
heuristic, and only keep it if measurement says it helps.

**IndicTrans2-dist-200M for translation**, rather than Whisper's built-in `task="translate"`. Whisper's
translate mode degrades further on noisy input; running ASR and MT as separate stages keeps the Hindi
transcript intact and lets us swap either one. 200M distilled is small enough for free hosting.

**Streamlit Community Cloud over HF Spaces.** HF now requires a paid plan for regular Gradio/Docker Spaces;
free accounts get 2 ZeroGPU Spaces with a **5 min/day owner quota and 2 min/day for unauthenticated visitors**.
Since we serve precomputed JSON, we need no GPU at all — and Streamlit's free tier has no per-visitor quota.
Its only real limit is sleeping after 12 hours idle, which costs a reviewer one cold start.

---

## 5. Practical constraints on this machine

### Measured: local CPU transcription is not viable for the batch pass

Benchmarked on this machine (i7-1360P, 16 threads, 15.7 GB RAM, Intel Iris Xe — **no CUDA**),
faster-whisper `small` at int8, `beam_size=1`, VAD on, against a 120 s clip:

| Threads | Wall time for 120 s audio | Speed |
|---|---|---|
| 8 | 282.8 s | **0.42× realtime** |
| 16 | 342.4 s | 0.35× realtime |

**Transcription runs ~2.4× slower than the audio plays.** Projected over the 252 minutes of dataset:

| Model | Projected local batch time |
|---|---|
| `whisper-small` | **~10 hours** |
| `whisper-hindi-medium` | **~30 hours** |
| `whisper-hindi-large-v2` | ~60 hours |

So **the batch pass must run on Google Colab's free T4** — the same job should land in roughly 10–20 minutes.
This is not a preference any more, it's a hard requirement. Keep a local CPU path working for the
offline-first narrative and for short uploads, but never plan to run the full corpus on it.

Two notes on the numbers:
- **More threads made it slower.** 16 threads lost to 8 — the i7-1360P's efficiency cores hurt here.
  Pin `cpu_threads=8`, don't hand it everything.
- **0.42× is well below what this CPU should manage** (faster-whisper `small` int8 usually clears several
  times realtime). Worth 10 minutes checking the Windows power plan and thermal throttling before accepting it —
  but the Colab conclusion holds either way.
- **`D:` has 3.7 GB free.** The model set is ~5 GB. **Set `HF_HOME` to a path on `C:`** (59 GB free) before
  downloading anything else.
- **Use a venv.** The system Python at `C:\Python312` cannot write script shims without admin; `--user`
  installs work but are not reproducible for a reviewer. `python -m venv .venv` in the project root.
- **Force UTF-8.** Devanagari crashes the default cp1252 console. Set `PYTHONIOENCODING=utf-8`.

---

## 6. Build order

| # | Phase | Output | Est. |
|---|---|---|---|
| 1 | venv, `HF_HOME` on C:, repo skeleton, config | runnable scaffold | 0.5 d |
| 2 | **ASR bake-off** — whisper-small vs whisper-hindi-medium vs indic-conformer on the same 5 clips, hand-scored | a table you can put in the README | 1 d |
| 3 | Segmentation + teacher/student classifier + **hand-labelled validation set** | reported accuracy, not a guess | 1 d |
| 4 | Metrics engine M1–M6 + `results/*.json` schema | precomputed results for all 5 sessions | 1 d |
| 5 | Insight generator (rule-based first; LLM narrative optional) | headline + suggestion + WhatsApp-length summary | 0.5 d |
| 6 | Streamlit dashboard — teacher view, trend view, admin view | working UI | 1 d |
| 7 | Deploy + README (structure, approach, assumptions, limitations) | **live URL** | 0.5 d |

**≈ 5.5 days.** Phase 2 is where the marks are — resist the urge to skip it and go straight to the UI.

Order matters: get *one session* end-to-end through phases 2–6 before processing all five. A thin vertical
slice surfaces the schema problems while they're still cheap.

---

## 7. Honest risks

| Risk | Reality | Mitigation |
|---|---|---|
| ASR too poor for wait-time / question metrics | Likely | M6 confidence gate; prosodic question detection as fallback; lead with acoustic metrics |
| Teacher/student split unreliable | Moderate | Hand-labelled validation; report accuracy openly; energy prior is strong because it's the teacher's phone |
| Shadow Art session unusable | Very likely (20% speech) | Ship it as the **worked example of graceful degradation** — that's a feature, not a gap |
| CPU too slow for 5 hours of audio | **Confirmed** — 0.42× realtime, ~30 h for medium | Colab free T4 for the batch pass; local path kept for short clips only |
| Free host sleeps / cold-starts | Certain after 12 h | Keep the served app tiny (JSON only); note it in the README |
| Raw data leaking into a public repo | Fatal if it happens | `.gitignore` already blocks `data/`; commit only derived JSON with names pseudonymised |

---

## 8. What will actually make this stand out

Five things, roughly in order of impact:

1. **Metrics that degrade gracefully.** A dashboard that says *"too noisy to analyse"* beats one that invents
   a number. Nobody else will build the confidence gate.
2. **The truncation bug.** Three of their five files are shorter than their own metadata claims. Reporting
   that — with the file-size arithmetic that proves it — shows you actually looked at the data instead of
   piping it into a model.
3. **The maker-session reframe.** Hands-on time vs dead time, and treating low teacher talk as success.
   Shows you read their curriculum, not just the assignment PDF.
4. **A real ASR comparison with hand-scored results**, plus an honest statement that clean-benchmark WER
   does not transfer to this audio.
5. **The teacher-first output** — one headline, one suggestion, a WhatsApp-length summary, and a
   same-teacher trend across two sessions. It answers *"so what?"*, which is the question most submissions
   leave hanging.

---

## Sources

ASR and models —
[ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) ·
[vasista22/whisper-hindi-medium](https://huggingface.co/vasista22/whisper-hindi-medium) ·
[vasista22/whisper-hindi-large-v2](https://huggingface.co/vasista22/whisper-hindi-large-v2) ·
[AI4Bharat IndicWhisper](https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicWhisper) ·
[Vistaar benchmark](https://github.com/AI4Bharat/vistaar) ·
[IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) ·
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

Hosting —
[HF Spaces overview](https://huggingface.co/docs/hub/spaces-overview) ·
[HF ZeroGPU quotas](https://huggingface.co/docs/hub/spaces-zerogpu) ·
[Streamlit Community Cloud limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)

Classroom discourse —
[TeachFX Analytics](https://teachfx.com/analytics) ·
[TeachFX measurement profile, WestEd](https://mpm.wested.org/measure/teachfx-app-student-teacher-talk-measurement-tool/) ·
[Next-Gen Classroom Observations, Education Next](https://www.educationnext.org/next-gen-classroom-observations-powered-by-ai/) ·
[Wait time after questions (NIH)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11202393/) ·
[Turn taking and wait time](https://www.sciencedirect.com/science/article/abs/pii/S037821661300324X) ·
[IRF patterns and turn-taking](https://pdfs.semanticscholar.org/e033/c5a1054151daba6b66b93b044b9a46062388.pdf) ·
[ClassInSight](https://arxiv.org/pdf/2403.00954) ·
[Noise-robust classroom ASR (ICASSP 2025)](https://arxiv.org/pdf/2409.14494)
