# Classroom Voice Analytics

Turns a phone recording of an Indian classroom into a small number of things a teacher can
act on — and **refuses to print a number it cannot stand behind.**

Built for the MakerGhat Jr. Full Stack Developer pre-work assignment (Task 1).

**Live demo:** _(deployment pending — see [Deploying](#deploying))_

---

## What it does

Five real Hindi lessons from Igatpuri, Nashik — 3 teachers, 8–27 children, 252 minutes of
audio recorded on a phone sitting somewhere in the room. Out the other end:

- a Hindi transcript
- who was talking, teacher or student, second by second
- five engagement metrics, each carrying its own formula, explanation and interpretation
- a plain-English headline and one thing to try next lesson
- a confidence score that decides how much of the above anyone is allowed to see

The last point is the design. Classroom audio is bad audio: 27 children, one microphone,
hammering and cutting in the background. A pipeline that always produces five confident
numbers on that input is producing fiction. This one measures how much it can trust itself
and withholds the metrics that do not clear the bar — **a withheld metric renders as `—`,
never as `0`.**

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -r requirements.txt
```

Diarization is gated on Hugging Face, and on **three** repos rather than the one the model
card names. Accept the conditions on `pyannote/speaker-diarization-community-1`,
`pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM` — accepting
`speaker-diarization-3.1`, which is what most instructions point you at, is not enough:
pyannote 4.x redirects that id to `community-1`, which is gated separately. Then:

```bash
export HF_TOKEN=hf_your_read_token
```

Run the offline pass, then the dashboard:

```bash
python -m src.run_pipeline --minutes 4
```

```bash
npm --prefix app start
```

Without a token the pipeline still runs: it falls back to energy-based attribution, says so
in `models.attribution`, and the confidence gate withholds accordingly.

---

## How it is put together

```
audio ──▶ ingest ──▶ diarization ──▶ timeline ──▶ attribution ──▶ metrics ──▶ insight
                          │                                          │
                          └────────────▶ ASR ────────────────────────┘
                                                                     │
                                                          results/<id>.json
                                                                     │
                                                          Angular dashboard
```

**The offline pass and the UI are separated by a JSON file on purpose.** Everything
expensive — transcription, diarization, metrics — happens once, offline, and writes
`results/<session_id>.json`. The dashboard is a static reader of that file. It opens
instantly, needs no backend, deploys anywhere, and costs nothing to host. It also means the
whole pipeline can be rewritten without touching the UI, as long as the JSON shape holds.

This also matches MakerGhat's stated constraint: their platform is **offline-first**. Every
model here runs locally on CPU. There is no API key in the critical path and no per-minute
cost.

### Repository layout

| Path | What is in it |
|---|---|
| `src/ingest.py` | Session discovery, metadata, audio probing, truncation detection |
| `src/asr/` | Three interchangeable ASR backends behind one `Transcript` contract |
| `src/signal_layer.py` | Speech / hands-on / dead-air timeline |
| `src/diarization.py` | pyannote turns, speech map, teacher identification, segment splitting |
| `src/attribution.py` | Energy-based fallback attribution |
| `src/metrics.py` | M1–M6, the confidence gate |
| `src/insight.py` | Rule-based headline, suggestion, WhatsApp message |
| `src/pipeline.py` | The `results/*.json` contract |
| `src/evaluation/` | WER/CER scoring, the ASR bake-off, hallucination signals, labelling |
| `app/` | Angular dashboard (static reader) |
| `tests/` | 591 tests |
| `docs/` | Measurements and decisions, with the numbers that drove them |

---

## The metrics

Each metric ships its formula, an explanation of what it measures, and an interpretation of
what to do about it — these travel inside the JSON, so the dashboard never has to hardcode
them and they cannot drift from the code that computes them.

### M1 — Teacher talk ratio

```
teacher_speech_sec / (teacher_speech_sec + student_speech_sec)
```

Share of talking time held by the teacher. Observed average in conventional classrooms is
around 61%; a hands-on maker session should sit well below that, so **0.40 is the target,
not the norm**. Above 0.70 the activity has become a lecture.

### M2 — Student participation

```
(student_turns / roster_total) / lesson_minutes * 60
```

Turns per child per hour. **Normalised by roster size**, because eight children each
speaking twice is a different lesson from twenty-seven children each speaking twice, and the
raw turn count cannot tell them apart.

### M3 — Interaction density

```
speaker_switches(teacher <-> student) / lesson_minutes
```

How often the floor changes hands. Needs no transcript at all, so it survives audio the ASR
cannot read. High means genuine back-and-forth. Low with a high talk ratio is a monologue;
low with a *low* talk ratio is unstructured activity with no teacher check-ins — two very
different problems that look identical without M1 beside it.

### M4 — Longest teacher stretch

```
max(continuous teacher speech, merging pauses under 3s)
```

The single longest uninterrupted stretch of teacher talk, in seconds. The most actionable
number on the page, because it is the easiest to picture and the easiest to change. In a
60-minute maker session, anything over about five minutes means the making stopped.

### M5 — Wait time (WT1)

```
median(start_of_next_student_turn - end_of_teacher_question)
```

Mary Budd Rowe's wait time: the pause after a question before a student answers. Below 1s,
answers are short and students say "I don't know". At 3s or more, responses get longer, more
correct, and more students join in. It is the most replicated finding in this literature.

This is the only **lexical** metric — it needs question detection, so it needs a transcript,
so it is the first to break in a noisy room. It is measured from raw diarization turns
rather than the timeline, and withheld below three question–answer pairs. See
[Wait time is harder than it looks](#wait-time-is-harder-than-it-looks).

### M6 — Analysis confidence

```
C = 0.30*speech_density + 0.30*attribution + 0.20*completeness + 0.20*asr
```

Governs everything above. Three verdicts:

| Verdict | What the reader sees |
|---|---|
| `usable` | Everything |
| `partial` | Acoustic metrics; the fragile ones withheld individually |
| `unreliable` | **No numbers at all** — a note about the recording instead |

Metrics also carry their own confidence, so M5 can be withheld on a session where M1 is
fine. And M1–M4 are withheld wholesale whenever the teacher/student split is below its
floor — a talk ratio computed from a speaker split you do not believe is not a weak number,
it is a meaningless one.

---

## The decisions, and what they were measured against

Every choice below was made by measurement, and the losing options are in `docs/` with their
numbers. Three of them were reversals of something that had already been built.

### ASR: IndicConformer, chosen on real WER

Measured on `google/fleurs` hi_in, 20 samples, corpus WER (total edits ÷ total words, not
the mean of per-sample rates):

| Model | WER | Notes |
|---|---:|---|
| **AI4Bharat IndicConformer (ONNX)** | **17.3%** | CTC — structurally cannot hallucinate |
| Groq `whisper-large-v3` | 26.9% | Hosted, needs a key |
| `whisper-small` | 63.8% | Returns near-gibberish on this audio |
| vasista22 Hindi fine-tune | 84.4% | Transcribes correctly, then rambles into news prose |

IndicConformer also runs the whole 252-minute corpus in **67 minutes on CPU** with no torch
dependency. Switching backend is one line in `src/config.py`.

That 84.4% is worth a sentence, because it validated the harness rather than breaking it:
the model's *best* sample scored 6% against a published 6.8%. It transcribes accurately and
then keeps going, inventing fluent news bulletins past the end of the audio. A
`words_per_second` signal now catches exactly that.

**Caveat, stated plainly:** FLEURS is clean read prose. It ranks the models; it does not
predict accuracy on 27 children near one phone.

### Hallucination detection

Good models fail fluently, which is worse than failing loudly. Four independent signals:
`repetition_rate` (looping), `artefact_hits` (known invented text — Groq produces
"subscribe"), `words_per_second` (rambling past the audio), and `cross_model_agreement`
(fluent nonsense agrees with nobody). The last one caught a wav2vec2 model that scored a
perfect 1.000 on Devanagari-shaped noise.

### Attribution: diarization, after energy was measured and thrown away

The first design assumed **the teacher is nearest the microphone** and clustered segments by
loudness. Measured against hand labels: **41.7%** — worse than chance, and beaten outright
by a constant "always teacher" baseline. The assumption is simply false; the phone hears one
room at one level.

Replaced with pyannote diarization, which assumes instead that **the teacher is one person
who holds the floor while many students each speak briefly** — true wherever the phone is
sitting. Measured **88.9%** on the same labels.

### Read the 88.9% with care

**All 36 hand labels say `teacher`. Not one says `student`.** A classifier that never
outputs "student" scores 100% on that set. The number measures teacher recall and nothing
else — and that is precisely why the following bug survived it.

---

## Bugs that only real audio would have found

The tests were green through every one of these. They are here because they are the most
useful thing in the repo to talk about.

### Students were handed to the teacher by a tie

`speaker_for_span` took the first strict maximum overlap. But diarization turns *overlap*,
and a teacher's turn routinely **encloses** a student's:

| Session | Student turns | Fully inside a teacher turn | Mislabelled |
|---|---:|---:|---:|
| OD11163_2026-01-28 | 25 | 25 | **25 (100%)** |
| OD11165_2026-01-06 | 38 | 38 | **38 (100%)** |

When the span *is* the student's turn, both turns cover it identically, so `>` kept whichever
came first in the list — always the enclosing teacher. The result was M1 reporting **100%
teacher talk** and M3 **zero exchanges** on two sessions, while M5, reading the same audio,
reported **28 and 11 answered student questions.** Both numbers printed on the same card.

Ties now go to the speaker whose turn is most *contained* in the span. Containment is only
ever a tie-break, so a student holding the floor is still not displaced by a teacher chipping
in for a second.

### Silero was calling speech "hands-on activity"

The signal layer classified anything loud and non-speech as children building things. With
Silero missing most of the speech on the two maker sessions, that rule claimed it:

| Session | Diarized speech | Landed in `speech` | Landed in `handson` |
|---|---:|---:|---:|
| OD11166_2026-01-12 (students only) | 70s | 3s | **67s** |
| OD11166_2026-01-20 (students only) | 118s | 9s | **109s** |

Children who spoke for 67 and 109 seconds were published as 1.6 and 6.5. Since the pipeline
already pays for diarization, and pyannote's segmentation is a speech detector before it is a
speaker detector, **its turns are now the speech map.** Silero remains the fallback when
there is no token, and `models.vad` records which one actually ran.

### Wait time is harder than it looks

Three separate defects, none visible from the tests:

1. **Pairing by adjacency.** Taking the turn *after* a question discarded it whenever the
   teacher spoke again first — which is most of the time. 1 of 3 questions survived on one
   session, 5 of 8 on another.
2. **Turns overlap.** Adjacency matched long teacher turns with student turns nested inside
   them, producing gaps as negative as **−11.0s**, all silently dropped.
3. **Zero is not always zero.** 3 of 11 surviving samples were exactly `0.000`. Real pauses
   never land on exactly zero; that spike is two turns sharing a snapped boundary.

Now: search forward from each question for the first other-speaker turn; count a student
already talking as a genuine zero (an interruption is a real observation); discard exact
`0.000` between separate turns as unmeasured. And a **three-pair minimum** — the first run
published `0.00s` from a single question.

### The dashboard was serving a stale copy

`app/public/results/` was a hand-copied duplicate that silently went out of date, so the
dashboard showed pre-fix numbers for a full day with nothing to indicate it. There is now
one copy: the build maps `results/` in directly, and a test asserts the duplicate has not
come back.

### A cross-check that needs no ground truth

The hand labels could not catch the enclosure bug, so consistency does the work instead.
`attribution_lost_students` fires when diarization heard more than two seconds of another
speaker but the timeline contains no student time at all. **Two views of one recording
disagreeing about whether a child spoke is a bug every time**, and it costs nothing to check.

---

## What it produces on the five sessions

Four-minute window per session, IndicConformer ASR, pyannote diarization for both the speech
map and the speaker split:

| Session | Verdict | Conf | M1 teacher | M2 turns/child/h | M3 exch/min | M4 stretch | M5 wait |
|---|---|---:|---:|---:|---:|---:|---:|
| OD11163_2025-12-23 | usable | 0.91 | 85% | 18.4 | 11.8 | 27s | 1.16s |
| OD11163_2026-01-28 | usable | 0.82 | 93% | 9.6 | 7.8 | 43s | 4.57s |
| OD11165_2026-01-06 | usable | 0.92 | 87% | 60.0 | 16.0 | 24s | 4.37s |
| OD11166_2026-01-12 | usable | 0.78 | **57%** | 23.3 | 15.0 | 17s | 0.28s |
| OD11166_2026-01-20 | usable | 0.83 | **61%** | 46.6 | 22.5 | 18s | — |

The last two are the hands-on sessions. They read 93% and 80% teacher talk until the Silero
bug above was fixed; they now read 57% and 61%, which is what a maker lesson should look
like. As a cross-check, the raw diarization airtime on OD11166_2026-01-12 is 91.4s teacher
against 69.9s student — **56.7%**, against the 57% the pipeline reports.

M5 is withheld on the last session: two question–answer pairs, below the floor of three, and
the JSON says exactly that rather than printing a number.

## Assumptions

Stated because they are all falsifiable, and one of them already was.

1. **The teacher is one person who holds the floor; students are many who each speak
   briefly.** This is what identifies the teacher. It fails for genuine group work with no
   dominant speaker — the confidence score drops when it does, but it does not detect it.
2. **The louder speaker is the teacher.** *False, measured at 41.7%.* Retained only as the
   fallback when no HF token is available, and labelled as such in the output.
3. **Loud non-speech is children making things, not dead air.** Central to not scoring a
   successful maker lesson as a failed one. It is also what caused the Silero bug above.
4. **A question can be found lexically.** Hindi has no inverted word order, so this is an
   interrogative word or a question mark. Deliberately fragile, and the reason M5 is gated
   hardest.
5. **A four-minute window represents the lesson.** It does not, really. It is a runtime
   choice, not a claim — `--minutes` controls it and the full corpus takes about 3.4 hours.
6. **FLEURS ranks ASR models for this audio.** Clean read prose is not a classroom. The
   ranking is informative; the absolute WER is not transferable.

## Limitations

- **`speech_density` no longer discriminates.** It is `min(1, speech_share / 0.5)`, calibrated
  when Silero was finding about 58% of the speech. pyannote finds 60–83%, so the component
  now reads **1.00 on every session** and 30% of the confidence weight is a constant. The
  overall score still varies (0.78–0.92) because the other three components do, and the
  verdict changes it caused are correct — a recording that genuinely is 60% speech *should*
  score better than Silero's reading of 9%. But the knee wants recalibrating against the new
  detector, and there is no ground truth to fit it to yet, so it has been left alone and
  written down rather than quietly re-tuned.
- **The validation set has no student rows.** Until the sheets are regenerated from the split
  timeline and re-labelled by ear, student recall is unmeasured. Everything about student
  detection currently rests on internal consistency, not ground truth.
- **Three of five recordings are truncated** against their own metadata — a bug in the
  capture app, surfaced in the Data Quality view because it matters more to the M&E team
  than any single lesson metric.
- **Transcripts are approximate** and presented as such. The metrics that matter most
  (M1–M4) are acoustic and do not depend on them.
- **Diarization costs 1.24× realtime** on CPU. Fine as a nightly batch, not interactive.

## Privacy

The recordings are **identifiable children**, teacher names and school GPS. `data/` is
gitignored and no raw audio is committed. Teachers appear in every published file as an
alias (`Teacher A`), never a name — `Session.to_public_dict()` enforces this, and a test
holds it in place.

---

## Testing

```bash
python -m pytest
```

591 tests, none skipped. Tests are written before the code they cover, and the ones worth
reading are the regression tests above — each names the real session and the real number that
exposed the bug.

## Deploying

The dashboard is a static bundle:

```bash
npm --prefix app run build
```

Publish `app/dist/dashboard/browser` on any static host with an SPA rewrite. `results/*.json`
is copied in by the build, so the deployed site carries its own data and needs no backend.

## Documentation

| Document | Contents |
|---|---|
| [`docs/data-notes.md`](docs/data-notes.md) | What is actually in the dataset, and every measurement that changed a decision |
| [`docs/asr-options.md`](docs/asr-options.md) | The full ASR landscape and the measured bake-off |
| [`docs/task1-plan.md`](docs/task1-plan.md) | Metric definitions with sources, tool stack |
| [`docs/architecture.md`](docs/architecture.md) | Diagrams and the user workflow |
