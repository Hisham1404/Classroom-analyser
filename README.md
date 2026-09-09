# Classroom Voice Analytics

Turns a phone recording of an Indian classroom into a small number of things a teacher can
act on — and **refuses to print a number it cannot stand behind.**

Built for the MakerGhat Jr. Full Stack Developer pre-work assignment (Task 1).

**Live demo:** <https://app-one-blue-47.vercel.app>

---

## What it does

Five real Hindi lessons from Igatpuri, Nashik — 3 teachers, 8–27 children, 252 minutes of
audio recorded on a phone sitting somewhere in the room. Out the other end:

- a Hindi transcript
- who was talking, teacher or student, second by second
- how many questions the teacher asked, and how many a student answered
- seven engagement metrics, each carrying its own formula, explanation and interpretation
- a plain-English headline and one thing to try next lesson
- a confidence score that decides how much of the above anyone is allowed to see

The last point is the design. Classroom audio is bad audio: 27 children, one microphone,
hammering and cutting in the background. A pipeline that always produces seven confident
numbers on that input is producing fiction. This one measures how much it can trust itself
and withholds the metrics that do not clear the bar — **a withheld metric renders as `—`,
never as `0`.**

---

## Quick start

**ffmpeg must be on PATH before anything else.** The ASR layer shells out to it to cut
long recordings into chunks, so transcription fails without it.

```bash
winget install Gyan.FFmpeg     # Windows   ·   macOS: brew install ffmpeg
                               #              Debian/Ubuntu: sudo apt install ffmpeg
ffmpeg -version                # confirm it resolves
```

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -r requirements.txt
```

Put the recordings in `data/audio/`. That directory is git-ignored and stays that way —
it is real classroom audio of identifiable children.

Then run it. **No API key and no token are needed to transcribe:**

```bash
python -m src.run_pipeline --minutes 4        # 4 minutes per session, to try it out
python -m src.run_pipeline                    # the whole corpus, ~67 min on CPU
```

The first run downloads the ASR model (~250 MB) from Hugging Face into the cache that
`src/config.py:configure_hf_cache()` pins via `HF_HOME`, deliberately off the project
drive. Every later run reads it from disk, and transcripts are content-hash cached in
`.cache/asr/`, so re-running the same audio does no work twice.

**Diarization turns are cached the same way, in `.cache/diarization/`.** It is the most
expensive step in the project - 1.24x realtime, about three hours for this corpus - and
until the cache existed its output was never persisted, so every change to a metric
downstream of it cost a full re-run. A cache hit also needs no `HF_TOKEN`: the token gates
the model *download*, not the arithmetic, so once the turns are on disk these results
reproduce without accepting four gated model licences first.

Results land in `results/<session_id>.json`. Serve the dashboard over them with:

```bash
npm --prefix app install
npm --prefix app start
```

### Transcription on its own

To transcribe a single file without the rest of the pipeline:

```bash
python -m src.cli data/audio/<file>.mp3 --backend indic
```

`--backend` overrides `ASR_BACKEND` in `src/config.py` for one run. All three backends
return the identical `Transcript`, so nothing downstream changes:

| Backend | What it is | Needs | Speed |
|---|---|---|---|
| `indic` **(default)** | AI4Bharat IndicConformer, ONNX | nothing but the install | ~3.8x realtime, all CPU |
| `local` | Hindi-tuned Whisper, CTranslate2 | `faster-whisper` | ~0.11x realtime (~38 h for the corpus) |
| `groq` | hosted `whisper-large-v3` | `GROQ_API_KEY` | network-bound |

Two things about `indic` worth knowing, both of which cost real debugging time:

- Its ONNX export has a **fixed positional-encoding length**. Anything past ~100 s fails
  with `Attempting to broadcast an axis ... 2501 by 7501`, so audio is chunked at 90 s
  with 2 s of overlap. Before that was found it was silently killing ASR on every session
  and leaving acoustic-only results.
- It is a **CTC** model, so unlike the Whisper family it cannot hallucinate fluently. The
  Whisper backends both do: Groq drifts into "subscribe", the Hindi fine-tune into news
  bulletins.

### Attribution, which is the part that needs a token

Diarization is gated on Hugging Face, and on **four** repos rather than the one the model
card names. Accept the conditions on `pyannote/speaker-diarization-community-1`,
`pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM` — accepting
`speaker-diarization-3.1`, which is what most instructions point you at, is not enough:
pyannote 4.x redirects that id to `community-1`, which is gated separately. Then:

```bash
export HF_TOKEN=hf_your_read_token
```

**This is only for telling teacher from student, never for transcription.** Without a
token the pipeline still runs end to end: it falls back to energy-based attribution, says
so in `models.attribution`, and the confidence gate withholds accordingly — but measured
teacher recall drops from 88.9% to 41.7%, so the talk-time metrics get much weaker.

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
| `src/metrics.py` | M1–M5, M7–M8, the confidence gate (M6) |
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

### M7 — Teacher questions asked

```
count(teacher turns whose transcript reads as a question)
```

Every question the teacher asked, **answered or not**. The unanswered ones are the point:
M5 can only see a question a student replied to, so a room where nobody answers looks
identical there to a room where nobody was asked.

Detection is lexical — a Hindi interrogative from a fixed word list, or a question mark.
Hindi does not invert word order for questions, so without prosody there is little else to
go on, and the detector is deliberately simple and deliberately fragile.

**One utterance is one question.** Diarization and ASR segment the audio independently, so
a single sentence often straddles several diarization turns. Each transcript segment is
therefore attributed to the first teacher turn that covers it, and a turn contributing no
new words is not a new question. Without that rule the count came out **1.6× too high** on
real audio — see [the post-mortem](#publishing-a-count-exposed-a-flaw-a-median-had-been-hiding).
It under-counts a teacher who asks three questions inside one unbroken turn, which is the
safe direction and matches Rowe's unit anyway: wait time is the pause after she stops. It will miss a
question asked in a flat statement ("बताओ" caught, "तो अब हम आगे बढ़ते हैं" not) and it will
occasionally fire on a rhetorical one.

The published value is a **count**, because that is what the brief asks for. A count is not
comparable between a 20-minute session and a 70-minute one, so the card also states the
rate per hour, and the band is computed from that rate rather than from the count.

### M8 — Student responses

```
count(questions answered by a student within 15s)
```

How many of those questions a student actually took up. Two rules decide what counts:

* A student **already talking** as the question ends counts. That is an answer with no
  wait, not a missing one.
* A student who speaks **more than 15 seconds later** does not. That is the next exchange,
  not this one.

M8 is not the same as M5's sample count, and the difference is deliberate. Every measurable
wait is a response, but a response whose two turns share a boundary to the microsecond has
no silence to report — the boundary was snapped, not measured — so it counts here and not
there. The invariant is `wait_samples <= responses <= questions`, and the distance between
the terms is information: it separates *nobody answered* from *the answer had no measurable
pause in front of it*.

**M7 and M8 read together are the metric.** A high question count with a low answered share
is the signature of rhetorical questions the teacher answers herself — the most common
thing this measurement finds, and invisible to every other number on the page.

Both are lexical, so they share M5's dependencies: no transcript, no counts, and no counts
when the teacher/student split is below its floor either, because counting *teacher*
questions means knowing which speaker the teacher is. They do **not** share M5's three-pair
sample floor — a median over two pauses is not a median, whereas a count of two questions is
exactly two questions. A session with no questions in it publishes `0`, not `null`; zero
questions is a finding, and the dash is reserved for *could not measure*.

> **Numbering:** the metric keys run M1–M5, M7, M8. M6 is the confidence score below, which
> lives in the JSON's own `confidence` block rather than in `metrics` — it governs the
> others rather than sitting beside them. The gap is deliberate, so that M6 means the same
> thing in the code, the JSON and this README.

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

### Publishing a count exposed a flaw a median had been hiding

The brief asks for a teacher question count and a student response count. The detector had
existed since Phase 4, so this looked like plumbing: run the walk, publish two integers.
The first numbers came back at **286 questions per hour** on a 22-minute session, which is
nearly five a minute. That is not impossible for a teacher on a roll, so it needed
checking rather than believing.

`text_for_span` returns **every transcript segment overlapping a span**. Diarization and
ASR segment audio independently, so a single teacher sentence routinely straddles two or
three diarization turns — and each of those turns read the whole sentence, found the
interrogative in it, and counted a question:

```
[   61.8s] क क्या है सेकंड रो हैब भी आप कुछ काम करग…
[   70.6s] क क्या है सेकंड रो हैब भी आप कुछ काम करग…   ← the same sentence
[   78.0s] क क्या है सेकंड रो हैब भी आप कुछ काम करग…   ← and again
```

Measured across the two sessions that had finished processing: **153 flagged spans
carrying 98 distinct texts, and 112 carrying 69** — a consistent **1.6×** inflation, which
is what a mechanical cause looks like rather than a noisy one.

**The bug was older than the counts.** The same walk has fed M5 since Phase 4, so its
"measured from 114 question–answer pairs" was inflated the same way. It never showed,
because **M5 publishes a median and a median barely moves when its samples are
duplicated** — the duplicates cluster around the same value. A count moves by the full
factor. Publishing the plain number is what made a year-old flaw visible, which is a
decent argument for publishing plain numbers.

The fix attributes each transcript segment to the **first teacher turn that covers it**, so
one utterance is one question. That immediately broke the timing in a way worth recording:
if the question is owned by the first turn, the wait is measured from *there*, while the
teacher is still mid-sentence — so the real answer arrives outside the 15-second window and
is filed as *unanswered*. The question's end is therefore the end of her **run** of turns
over that utterance, and the run stops when the utterance's words run out, not when she
eventually stops talking.

Two of the eight mutations run against the fix survived the first pass, both because
nothing pinned *where* that run ends. A third survivor turned out to be an equivalent
mutant — the `if not fresh` guard is unreachable, since the joined text of no segments is
`""` and `is_question("")` is already `False`. It stays in for readability, and it is
honest to say no test covers it.

### A cross-check that needs no ground truth

The hand labels could not catch the enclosure bug, so consistency does the work instead.
`attribution_lost_students` fires when diarization heard more than two seconds of another
speaker but the timeline contains no student time at all. **Two views of one recording
disagreeing about whether a child spoke is a bug every time**, and it costs nothing to check.

---

## What it produces on the five sessions

**The whole corpus, end to end: 251.8 minutes, every session analysed at 100% of the audio
that exists on disk.** IndicConformer ASR, pyannote diarization for both the speech map and
the speaker split. 3,405 transcribed utterances.

| Session | Verdict | Conf | M1 teacher | M2 turns/child/h | M3 exch/min | M4 stretch | M5 wait |
|---|---|---:|---:|---:|---:|---:|---:|
| OD11163_2025-12-23 | usable | 0.88 | 70% | 24.5 | 14.4 | 52s | 2.19s |
| OD11163_2026-01-28 | usable | 0.82 | 92% | 9.3 | 7.4 | 49s | 6.12s |
| OD11165_2026-01-06 | usable | 0.89 | 63% | 88.3 | 16.2 | 48s | 1.30s |
| OD11166_2026-01-12 | usable | 0.73 | 47% | 24.1 | 12.5 | 15s | 0.94s |
| OD11166_2026-01-20 | usable | 0.86 | 46% | 42.6 | 16.9 | 35s | 0.89s |

### Sampling the first four minutes overstates teacher talk — every time

An earlier pass measured a four-minute window per session. Re-running on the full recordings
moved **M1 in the same direction on all five**, never the other way:

| Session | M1 on 4 min | M1 on the full lesson | shift |
|---|---:|---:|---:|
| OD11163_2025-12-23 | 85% | 70% | −15pp |
| OD11163_2026-01-28 | 93% | 92% | −1pp |
| OD11165_2026-01-06 | 87% | 63% | **−24pp** |
| OD11166_2026-01-12 | 57% | 47% | −10pp |
| OD11166_2026-01-20 | 61% | 46% | −15pp |

Five out of five in the same direction is not sampling noise, it is a **bias in where the
window sits**. A lesson opens with instructions and settling-in, which is the most
teacher-dominated stretch it will ever have; the group work that gives students the floor
comes later. Any short sample anchored at the start will therefore flatter the teacher-talk
figure, and by up to 24 points.

Worth stating plainly because the cheap version of this product is exactly that — sample a
few minutes and report a number. On this corpus that would have been wrong on every session.

The corpus now spans **46% to 92%** teacher talk. The two hands-on sessions sit at the bottom
(47% and 46%) and the lecture-style ones at the top, so activity type predicts participation
far better than teacher identity does — the same teacher (OD11166) runs both hands-on
lessons, and OD11163 appears at both 70% and 92% on different days.

As a cross-check on the attribution, raw diarization airtime on OD11166_2026-01-12 is 91.4s
teacher against 69.9s student — **56.7%** — against the 57% the pipeline reported for the
same window before the full run.

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
- **M8's response rate barely discriminates on this corpus.** Four of the five sessions land
  at 92-100% answered, because a "response" is any student turn beginning within 15 s of the
  question - and in a room of 8 to 27 children, almost any pause gets filled by somebody. It
  measures *someone spoke soon after*, which is a proxy for *the question was taken up*, not
  proof of it. The one session that does stand out is the informative one: 73% answered with
  the longest wait time in the corpus, on the recording where the teacher held **92%** of the
  talking time. That is a coherent story - dominate the floor and children answer less, and
  slower - but one session is an anecdote, not a validated signal. Read M7 and M8 together,
  and read the wait time beside them.
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

629 tests, none skipped. Tests are written before the code they cover, and the ones worth
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
