# Task 1 Dataset — What We Actually Got

Downloaded 2026-09-05 from the Drive folder in the assignment PDF, into `data/audio/`.
243 MB, 5 sessions, each a folder of `.mp3` + `.jpg` + `.json`.

Probes are reproducible: `scripts/probe_audio.py` (language + sample ASR), `scripts/probe_vad.py` (speech density).

---

## 1. This is real, unredacted field data

Not synthetic samples. Every session carries:

- **Audio of identifiable children** (8–27 per class) and their teacher
- **Teacher's real name** (`photoMetadata.teacherName`) and staff ID (`teacherId`)
- **A classroom photograph** — children's faces
- **GPS coordinates + street address** of the school (2 of 5 sessions)
- **Device file paths** from the capture phone

> ⚠️ **Do not commit `data/` to a public GitHub repo.** The assignment requires a public repo and a
> public live demo. Ship the code, ship derived/anonymised outputs, keep the raw media out.
> A `.gitignore` excluding `data/` is already in place. Saying this explicitly in the README is a
> point in your favour — it shows you thought about what handling children's classroom audio implies,
> which is exactly the judgement an M&E-driven org should want.

**Their capture app is identified:** `com.example.mneapplication` — an Android M&E app, sideloaded.
The `com.example.` package prefix is Android Studio's default placeholder and is **blocked from Google Play**,
so "the existing MakerGhat mobile app" from the AI/ML intern JD is an internal prototype, not a store app.
That's consistent with everything else in `company-brief.md` §4.

---

## 2. Session inventory

| Session | Teacher | Activity | Students | Meta min | Audio min | Lost | Speech % | Turns/min | Mean dB | Peak dB | GPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| OD11163_2025-12-23 | Santana | Trumpet | 22 (15b/7g) | 67.7 | 67.7 | — | 61.3 | 14.5 | −27.0 | −1.9 | ✓ |
| OD11163_2026-01-28 | Santana | Trumpet | 25 (13b/12g) | 64.0 | **22.0** | **−66%** | 71.7 | 15.5 | −25.7 | −2.7 | ✓ |
| OD11165_2026-01-06 | Aayushi | Trumpet | 8 (5b/3g) | 68.9 | 64.3 | −7% | **85.6** | 16.2 | −23.5 | −0.0 | ✗ |
| OD11166_2026-01-12 | Sudipto | Shadow Art | 27 (17b/10g) | 57.2 | **37.5** | **−34%** | **20.4** | 6.7 | −18.2 | 0.0 | ✗ |
| OD11166_2026-01-20 | Sudipto | Wind Anemometer | 19 (9b/10g) | 60.2 | 60.2 | — | 34.8 | 9.0 | −20.5 | 0.0 | ✗ |

**Totals:** 318 min claimed · **252 min actually present** · 66 min (21%) missing.
All files: MP3, mono, 44.1 kHz, ~130 kbps.

**Three teachers, three sites, Dec 2025 – Jan 2026.** Santana and Sudipto each have two sessions —
enough for a *same-teacher-over-time* comparison, which is a far more interesting demo than five
unrelated snapshots. Use it.

---

## 3. Four data-quality findings worth naming in the README

**(a) Three of five recordings are truncated.** The JSON `duration` field and the actual MP3 length
disagree by 42 min, 20 min and 5 min. Confirmed by file size, not just container metadata —
the 64-minute session is a 21 MB file that can only hold 22 minutes at 129 kbps. Their capture app is
losing audio, probably on backgrounding or storage pressure.
*Consequence: never normalise a metric by `duration` from the JSON. Use the decoded audio length.*
This is a genuine bug in their pipeline and a strong thing to surface.

**(b) Clipping.** Three sessions peak at 0.0 dB — the mic is driven into distortion.
Automatic gain on the phone, plus children shouting near it. Distorted speech transcribes badly.

**(c) There is no silence.** At a −45 dB threshold, **zero** silent runs longer than one second across
all five hours. The noise floor never drops. So the assignment's optional "silence duration" metric
cannot be done with a dB gate — it needs VAD, and it should be reported as *"no-speech time"* rather
than silence, because that is what is honestly measurable.

**(d) GPS is missing on 3 of 5.** Offline-first field reality. Any pipeline must treat every
metadata field as optional.

---

## 4. Language: Hindi

18 of 20 sampled clips detect as **Hindi**, typically at 0.85–0.99 confidence.
**Marathi** surfaces as the runner-up in several clips, which fits — the one session with GPS is in
**Igatpuri, Nashik district, Maharashtra**, matching MakerGhat's district-wide Nashik program.
English loanwords appear throughout ("okay", "size", "shoes").

Treat it as **code-mixed Hindi–Marathi–English**, transcribe with `language="hi"`, and say so in the README.
That satisfies the "at least one Indian language" requirement with evidence rather than assumption.

---

## 5. Transcription quality — CORRECTED 2026-09-05 (Phase 2)

> **The original finding below was measured with `whisper-small` and is wrong as a general
> claim.** A Hindi-fine-tuned model produces genuinely usable transcripts on this same audio.
> See `eval/asr_bakeoff.md`. The corrected position:
>
> | | `faster-whisper-small` | `vasista22-whisper-hindi-large-v2` (ct2-int8) |
> |---|---:|---:|
> | Composite | 0.533 | **0.911** |
> | Mean logprob | −1.32 | **−0.41** |
> | Devanagari ratio | 0.64 | **1.00** |
> | Sec per 30 s clip | 42 | 270 |
>
> On one clip `small` returned a wall of gibberish salted with Latin and Cyrillic characters,
> while the Hindi model returned coherent narrative Hindi — a teacher telling a story about a
> flood reaching a village. On another, `small` returned **nothing at all** and the Hindi model
> returned a clean 143-character sentence.
>
> **What this changes:** the lexical layer is worth building on after all, so question detection
> and wait time move from "probably unusable" to "worth attempting". The acoustic-first
> architecture still stands — it is what makes the pipeline degrade gracefully — but the
> transcript is no longer assumed to be garbage.
>
> **What it costs:** the Hindi model runs at ~9x *slower* than realtime on this CPU, so the full
> 4.2-hour corpus would take roughly **38 hours locally**. A GPU or the hosted backend is now a
> hard requirement, not a convenience.

### Original finding (whisper-small only — kept for the record)

Whisper `small` on these files produces largely incoherent Devanagari. A real sample:

```
[ 0.0- 9.5] अदिस गाम बाना ब्या बाखु
[ 9.5-14.7] एक लगता आँ उसकना में लाहुल
```

That is not a Hindi sentence. It's the model hallucinating plausible phonemes out of noise.

**Why:** far-field phone mic, 8–27 children talking at once, heavy reverb, clipping, and code-mixing —
every adversarial condition for ASR at the same time.

**Worst case:** the Shadow Art session (27 students) has only **20% speech density** despite being the
*loudest* file at −18.2 dB. Loud but not speech — that is continuous classroom din. Whisper's language ID
collapses there to Javanese/Indonesian (`jw:0.46, id:0.33`), which is its characteristic output for
unintelligible audio, and VAD found no speech at all in one 30-second clip.

**Best case:** Aayushi's 8-student session — 85.6% speech density, the most coherent transcripts.
Small class, quiet room. **Use this session as the demo default.**

---

## 6. What this means for the architecture

This is the finding that should shape the whole submission, and it's the "system design thinking"
they said they're grading.

**Split the pipeline into two layers by robustness:**

| Layer | Method | Robust here? | Feeds |
|---|---|---|---|
| **Acoustic** | VAD + diarization + energy | **Yes** — works on unintelligible audio | Talk-time split, turn counts, interaction rate, no-speech time |
| **Lexical** | ASR transcript | **No** — degrades badly in noisy rooms | Question counts, keyword/topic analysis |

**Put the load-bearing metrics on the acoustic layer.** Talk-time ratio and turn-taking survive when the
words don't. Then attach the transcript-derived metrics with an explicit **confidence score** driven by
speech density and ASR log-probability — so the Shadow Art session self-reports as low-confidence instead
of quietly emitting nonsense numbers.

**A dashboard that says "I can't read this room" is more useful to a teacher than one that invents a
statistic.** Building that in is the single biggest differentiator available on this assignment.

**Two specific leverage points:**

- **The recording is made on the teacher's own phone.** So the teacher is consistently the nearest,
  loudest, highest-SNR voice. Per-segment energy is therefore a strong teacher/student discriminator —
  cheap, and it works without diarization. Combine with "longest cumulative speaker = teacher".
- **`totalStudents`, `boys`, `girls` are ground truth.** They let you normalise participation per capita
  and split engagement by gender — which lands directly on MakerGhat's stated priority on girls
  (700,000+ girls trained). Almost no other candidate will do this.

**On ASR:** `small` is not enough. Options, roughly in order of effort — Whisper `medium`/`large-v3`;
AI4Bharat **IndicWhisper** or **IndicConformer**; `vasista22/whisper-hindi-medium`. Expect a real but
limited gain: no model fixes 27 children shouting at one phone. Say that plainly in the README rather
than overclaiming.

---

## 7. Environment notes

- Python 3.12.3 · ffmpeg present · faster-whisper 1.2.1 (installed `--user`; system `pip` can't write
  script shims into `C:\Python312\Scripts` without admin — **use a venv for the real project**)
- Whisper models cache to `C:\Users\hisha\.cache\huggingface` (on C:, which has ~59 GB free)
- **`D:` has only 3.7 GB free of 100 GB.** The dataset is 243 MB. Tight — keep model weights off D:.


---

## 8. Teacher/student attribution — MEASURED, and it does not work (2026-09-06)

The energy-based split assumed the phone belongs to the teacher, so she is the loudest
voice. **On this audio that assumption is false**, and it is now measured rather than
suspected.

36 of 47 labelling rows were labelled from **linguistic register** in the transcript —
imperatives, first-person organising, rules, questions put to a class. That is independent
evidence from loudness, so it is a fair cross-check (`scripts/label_from_transcript.py`).

| | |
|---|---:|
| Rows labelled | 36 of 47 |
| **Energy-based accuracy** | **41.7%** |
| Accuracy on "confident" rows (≥ 0.15) | 41.9% |
| **A constant "always teacher" baseline** | **100%** |
| Confusion | 21 teacher turns called *student*, 15 correct |

Three things follow, and each one matters more than the last.

**It is worse than chance.** A coin flip scores 50% on a binary task; this scores 41.7%.

**Confidence does not discriminate.** 41.9% on rows the model called confident versus
41.7% overall. The M6 thesis — *right whenever it is sure, so gating works* — **fails for
attribution**. The band breakdown is not reassuring either: 69% above 0.30, but only 22%
between 0.15 and 0.30, which is where four of the five sessions actually sit.

**The signal is not weak, it is absent.** On OD11163_2025-12-23 every segment fell between
−27.1 and −29.7 dB. A 2.6 dB spread is not a quiet speaker and a loud one; it is one room
recorded at one level. Clustering it produces noise with a confidence attached.

### Caveat on the labels

They are all `teacher`, which is *partly* an artefact: the phone is the teacher's, so her
speech is what transcribes legibly, and a student speaking from across the room shows up as
a row with garbled text that was left blank. So this measures **recall on teacher turns**,
not balanced accuracy.

But that caveat cannot rescue the result. Calling 21 of 36 teacher turns "student" is not a
subtle miscalibration, and the always-teacher baseline beating the model 100% to 41.7%
means the energy feature is **adding noise, not information**.

### What has to change

Not a threshold. The fix is real speaker diarization:

- **pyannote 3.1** — MIT, gated behind an HF token, runs locally. Purpose-built for this.
- **Sarvam** — bundles diarization into the same call as transcription for ₹15/hour extra;
  the whole corpus would be about ₹89. Needs credits on the account.

Until one of those is in, M1-M4 rest on a coin flip, and `config.ROLE_CONF_FLOOR` at 0.15
is letting four of five sessions through when the measurement says it should let none.


---

## 9. Attribution fixed with diarization (2026-09-06)

Replacing the feature rather than tuning the threshold. Same windows, same 36 labels,
same scorer:

| | Accuracy |
|---|---:|
| Energy k-means | 41.7% |
| **pyannote diarization** | **88.9%** |

| Session | Energy | Diarization |
|---|---:|---:|
| OD11163_2025-12-23 | 69% | **92%** |
| OD11163_2026-01-28 | 60% | **100%** |
| OD11165_2026-01-06 | 33% | **100%** |
| OD11166_2026-01-20 | 8% | **75%** |

Three sessions now reach the `usable` verdict (confidence 0.82-0.92, up from 0.53-0.71).

**Why it works where energy did not.** Energy assumed *the teacher is nearest the mic* -
false here, the phone hears one room at one level. Diarization assumes *the teacher is one
person who holds the floor while many students each speak briefly*, which survives wherever
the phone is sitting.

**Cost:** 1.24x realtime, so ~3.4 h for the full corpus. One-time and cached.

### Three bugs this surfaced, all fixed

1. **Granularity.** VAD segments run to 63 s; diarization turns are seconds. Labelling each
   segment by whoever dominated it erased every student turn - M1 reported **100% teacher
   talk** on three sessions. Segments are now split at speaker changes.
2. **Fragmentation.** Splitting produced 185 pieces in 4 minutes, median 0.68 s, with
   adjacent same-role pieces unmerged. That inflated M2 (23.2 turns/child/hour) and M3
   (9.8 switches/min) and collapsed M4 from 241 s to 16 s. Adjacent pieces now merge, short
   holes between same-speaker turns are absorbed, and the minimum turn is 0.30 s.
3. **Confidence semantics.** The airtime margin measures how *dominant* the teacher is, not
   how *reliable* the split is - one session had margin 0.06 and 75% accuracy. Now
   `coverage x (0.5 + 0.5 x margin)`.

### The 88.9% was measured against one-sided ground truth (2026-09-07)

**All 36 hand labels say `teacher`. Not one says `student`.** So a classifier that never
outputs "student" scores **100%** on that set, and the accuracy number cannot see student
recall at all. This is not a small caveat — it is why the next bug survived validation.

The sheets were generated from the *pre-split* VAD segments, which run up to 145 s and are
inevitably teacher-dominated, so there were no short student spans on the page to label.
Regenerating them from the split timeline is what would fix it, and that needs listening.

### Students were being handed to the teacher by a tie (2026-09-07)

`speaker_for_span` took the first strict maximum overlap. Diarization turns *overlap*, and
a teacher's turn routinely **encloses** a student's:

| Session | Student turns | Fully inside a teacher turn | Mislabelled |
|---|---:|---:|---:|
| OD11163_2026-01-28 | 25 | **25** | **25 (100%)** |
| OD11165_2026-01-06 | 38 | **38** | **38 (100%)** |
| OD11166_2026-01-12 | 73 | 8 | 8 |

When the span *is* the student's turn, the student turn and the enclosing teacher turn cover
it identically. A strict `>` therefore kept whichever came first in the list — always the
enclosing teacher. Every nested student turn was labelled teacher.

**What it produced:** M1 = 100% teacher talk and M3 = 0 exchanges on two sessions, while M5,
reading the same turns, reported **28 and 11 answered student questions**. Both numbers were
printed on the same card and nothing noticed.

**Fix.** Ties go to the speaker whose turn is *most contained* in the span — a student turn
matching the span scores 1.0, a teacher turn blanketing twenty seconds of it scores 0.05.
Containment is only ever a tie-break, so a student holding the floor is still not displaced
by a teacher chipping in for a second. Recovered **14.9 s and 19.0 s** of student speech on
the two sessions.

**And a cross-check, since the labels could not provide one.** `attribution_lost_students`
fires when diarization heard more than 2 s of another speaker but the timeline contains no
student time at all. Two views of one recording disagreeing about whether a child spoke is
a bug every time, and it needs no hand labels to detect.

> This also closes the question left open on 2026-09-06 — *"two sessions report no student
> turns; only listening settles it."* Listening was never needed. The turns already proved
> students spoke; the split was throwing them away.

### M5 wait time — three defects, all found by reading the real turns

Splitting makes segments tile exactly, so a student turn begins at the instant the teacher's
ends and the measured gap is always 0.00. Wait time therefore has to come from the raw
diarization turns. Moving it there was necessary but not sufficient — the turns themselves
broke three assumptions:

| # | Assumption | What the turns actually do |
|---|---|---|
| 1 | The turn after a teacher question is the answer | The teacher usually speaks again first. Walking the list in pairs kept **1 of 3** questions on `OD11166_2026-01-20` and **5 of 8** on `OD11166_2026-01-12`. |
| 2 | Turns are sequential | They overlap. Pairing by adjacency matched a long teacher turn with a student turn *nested inside it*, giving gaps as negative as **−11.0 s** — 17 of 49 were negative, and all were silently dropped. |
| 3 | A gap is a gap | **3 of 11** surviving samples were exactly `0.000`. Real wait times never land on exactly zero; that spike is two turns sharing a snapped boundary, i.e. no silence was resolved. |

The fix searches forward from each question for the first other-speaker turn starting at or
after it ends, rather than trusting adjacency. A student already talking as the question
lands counts as a genuine **zero** — that is an interruption, a real observation. An exact
`0.000` between two separate turns is discarded as unmeasured.

**And a sample floor.** The first run published `0.00 s` for one session off a **single**
question–answer pair. `WAIT_TIME_MIN_SAMPLES = 3`; below that M5 is withheld and says how
many pairs it found. Confidence scales with the count, reaching full only at twice the floor.

### Diarization is now the speech detector too (2026-09-07)

Silero was missing most of the speech on the hands-on sessions, and `classify_non_speech`
then filed it as activity noise — the rule that exists so a maker lesson does not score as a
failed one was swallowing the lesson:

| Session | Diarized speech | Landed in `speech` | Landed in `handson` |
|---|---:|---:|---:|
| OD11166_2026-01-12 — all turns | 161s | 20s | **142s** |
| — student turns only | 70s | 3s | **67s** |
| OD11166_2026-01-20 — all turns | 279s | 37s | **242s** |
| — student turns only | 118s | 9s | **109s** |

pyannote's segmentation is a speech detector before it is a speaker detector, and the
pipeline already pays for it, so where diarization runs its turns are the speech map
(`speech_spans()` — a proper interval union, since turns overlap and arrive in speaker
order). Silero stays as the fallback, and `models.vad` records which one ran.

**Effect on the two hands-on sessions:**

| | Before | After |
|---|---|---|
| OD11166_2026-01-12 | partial 0.53, TTR 93%, student 1.6s | **usable 0.78, TTR 57%, student 61.2s** |
| OD11166_2026-01-20 | partial 0.62, TTR 80%, student 6.5s | **usable 0.83, TTR 61%, student 83.7s** |

Independent cross-check: raw diarization airtime on OD11166_2026-01-12 is 91.4s teacher to
69.9s student — **56.7%**, against the 57% the pipeline now reports.

Hands-on time survives rather than collapsing (78.2s and 15.6s), so the maker distinction
still works; it just is not eating the speech any more.

### Still open

**The validation set has no student rows.** Until the sheets are regenerated from the split
timeline and re-labelled by ear, the attribution accuracy figure measures teacher recall
only. Everything about student detection currently rests on internal consistency checks,
not on ground truth.

**`speech_density` has stopped discriminating.** It is `min(1, speech_share / 0.5)`,
calibrated when Silero was finding ~58% of the speech. pyannote finds 60-83%, so the
component now reads **1.00 on every session** and 30% of the confidence weight is a
constant. The overall score still varies (0.78-0.92) on the other three components, and the
verdict changes it produced are right — a recording that genuinely is 60% speech should beat
Silero's reading of 9%. But the 0.5 knee wants recalibrating, and there is no ground truth to
fit it to, so it is written down rather than quietly re-tuned.

*(The earlier note here — "VAD misses 8.7 s of student speech on OD11165" — is resolved by
the change above; that was Silero, and it is no longer the detector.)*
