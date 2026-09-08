# ASR options for Hindi classroom audio — the full landscape

Compiled 2026-09-05. Corpus for all cost maths: **252 min (4.2 h)** of real audio.

Three of these are **measured on our own recordings**; the rest are documented claims and are
marked as such. Never quote an unmeasured number as a result.

---

## 1. What we have actually measured

Three contenders, three identical 30-second clips, three different teachers.
Full output in [`eval/asr_bakeoff.md`](../eval/asr_bakeoff.md).

| | Composite | Mean logprob | Devanagari | Sec / 30 s clip | Corpus (4.2 h) | Cost |
|---|---:|---:|---:|---:|---:|---|
| **`ai4bharat-indicconformer-hi`** (ONNX int8, local) | **0.954** | −0.43 | 1.00 | **7.9** | **67 min** | ₹0 |
| `vasista22-whisper-hindi-large-v2` (ct2-int8, local) | 0.759 | −0.41 | 1.00 | 270 | 38 h | ₹0 |
| Groq `whisper-large-v3` (hosted) | 0.728 | −0.86 | 1.00 | 3 | ~7 min | ₹0 |
| `faster-whisper-small` (local) | 0.533 | −1.32 | 0.64 | 42 | 6 h | ₹0 |

*(Composites include the artefact penalty from §5.)*

## THE ANSWER: AI4Bharat IndicConformer, run as ONNX

`OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` via the `onnx-asr` library wins on every axis
that matters here:

- **Best quality measured** — 0.954 composite, ahead of both Whisper variants.
- **3.78× faster than realtime on this CPU.** The whole corpus in **67 minutes, locally.**
  No GPU, no Colab, no API key, no quota, no overnight run.
- **No torch.** It runs on the onnxruntime faster-whisper already installs — critical on a
  drive with 3.2 GB free. Install cost was a few MB.
- **CTC/RNNT, so it does not hallucinate fluently.** No `सब्सक्राइब`, no news bulletins.

It also settled the two disputed clips. On the clip where Groq hallucinated "subscribe" and the
Hindi Whisper returned only `ठीक है`, IndicConformer found the actual lesson: **`क्लोज सर्क`** —
*closed circuit* — children standing and sitting to model an electrical circuit. And on the flood
clip it read `बाढ आके` (*a flood came*) where the Hindi Whisper had guessed `एक बार आगे`.

**It is now the recommended primary backend.**

### But the composite is not the whole story — read the transcripts

Hand-inspecting the three clips shows something the proxies could not catch.
**Both good models hallucinate, in different and characteristic ways.**

| Clip | Winner by eye | What happened |
|---|---|---|
| c00 (Santana, Dec) | **Hindi model** | It returned just `ठीक है` ("okay"). Groq returned 172 characters beginning **`सब्सक्राइब करो`** — "subscribe" — Whisper's notorious YouTube-caption hallucination. Staying quiet was the more honest answer. |
| c00 (Santana, Jan) | **Hindi model, clearly** | Coherent narrative about a flood reaching a village (logprob −0.10). Groq garbled the same audio (−1.60) — though both independently produced `और एक बार आगे पूरा चारों … पानी`, so they agree on the ending. |
| c00 (Aayushi) | **Groq, arguably** | Groq gave `प्रॉब्लम … क्रिएटिव नाम … रूप टू … रूप थ्री` — garbled, but a teacher organising groups, which fits a maker session. The Hindi model drifted into *news-broadcast* Hindi: "awareness programmes are being organised in other districts of the state under this campaign." No classroom says that — it is bleed-through from its news-corpus fine-tuning. |

**The lesson:** each model hallucinates toward its own training data. Whisper drifts to YouTube
captions; the Hindi fine-tune drifts to news bulletins. Both are fluent, low-repetition Devanagari,
so the original proxy signals scored them as good. **That gap is now closed — see §5.**

### Per-clip scores after the artefact penalty

| Clip | primary | groq | baseline | Agreement | Automatic verdict |
|---|---:|---:|---:|---:|---|
| c00 Santana Dec | **0.838** | 0.466 `subscribe` | 0.783 | 0.037 | primary |
| c00 Santana Jan | **0.982** | 0.786 | 0.815 | 0.115 | primary |
| c00 Aayushi | 0.457 `news_bulletin` | **0.932** | 0.000 | 0.019 | groq |

**The automatic scoring now picks the same winner as reading the transcripts by hand, on all
three clips.** That is the strongest evidence we have that the proxies are measuring something real.

**Cross-model agreement is 0.02–0.12** — two independent models barely share a word on this audio.
Read that as a warning: whichever model wins, individual transcripts are not trustworthy yet.
It is also a useful per-clip confidence input for M6.

---

## 2. Hosted APIs with a usable free tier

Costs are for one full pass over the 252-minute corpus.

| Provider | Model | Free allowance | Corpus cost after free | Hindi | Verified? |
|---|---|---|---|---|:-:|
| **Groq** | `whisper-large-v3` | 28.8 K audio-sec/day ≈ **8 h/day** | **₹0** — whole corpus fits in one day | multilingual | ✅ measured |
| **Bhashini** (MeitY, Govt of India) | IIT-contributed models | **free, but "PoC only"** | ₹0 | **native, 22 languages** | ❌ |
| **Sarvam AI** | Saaras v3 / Saarika | ₹100 credits ≈ 3.3 h | ₹126 total → **₹26** | **native + code-mixed** | ❌ |
| **Speechmatics** | — | **480 min/month** | ₹0 — corpus fits twice over | yes | ❌ |
| **Deepgram** | Nova-3 | $200 credit | ~$1.08, inside credit | yes | ❌ |
| **AssemblyAI** | Universal-3.5 | credits on signup | ~$0.63 | 99 languages | ❌ |
| **Google Cloud STT** | Chirp | 60 min/month + $300/90 days | ~$12.3, inside credit | yes | ❌ |
| **Azure Speech** | — | 5 h/month | ₹0 | yes | ❌ |
| **AWS Transcribe** | — | 60 min/month × 12 months | mostly paid | yes | ❌ |
| **ElevenLabs** | Scribe v2 | limited | $0.22/h ≈ $0.92 | 90+ languages | ❌ |
| **OpenAI** | `gpt-4o-transcribe` | none | paid | yes | ❌ |

### The two worth taking seriously

**Sarvam AI** — the strongest untested candidate, for three reasons that matter here:

- Trained on **1M+ hours of real Indian audio**, not Western speech with Hindi bolted on.
- Explicit **code-mixing** support (Hinglish) — our audio is Hindi–Marathi–English.
- **Speaker diarization built in** at ₹45/h instead of ₹30/h. That is *Phase 3's entire problem*
  solved for ₹15 an hour.

Whole corpus with diarization: 4.2 h × ₹45 = **₹189**, of which ₹100 is free credit.
**₹89 out of pocket** to get transcription and speaker separation in one call.

**Bhashini** — the best *narrative* fit. It is the Government of India's national language platform
under MeitY, free, and covers all 22 scheduled languages. MakerGhat is a NITI Aayog Atal Tinkering
Labs curriculum partner, so building on India's public language infrastructure is squarely on-brief
and worth a paragraph in the README either way.

Two caveats: the docs say API use is **"for PoC only"** with production requiring a paid plan
(a pre-work assignment is genuinely PoC, so this is fine), and access is not self-serve — you go
through a pipeline-search → config → compute flow and the docs point you at contacting their team.
**Budget half a day for onboarding, not ten minutes.**

---

## 3. Open weights — local or Colab

| Model | Format | Size | Hindi-tuned | Notes |
|---|---|---:|:-:|---|
| `Systran/faster-whisper-small` | CT2 | 0.5 GB | ✗ | measured: not usable here |
| `Systran/faster-whisper-medium` | CT2 | 1.5 GB | ✗ | untested |
| `Systran/faster-whisper-large-v3` | CT2 | 3.1 GB | ✗ | ~8 min per 30 s clip locally — Groq runs the same weights in 3 s |
| **`digikar/vasista22-whisper-hindi-large-v2-ct2-int8`** | CT2 | 1.6 GB | ✅ | **measured best quality**, 9× slower than realtime |
| `vasista22/whisper-hindi-{small,medium,large-v2}` | transformers | 1–5.7 GB | ✅ | needs CT2 conversion or torch |
| `ai4bharat/indic-conformer-600m-multilingual` | NeMo/transformers | 2.4 GB | ✅ 22 langs | needs torch — blocked on disk |
| **`OpenVoiceOS/ai4bharat-indicconformer-hi-onnx`** | **ONNX** | ~0.6 GB | ✅ | **measured winner** — same AI4Bharat model, ONNX export, **no torch**. Siblings exist for mr/ta/te/ml/bn/kn/ur/or/ne. |
| `ai4bharat/indicwhisper` | transformers | varies | ✅ | Vistaar benchmark leader for Hindi |
| `openai/whisper-large-v3-turbo` | transformers | 1.6 GB | ✗ | ~8× faster than large-v3, small accuracy cost |

**The constraint that decides this:** CTranslate2 models run on `faster-whisper` with no torch.
Anything else needs ~1 GB of torch installed, and the venv lives on a drive with 3.2 GB free.
Moving the venv to C: would unblock `indic-conformer` and `indicwhisper` if we decide they matter.

---

## 4. Recommendation

**Two backends, one for speed and one for quality, already switchable in one line.**

| Use | Backend | Why |
|---|---|---|
| **Primary — batch and demo** | **`indic`** IndicConformer ONNX | Best measured quality (0.954), 3.78× realtime, 67 min for the corpus, no torch, no network, no quota. Offline-first for real. |
| Hallucination cross-check | **`groq`** `whisper-large-v3` | Free and instant. A different architecture, so disagreement is a signal. |
| Quality reference | **`local`** Hindi CT2 | Best Whisper-family Hindi, but 38 h for the corpus. Spot checks only. |
| **Next to test** | **Sarvam Saaras** | Purpose-built for Indian code-mixed audio *and* brings diarization, which Phase 3 needs. ₹89 for the whole corpus. |
| Worth a mention regardless | **Bhashini** | Free, government, on-brief. Test if onboarding is quick; cite it in the README either way. |

Deliberately **not** pursuing: Google STT, AWS, Azure, OpenAI — all cost real money for a result we
already get free, and none is tuned for Indian classroom audio. Speechmatics and Deepgram have
generous free tiers but no Indian-language specialisation; they are fallbacks, not contenders.

---

## 5. The scoring gap — found by this run, now closed

The original proxies caught **looping, script drift and low confidence**. They did **not** catch a
fluent, confident, well-formed hallucination — exactly what both good models produced on one clip each.

Two additions, both implemented and tested:

1. **`artefact_hits()`** — a list of known invented-text tells: `सब्सक्राइब` (subscribe),
   `देखने के लिए धन्यवाद` (thanks for watching), and news-bulletin boilerplate
   (`प्रदेश के अन्य जिलों`, `इस अभियान के तहत`, `जागरूकता कार्यक्रम`). A hit **halves** the
   composite, because a model being confident about something it invented is evidence against it,
   not for it.
2. **`cross_model_agreement()`** — mean pairwise word overlap between independent transcripts.
   Where two models sharing no weights produce the same phrase (both gave
   `और एक बार आगे पूरा चारों … पानी`) it is very likely real.

**Caveat worth keeping:** the artefact list is hand-curated from three clips. It will miss tells we
have not seen yet, and it could in principle fire on a genuine sentence — a teacher really could say
`समाचार`. Treat it as a strong hint, not a verdict. The hand-scoring sheet is still the ground truth.

---

## Sources

[Sarvam pricing](https://docs.sarvam.ai/api-reference-docs/pricing) ·
[Sarvam Saarika model](https://docs.sarvam.ai/api/getting-started/models/saarika) ·
[Sarvam STT](https://www.sarvam.ai/speech-to-text) ·
[Bhashini APIs](https://bhashini.gitbook.io/bhashini-apis) ·
[Bhashini](https://www.bhashini.ai/) ·
[ULCA](https://github.com/bhashini-dibd/ulca) ·
[Google Cloud STT pricing](https://diyai.io/ai-tools/speech-to-text/google-cloud-speech-to-text-pricing/) ·
[Free STT APIs compared (AssemblyAI)](https://www.assemblyai.com/blog/the-top-free-speech-to-text-apis-and-open-source-engines) ·
[STT benchmarks 2026](https://futureagi.com/blog/speech-to-text-apis-in-2026-benchmarks-pricing-developer-s-decision-guide/) ·
[AI4Bharat IndicWhisper](https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicWhisper) ·
[Vistaar benchmark](https://github.com/AI4Bharat/vistaar) ·
[digikar CT2 Hindi build](https://huggingface.co/digikar/vasista22-whisper-hindi-large-v2-ct2-int8)


---

## 6. A warning for Phase 3: Silero VAD is too aggressive on this audio

Wiring up IndicConformer surfaced something that matters well beyond ASR.

`onnx-asr`'s Silero VAD returned **zero speech segments on 2 of 3 real classroom clips** — clips
where the very same model, run without VAD, produced 308 and 167 characters of correct speech.

The backend now falls back to no-VAD when VAD rejects a whole clip, so nothing is lost. But the
architecture in `architecture.md` puts **VAD at the base of the acoustic layer**, feeding M1–M4 —
the metrics chosen precisely because they were supposed to be the robust ones.

### Resolved — it is onnx-asr's VAD, not Silero, and not the audio

Swept faster-whisper's Silero VAD over the same three clips (`scripts/tune_vad.py`):

| threshold | min_speech | min_silence | Santana Dec | Santana Jan | Aayushi | mean |
|---:|---:|---:|---:|---:|---:|---:|
| **0.50** | 250 ms | **2000 ms** *(fw default)* | 29.7% | 100% | 45.9% | **58.5%** |
| 0.50 | 250 ms | 400 ms *(Phase 0)* | 25.7% | 100% | 38.7% | 54.8% |
| 0.35 | 200 ms | 300 ms | 84.5% | 100% | 68.7% | 84.4% |
| 0.25 | 150 ms | 250 ms | 96.8% | 100% | 76.7% | 91.2% |
| 0.15 | 100 ms | 150 ms | 100% | 100% | 98.0% | 99.3% |

**faster-whisper's Silero finds speech on every clip at every setting**, and its defaults land at
coverage consistent with the 20–86% per-session density measured in Phase 0. Below ~0.25 it starts
calling classroom din speech, which would inflate every talk-time metric.

**Conclusion: use faster-whisper's Silero VAD (already installed) at threshold 0.50 for the acoustic
layer, and treat onnx-asr's built-in VAD as ASR-segmentation only.** The acoustic layer is safe;
the bug was confined to one library's wrapper. The `indic` backend's no-VAD fallback stays as a guard.


---

## 7. The torch-only models (explored 2026-09-05)

torch CPU-only + transformers installed (venv 1.1 GB). **The 3.2 GB disk figure I had been
designing around was stale — `D:` actually has ~31 GB free**, so the "blocked on disk" verdict
on several models in §3 was wrong. Reprising them with `scripts/explore_torch_models.py`:

| Model | Composite | Speed | Verdict |
|---|---:|---:|---|
| **`ai4bharat-indicconformer-hi-onnx`** *(no torch)* | **0.954** | 3.78× | **still the winner** |
| `Oriserve/Whisper-Hindi2Hinglish-Swift` | 0.768\* | 1.08× | usable, but loops catastrophically on the hard clip |
| `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` | 0.333\* | 13.4× | fastest by far, but empty on 2 of 3 clips |
| `ai4bharat/indicwav2vec-hindi` | — | — | **gated repo (401)** |
| `ai4bharat/indic-conformer-600m-multilingual` | — | — | **gated repo (401)** |
| `ARTPARK-IISc/whisper-large-v3-vaani-hindi` | 0.937\* | 0.10x | **best quality by eye**, but 41 h for the corpus |
| `Oriserve/Whisper-Hindi2Hinglish-Prime` | 0.839\* | 0.37x | clearest romanised output, 11 h for the corpus |
| `theainerd/Wav2Vec2-large-xlsr-hindi` | **1.000\*** | 3.80x | **pure gibberish that scored a perfect 1.000 — see below** |

**\* These scores are inflated and not directly comparable to the bake-off numbers.** The explorer
does not extract per-token logprobs, so every torch model gets `mean_logprob = None`, which the
composite treats as *full confidence* (0.40 of the score, free). IndicConformer's 0.954 was earned
with a real −0.43 logprob. The true gap is wider than the table suggests.

### The gating finding

Both AI4Bharat repos require an account, accepted terms and a token. **This is precisely why the
ONNX route mattered:** `OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` is a community re-export of
the same weights and is *not* gated. The winner came in through the side door.

### Cross-model confirmation is now three deep

On the clip where Groq emitted `सब्सक्राइब`, three independent architectures agree on the content:

| Model | Output |
|---|---|
| IndicConformer (CTC) | `क्लोज सर्क है न … आप ऐसे रूम में बैठो या फिर ऐसे खड़े होना` |
| Groq `whisper-large-v3` | `ओपने साकिक … आपको ऐसे रूप में खड़ा होना है, बैठना होना है` |
| Hindi2Hinglish-Swift | `open circle? Close sir. … To aap aise hi gho` |

Different architectures, different corpora, same lesson: **open and closed circuits, with children
standing and sitting to model one.** That confirms Groq's `सब्सक्राइब` was a hallucinated *preamble*
on top of real speech, not a wholesale invention — and it is exactly the cross-model agreement
signal from §5 doing its job.

### Two limitations this exposed in the composite

1. **Romanised output is punished as script drift.** `devanagari_ratio` docks Hindi2Hinglish 15%
   for writing Latin *by design*. The signal should key off the model's declared output script.
2. **A model that reports no confidence gets 0.40 for free.** `mean_logprob = None` currently means
   "assume certain". It should mean "unknown" and score mid-range, not full marks.

Neither changes the ranking here, but both would matter if a future contender exploited them.


---

## 8. The metric failed, and how it was fixed

`theainerd/Wav2Vec2-large-xlsr-hindi` — the most-downloaded Hindi ASR model on the Hub at
1.6 M downloads — produced this:

```
रबअबपूके सारे टिया  ह टो बड़ड़िययों रया रें टसन हें ग्या गेाहयीहैतु पुवान ेने ो सररी कोल नेरा
```

Devanagari-shaped noise. Not words. It scored a **perfect 1.000**.

Every signal was satisfied: pure Devanagari (no script drift), no looping, no known artefacts,
healthy speech coverage — and **no reported confidence at all**, which the composite was treating
as *certainty*. A model that says nothing about how sure it is was getting 40% of the score free.

### Two fixes, both test-covered

1. **`mean_logprob = None` now scores 0.5, not 1.0.** Unknown means unknown.
2. **Cross-model agreement is computed per contender and reported in the ranking.**

The second is the one that actually works. After the first fix, gibberish and good Hindi *still*
score identically (0.800 each) when neither reports a logprob — because no reference-free text
statistic can tell them apart. Agreement can:

| | vs vaani | vs indic |
|---|---:|---:|
| vaani ↔ indic | — | **0.440** |
| **xlsr** | **0.000** | **0.000** |

Two models sharing no weights that produce the same words are almost certainly right.
A model agreeing with nobody is the one inventing.

### A bug inside the fix

The first implementation passed the whole set to `cross_model_agreement`, which returns the same
all-pairs mean for every row — every contender scored an identical 0.083, hiding precisely the
odd-one-out it existed to expose. Only a **three**-contender test caught it; with two contenders the
value is symmetric and the bug is invisible. Now scored pairwise per row:

| Contender | Composite | Agreement |
|---|---:|---:|
| **indic** | 0.887 | **0.122** |
| groq | 0.728 | 0.087 |
| primary | 0.759 | 0.079 |
| baseline | 0.533 | 0.042 |

### What this means for the submission

**The composite is a garbage detector, not a quality meter.** It catches looping, script drift, low
confidence and known artefacts. It cannot catch well-formed nonsense. That limitation belongs in the
README — and it is a direct argument for the M6 confidence gate: a system that scores its own output
has to be able to be wrong about it, and say so.

### On `vaani-large-v3`

By eye it produced the cleanest transcript of anything tested — `और ओपन साइकिल में क्या होता है …
तो कौन लेगा क्लोज़ साइकिल` — the clearest reading of the circuit lesson we have. But at 294 s per
30-second clip (0.10x realtime) the corpus would take **41 hours**.

It is a whisper-large-v3 fine-tune, so converting it to CTranslate2 int8 would give roughly an
8–10x speedup — around 4–5 hours. Still four times slower than IndicConformer's 66 minutes, for a
quality gain that is visible but unquantified. **Worth revisiting only if hand-scoring shows
IndicConformer is not good enough.**


---

## 9. Measured WER — the first objective number

`google/fleurs` `hi_in` test, 20 samples, human reference transcripts, every model fed
byte-identical 16 kHz mono audio. Harness: `src/evaluation/benchmark.py`.

| # | Model | **WER** | CER | Sec/sample |
|---|---|---:|---:|---:|
| **1** | **`ai4bharat-indicconformer-hi-onnx`** | **17.3%** | — | 3.0 |
| 2 | Groq `whisper-large-v3` | 26.9% | — | 1.2 |
| 3 | `faster-whisper-small` | 63.8% | — | 10.6 |
| 4 | `vasista22-whisper-hindi-large-v2-ct2-int8` | **84.4%** | — | 95.7 |

**IndicConformer wins on real WER too**, and by a wide margin. Every proxy we built —
composite, agreement-with-the-field, hand reading — pointed the same way. That convergence
is the strongest evidence any of it is measuring something real.

### The 84.4% is not what it looks like, and it validated the harness

`vasista22-whisper-hindi-large-v2` publishes **6.8% WER on FLEURS**. Measuring 84.4% should
mean our harness is broken. It is not — reading the output settles it:

> **reference:** कुछ अणुओं में अस्थिर केंद्रक होता है, जिसका मतलब यह है कि उनमें थोड़े या बिना किसी झटके से टूटने की प्रवृत्ति होती है
>
> **hypothesis:** कुछ अड़गों में अस्थिर केंद्रक होता है जिसका मतलब यह है कि उनमें थोड़े या बिना किसी झटके से टूटने की प्रवृत्ति होती है **इसके अलावा उन्होंने अपने प्रधानमंत्री नरेंद्र मोदी के नेतृत्व में भी इस…**

It transcribes the sentence **near-perfectly** — better than IndicConformer on this sample —
then keeps generating news prose after the audio ends. WER is 138% because of insertions,
not errors.

Its **best sample scored 6% against the published 6.8%**, and its median is 84%. So:

- **The harness is correct.** It reproduces the published figure when the model behaves.
- **The model is accurate and cannot stop talking.** That is a different defect from
  inaccuracy and needs a different fix.
- **The news-bulletin hallucination is confirmed on a second, independent dataset.** We first
  saw it on our own Aayushi clip; FLEURS reproduces it with `प्रधानमंत्री नरेंद्र मोदी`.

### New signal: `words_per_second`

The keyword list could never catch this — the invented text is unbounded and unpredictable.
The general form is that **the output is longer than the audio could physically contain.**
Hindi runs 2–3 words/second; anything past 4.0 is a model generating past the end of the
speech, and the composite is now scaled down in proportion.

That is three independent hallucination detectors now, each catching a class the others miss:

| Signal | Catches |
|---|---|
| `repetition_rate` | looping — the same phrase over and over |
| `artefact_hits` | known invented text (`सब्सक्राइब`, news boilerplate) |
| **`words_per_second`** | **rambling past the end of the audio** |
| `cross_model_agreement` | fluent nonsense that agrees with nobody |

### The caveat that still stands

FLEURS is clean, read, single-speaker news prose. It proves these models can transcribe
Hindi and that our measurement is sound. **It does not predict performance on 8–27 children
near one phone.** The hand-scoring sheet on our own clips remains the deciding evidence.
