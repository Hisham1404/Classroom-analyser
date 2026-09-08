# ASR bake-off

Same clips, same preprocessing, every contender.

> **These are proxy signals, not ground truth.** There is no human reference transcript for this audio, so nothing here is a word error rate. The proxies reliably catch *garbage* — looping, script drift, confident text over silence — which is enough to shortlist models before hand-scoring. The verdict column in `asr_handscoring.md` is the one that decides.

## Ranking

| # | Contender | Backend | Model | Clips | Composite | Agreement | Mean logprob | Looping | Devanagari | Sec/clip |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **indic** | indic | `OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` | 3 | 0.887 | 0.122 | -0.43 | 0.00 | 1.00 | 0.0 |
| 2 | **primary** | local | `digikar/vasista22-whisper-hindi-large-v2-ct2-int8` | 3 | 0.759 | 0.079 | -0.41 | 0.00 | 1.00 | 0.0 |
| 3 | **groq** | groq | `whisper-large-v3` | 3 | 0.728 | 0.087 | -0.86 | 0.01 | 1.00 | 0.0 |
| 4 | **baseline** | local | `Systran/faster-whisper-small` | 3 | 0.533 | 0.042 | -1.32 | 0.00 | 0.64 | 0.0 |

**Agreement** is mean word overlap with the other contenders on the same clip. A contender that agrees with nobody is inventing - that is the only signal that catches fluent-looking nonsense. **Composite** weights model confidence (0.40), speech coverage (0.20), absence of looping (0.25) and staying in Devanagari (0.15).
**Looping** is the share of repeated 4-grams. **Compression ratio above 2.4** is Whisper's own hallucination alarm.

## Transcripts

### `OD11163_2025-12-23-121239_c00`
_33.6–34.1 min_

| Contender | Lang | Composite | Logprob | Compression | Text |
|---|---|---:|---:|---:|---|
| indic | hi | 0.943 | -0.43 | 2.72 ⚠️ | और ओबल साइक में क्या होगा लड़के होगा या फिर दद होगा ठीक है तो कौन लेगा क्लोज कन लेगा ओब क्लोज तो इस रू क्लोज सर्क है न और आपका स तो आप ऐसे रूम में बैठो या फिर ऐ… |
| primary | hi | 0.838 | -0.53 | 0.43 | ठीक है |
| baseline | hi | 0.783 | -1.47 | 1.94 | पन्त हो अदी होटारी यी है दो थी क्योचा खीचूच के चट्या और याद का टोगा साद्टाटी तो आप पनिखे ख़े खियोटेग साट्टिक है तो तो रगगग्ग बरागे नहीं |
| groq | hi | 0.466 | -0.51 | 2.15 | सब्सक्राइब करो कुछ भी करो, जो लगे की ये ओपने साकिक है आपको ऐसे रूप में खड़ा होना है, बैठना होना है कुछ भी ऐसे कुछ अक्रिमिटिक्स करो काकी लगे की ये साकिक पुरा ग्ल… |

### `OD11163_2026-01-28-121933_c00`
_10.8–11.3 min_

| Contender | Lang | Composite | Logprob | Compression | Text |
|---|---|---:|---:|---:|---|
| primary | hi | 0.982 | -0.10 | 2.63 ⚠️ | मतलब आसमान से बादल भी जा रहा था जा रहा था इतना भारी वर्षा हो रहा था भारी वर्षा पता है ना बहुत बारिश हो रहा था बहुत ज्यादा और अचानक से ये तो गांव के बीच में जो र… |
| baseline | hi | 0.815 | -1.16 | 2.01 | प्रड़ भी न्दीखिई थाद़ाग। लिए भार कल दिलकों आतुगाठ। प्रड़, वारी वर्सन् कता लगा् chair, खाहना ब्रैसिग। तो लष्त का उस में लदी आंके पहरешьे एं लदी आंके वू़ थास्ते क… |
| groq | hi | 0.787 | -1.60 | 1.97 | जरूरत आमकुटबीदेशंय ना बूसे लगा था ? दुषाब से व्यक्त सुखी उठा रहे थे और ऐसा गौव बूनिक मद आउट्या उत्पास पहार टक लंकुं पहार की, जी आप मासी को संगीव नहीं आता और एक… |
| indic | hi | 0.775 | — | 2.36 | आसमानऐ बादल भी जा रहा था जा रहा था इतना भारी बर्सन हो रहा था भारी बर्सत ब पवर प्राइस़ोड़ा बहुत ज़्यादा और अचानक से ये दो गाँव के बीच में ो उसने एक नदी आके पहाड़… |

### `OD11165_2026-01-06-114155_c00`
_31.9–32.4 min_

| Contender | Lang | Composite | Logprob | Compression | Text |
|---|---|---:|---:|---:|---|
| indic | hi | 0.943 | -0.43 | 2.46 ⚠️ | ग्रु ग्रुप ऑ ग्रुप टू ग्रुप ग न ग्रु बे थोड़ा अच्छे क्रिएटर न बोऐ बहुत अच्छे अच्छे नामठीक है चलो ग्रुप ने ग्रुप टूीक है ग्रुप थ्री पहले जागाीक हैठीक हैना आप ब च… |
| groq | hi | 0.932 | -0.47 | 2.39 | प्रॉब्लम का विशेष एंट्रूप प्रॉब्लम प्रॉब्लम कुम अच्छी क्रिएटिव नाम आएंगे कि मेरे वजह कहीं और से बहुत अच्छे नाम आज ठीक है चलो रूप पर रूप टू अब जय है अब अब ठीक है… |
| primary | hi | 0.457 | -0.60 | 1.99 | ग्रुप में ग्रुप से ग्रुप बन गई कैसे करना बदला इसके अलावा प्रदेश के अन्य जिलों में भी इस अभियान के तहत जागरूकता कार्यक्रम आयोजित किये जा रहे हैं |
| baseline | hi | 0.000 | — | 0.00 | _(no speech detected)_ |
