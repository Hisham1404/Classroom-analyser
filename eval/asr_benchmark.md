# ASR benchmark — measured WER

`google/fleurs` · 20 samples · reference transcripts written by humans.

> **This is clean, read, single-speaker prose on good microphones.** It measures whether a model can transcribe Hindi at all. It does **not** predict performance on our recordings — 8–27 children near one phone in a classroom is a different problem. Use it to sanity-check the models and this harness, not to pick a winner for the actual corpus.

| # | Contender | Model | Scored | **WER** | CER | Failed | Sec/sample |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | **indic** | `OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` | 20 | **17.3%** | 6.8% | 0 | 3.0 |
| 2 | **groq** | `whisper-large-v3` | 20 | **26.9%** | 10.7% | 0 | 1.2 |
| 3 | **baseline** | `Systran/faster-whisper-small` | 20 | **63.8%** | 39.4% | 0 | 10.6 |
| 4 | **primary** | `digikar/vasista22-whisper-hindi-large-v2-ct2-int8` | 20 | **84.4%** | 74.9% | 0 | 95.7 |

## Sample outputs

### indic — best sample (WER 0.0%)

- **reference:** हर कोई समाज से जुड़ा होता है और ट्रांसपोर्ट सिस्टम का उपयोग करता है. लगभग सभी लोग ट्रांसपोर्ट सिस्टम के बारे में शिकायत करते हैं.
- **hypothesis:** हर कोई समाज से जुड़ा होता है और ट्रांसपोर्ट सिस्टम का उपयोग करता है लगभग सभी लोग ट्रांसपोर्ट सिस्टम के बारे में शिकायत करते हैं

### groq — best sample (WER 4.0%)

- **reference:** हर कोई समाज से जुड़ा होता है और ट्रांसपोर्ट सिस्टम का उपयोग करता है. लगभग सभी लोग ट्रांसपोर्ट सिस्टम के बारे में शिकायत करते हैं.
- **hypothesis:** हर कोई समाज से जुड़ा होता है और ट्रांसपोर्ट सिस्टम का उपयोग करता है लगबग सभी लोग ट्रांसपोर्ट सिस्टम के बारे में शिकायत करते हैं

### baseline — best sample (WER 35.0%)

- **reference:** इसे केमिकल का pH कहा जाता है. आप लाल गोभी के जूस का इस्तेमाल करके एक संकेतक बना सकते हैं.
- **hypothesis:** इसे केमिकल का पीज कहा जाता है, आप लाल गोवी के जूस को अस्तमाल कर के एक संकेत बना सकते हैं।

### primary — best sample (WER 5.6%)

- **reference:** कैल्शियम और पोटैशियम जैसे तत्वों को धातु माना जाता है। बेशक, चांदी और सोने जैसी धातुएं भी हैं।
- **hypothesis:** कैल्शियम और पोटेशियम जैसे तत्वों को धातु माना जाता है, बेशक चांदी और सोने जैसी धातुएं भी हैं.
