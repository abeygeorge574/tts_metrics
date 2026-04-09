# wav2vec2 vs Whisper — Experiment Notes

## What we tested
Ran both wav2vec2-base-960h and Whisper (MLX medium) on the same 7 TTS models × 5 samples.
Compared WER and agreement between the two transcribers.

## What wav2vec2 was doing wrong

wav2vec2-base-960h is a small CTC-based model (95M params) with no language model.
It decodes characters greedily with no contextual recovery.

### Specific failures on clean TTS audio
- gtts sample_2: wav2vec2 WER=0.21, Whisper WER=0.0 — clean sentence Whisper got perfectly
- fastspeech2 sample_2: wav2vec2 WER=0.57, Whisper WER=0.0 — completely botched
- mms: 0/5 agreement — wav2vec2 found errors on every sample Whisper passed

### Agreement summary
| Model | Whisper avg WER | wav2vec2 avg WER | Agreement |
|-------|----------------|-----------------|-----------|
| kokoro | 0.0 | 0.029 | 4/5 |
| edge_tts_ava/andrew | 0.0 | ~0.05 | partial |
| gtts | 0.0 | 0.118 | 2/5 |
| fastspeech2 | 0.0 | 0.128 | 3/5 |
| speecht5_hifigan | 0.0 | 0.083 | 2/5 |
| mms | 0.011 | 0.118 | 0/5 |
| f5tts | 0.112 | 0.126 | 4/5 |
| speecht5_griffinlim | 1.0 | 1.0 | 5/5 |

The two models only agreed on clearly good audio (kokoro) and clearly broken audio
(speecht5_griffinlim). In the middle ground, wav2vec2 produced false positives.

## Root cause
No language model → can't recover from acoustic ambiguity in natural TTS prosody.
Whisper has a strong language model that fills in context.

## Conclusion
**Whisper is the correct choice** for this pipeline. wav2vec2 generates false WER failures
on perfectly acceptable TTS output. It would cause good models to fail the WER gate
unnecessarily.

The one advantage of wav2vec2 — it caught mms more aggressively (0/5 agreement vs
Whisper's lenient 5/5 pass) — but listening confirms mms is intelligible, so Whisper
is correct and wav2vec2 is being too strict.
