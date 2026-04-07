# UTMOS vs NISQA — Development Notes

## Context

Both UTMOS and NISQA produce MOS-like naturalness scores. The question was whether both are
needed in the pipeline, or whether one can be dropped without losing coverage.

---

## What each model does

### UTMOS (UTokyo-SaruLab MOS)
- Architecture: wav2vec2 fine-tuned on TTS-specific MOS listening test data (VoiceMOS Challenge)
- Training domain: synthetic TTS outputs rated by human listeners in controlled MOS tests
- Output: single MOS score (1–5 scale)
- Strength: calibrated specifically to *synthesized* speech naturalness; listeners were rating TTS
- Weakness: single scalar — no subscores, no diagnostic signal
- Threshold in use: `UTMOS_THRESHOLD = 3.0`
- Env: requires Python 3.9 (`utmos` conda env), runs as subprocess

### NISQA (Non-Intrusive Speech Quality Assessment)
- Architecture: transformer fine-tuned on ITU-T/telephony degradation datasets
- Training domain: telephony degradation (codec noise, background babble, packet loss, reverberation)
- Output (NO_REF mode): MOS + 4 subscores — Noisiness, Discontinuity, Coloration, Loudness
- Output (WITH_REF mode): same + delta comparisons against reference
- Strength: 5 interpretable dimensions; Discontinuity and Coloration sometimes catch artifacts
  that UTMOS misses
- Weakness: not trained on TTS vocoder artifacts — Noisiness dimension misses synthetic buzz
- Thresholds: `NISQA_THRESHOLDS` (absolute) and `NISQA_DELTA_THRESHOLDS` (ref-relative) in config.py
- Env: base Python 3.13, runs as direct import

---

## Test results — all models (NO_REF mode)

Measured on the standard 5-sample test set.

| Model | UTMOS | NISQA MOS | NISQA Noisiness | Gate verdict (ground truth) |
|-------|-------|-----------|------------------|-----------------------------|
| gtts | ~3.8 | ~4.2 | ~4.0 | PASS ✓ |
| kokoro | ~4.3 | ~4.5 | ~4.3 | PASS ✓ |
| samantha | ~3.6 | ~3.8 | ~3.9 | PASS ✓ |
| kokoro_v1 | ~4.1 | ~4.3 | ~4.1 | PASS ✓ |
| parler_mini | ~3.8 | ~4.0 | ~3.9 | PASS ✓ |
| edge_tts_andrew | ~4.2 | ~4.4 | ~4.3 | PASS ✓ |
| edge_tts_ava | ~4.3 | ~4.5 | ~4.4 | PASS ✓ |
| fastspeech2 | ~3.5 | 4.2–4.6 | 3.8–4.1 | **FAIL** (missed by both) |
| mms | ~3.2 | 4.5–4.8 | 4.2–4.6 | **FAIL** (missed by both) |
| f5tts | ~4.0 | 4.1–5.0 | 3.9–4.5 | **FAIL** (missed by both) |
| speecht5_hifigan | ~2.1 | 1.9–3.5 | 2.3–3.4 | FAIL ✓ (caught by both) |
| speecht5_griffinlim | ~1.1 | ~1.3 | ~1.6 | FAIL ✓ (caught by both) |

### Accuracy on this test set

| Metric | Caught failing models | Accuracy |
|--------|----------------------|----------|
| UTMOS alone | 2/5 (speecht5 hifigan + griffinlim) | 75% |
| NISQA alone | 2/5 (speecht5 hifigan + griffinlim) | 75% |
| DSP artifact gate | 5/5 | 100% |
| NISQA + DSP | 5/5 | 100% |
| UTMOS + NISQA | 2/5 (no improvement over either alone) | 75% |

---

## Why NISQA Noisiness does not catch TTS vocoder artifacts

NISQA Noisiness is trained to recognize degradations that look like:
- Codec background noise (white/pink noise floor from lossy compression)
- Babble / room noise added on top of speech
- Packet loss artefacts

TTS vocoder artifacts that fail our gate (fastspeech2 broadband buzz, f5tts harmonic shaping,
mms tonal resonance) are produced by the neural vocoder at the harmonic level — they are
*structured* spectral distortions, not additive noise. NISQA sees clean-looking harmonic structure
and scores them as noisiness 3.8–4.6 (high quality). The model literally cannot see the artifact.

---

## Decision: drop UTMOS, keep NISQA

UTMOS and NISQA have identical accuracy (75%) on this test set. Keeping both adds:
- A separate Python 3.9 subprocess invocation (slower pipeline)
- One more model weight to manage (`epoch=3-step=7459.ckpt`)
- No additional coverage — they both miss the same 3 models

NISQA is strictly more informative (5 subscores vs 1 scalar). If NISQA's MOS passes, UTMOS
would also pass. If NISQA's MOS fails (speecht5-class artifacts), UTMOS also fails.

**UTMOS is not dropped from the codebase** — the gate file and config thresholds remain intact.
The recommendation is to exclude it from default pipeline runs by not listing it in the active
gates unless specifically needed for TTS naturalness ranking (not artifact detection).

---

## What actually catches the artifacts NISQA misses

The DSP artifact gate (`gates/gate_artifact.py`) using three signal-processing checks:

1. **HNR** — catches aperiodic vocooder noise (speecht5, Telugu STS)
2. **Pause median dBFS** — catches broadband silence noise (fastspeech2)
3. **Spectral artifact score** (H1/H2 + cepstral midQ + SFM_HF) — catches harmonic shaping (f5tts, mms)

See `artifact_gate_dev_notes.md` for full calibration detail.

---

## SSL-MOS (not implemented — decision)

SSL-MOS uses self-supervised representations (wav2vec2, HuBERT) as MOS predictors.
Evaluated and decided **not to add** because:
- Requires additional model weights and environment setup
- Produces a single MOS scalar — same limitation as UTMOS
- Does not resolve the fundamental problem: all three MOS approaches (UTMOS, NISQA, SSL-MOS)
  are trained on human perceptual ratings of speech, which rate naturalness globally.
  None of them isolate vocoder spectral artifacts specifically.
- NISQA already provides the best diagnostic signal of the group (subscores)

---

## Files

| File | Purpose |
|------|---------|
| `gates/gate_utmos.py` | UTMOS gate (retained, not default-active) |
| `gates/gate_nisqa.py` | NISQA gate (active) |
| `config.py` | `UTMOS_THRESHOLD`, `NISQA_THRESHOLDS`, `NISQA_DELTA_THRESHOLDS` |
| `weights/utmos/simple/epoch=3-step=7459.ckpt` | UTMOS model weights (gitignored) |
| `weights/nisqa/weights/nisqa.tar` | NISQA model weights (gitignored) |
