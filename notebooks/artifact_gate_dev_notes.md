# Artifact Gate — Development Notes

## Context

Starting point: UTMOS scores were catching some robotic/metallic TTS as low-quality, but
UTMOS is a black-box MOS predictor. No interpretable signal for *why* audio fails. Goal: build
an explicit artifact detector that gives labeled error codes and validated thresholds.

---

## Models tested (ground truth established by user listening)

| Model | Artifact type | Expected | Final gate |
|-------|--------------|----------|-----------|
| gtts | Clean | PASS | PASS ✓ |
| kokoro | Clean | PASS | PASS ✓ |
| samantha | Robotic timbre, no buzz | PASS | PASS ✓ |
| kokoro_v1 | Very light silence buzz | WARN | PASS [WARN_SILENCE_FLOOR] ✓ |
| parler_mini | Very light silence buzz | WARN | PASS [WARN_SILENCE_FLOOR] ✓ |
| edge_tts_andrew | Clean (Microsoft neural) | PASS | PASS ✓ |
| edge_tts_ava | Clean, very slight tonal | PASS | PASS [WARN_SILENCE_FLOOR on 4/5] ✓ |
| fastspeech2 | Broadband constant buzz in silence and speech | FAIL | FAIL [ERR_SPECTRAL_ARTIFACT] ✓ |
| mms | Robotic, tonal artifacts | FAIL | FAIL [ERR_SPECTRAL_ARTIFACT] ✓ |
| f5tts | Tonal artifact on speech (samples 3/4/5 clear) | FAIL | FAIL [ERR_SPECTRAL_ARTIFACT] ✓ |
| speecht5_hifigan | Buzzing + robotic | FAIL | FAIL [ERR_VOICE_BUZZ] ✓ |
| speecht5_griffinlim | All noise, no intelligible speech | FAIL | FAIL [ERR_VOICE_BUZZ + ERR_SPECTRAL_ARTIFACT] ✓ |
| Telugu STS 1,2 | Tonal artifact on voiced/high-pitch | FAIL | FAIL [ERR_VOICE_BUZZ, HNR 7.3-7.5 dB] ✓ |
| Hindi STS A,D | Subtle silence-only broadband buzz | FAIL (known gap) | PASS — not detectable (see limitations) |

---

## Metrics in the gate (gates/gate_artifact.py)

### Check 1: HNR — Harmonic-to-Noise Ratio
- Tool: parselmouth (Praat autocorrelation) on voiced frames
- Threshold: HNR_mean < 8 dB → **ERR_VOICE_BUZZ**
- Catches: speecht5_hifigan (2-5 dB), speecht5_griffinlim (-6.6 dB), Telugu STS (7.3-7.5 dB)
- Why it works: aperiodic noise or non-harmonic tonal overlay reduces HNR
- Why it misses: if noise is *at harmonic frequencies*, Praat still sees it as periodic → HNR stays high. This is why F5-TTS passed HNR (its artifacts are harmonic shaping, not random noise).

### Check 2: Pause median dBFS — Background static in silence
- Method: find contiguous frame runs ≥ 160ms all below -30 dBFS. Take median dBFS of those frames.
- Two tiers:
  - > -35 dBFS → **ERR_BACKGROUND_STATIC** (hard FAIL)
  - > -58 dBFS → **WARN_SILENCE_FLOOR** (advisory, not a FAIL)
- WARN does NOT cause FAIL — it appears in the flags column but result stays PASS
- Why -30 dBFS for pause detection (not the VAD gate's -40 dBFS): fastspeech2's noisy silence frames
  are at -40 to -44 dBFS. Using -40 would exclude them from measurement. Critical distinction.
- Catches: fastspeech2 via spectral artifact (the pause floor itself is borderline at -40 to -50 dBFS
  and no longer the primary detector for it)

### Check 3: Spectral shape artifact score — Combined of three sub-metrics on voiced frames
- This was added specifically to catch F5-TTS, which passed both HNR and pause checks.
- All three sub-metrics computed on voiced frames only (rms_db > -25)

**a) H1/H2 ratio** (via pyin F0 detection):
- Amplitude of first harmonic / second harmonic in voiced frames
- Clean speech: 1.65–2.89. F5-TTS: 0.71–1.59.
- Low ratio = flattened harmonic slope → spectral shaping artifact

**b) Cepstral mid-quefrency energy**:
- Normalised energy in the pitch-period quefrency range (60–400 Hz)
- Clean: 0.015–0.019. F5-TTS: 0.024–0.032.
- High value = excessive periodic structure in spectrum → tonal artifact

**c) SFM 4-8kHz (voiced)**:
- Spectral flatness of the 4–8 kHz band in voiced frames
- Clean: 0.130–0.138. F5-TTS: 0.156–0.183.
- High value = broadband noise in HF voiced region

**Combined score** = mean of 3 normalised deviations from clean reference values.
Threshold: combined > 0.15 → **ERR_SPECTRAL_ARTIFACT**

Voting: any ERR_* → FAIL. WARN_* → PASS with flag. No NEAR_MISS.

---

## Issues encountered and how they were resolved

### F5-TTS bypassing HNR
- Problem: F5-TTS artifacts are at harmonic frequencies (harmonic slope flattening).
  HNR measures aperiodic noise — it sees the periodicity and passes.
- Fix: Added the three spectral sub-metrics (H1/H2, cepstral midQ, SFM_HF).
  F5-TTS scores 0.36-0.54 combined vs clean models at -0.15 to 0.07.

### Wrong silence detection threshold
- Problem: Gate originally used `config.SILENCE_DB = -40` (shared with VAD gate).
  FastSpeech2's noisy silence frames are at -40 to -44 dBFS.
  Using -40 as threshold excluded them, so SNR was measured as clean — false PASS.
- Fix: Added `ARTIFACT_SILENCE_DETECT_DB = -30` separate from VAD's SILENCE_DB.

### SFM frame alignment IndexError
- Problem: `librosa.util.frame` and `librosa.stft` produce different frame counts.
  Masked frame selection crashed.
- Fix: Compute RMS directly from STFT magnitudes so they share the same grid.

### Kokoro_v1 / parler_mini light silence buzz
- Problem: These are acceptable in production per user, but gate was flagging FAIL.
- Fix: Two-tier silence system — WARN range (-35 to -58 dBFS) gives advisory flag
  without causing FAIL. Only silence floors louder than -35 dBFS cause hard FAIL.
- Threshold calibration: fastspeech2 separation from parler_mini is done by
  ERR_SPECTRAL_ARTIFACT, so silence threshold is free to be more permissive.

### H1/H2 individual gate causing edge-tts false positives
- Problem: Added an individual H1/H2 < 1.5 gate to catch cases where combined
  score was diluted. Worked for fastspeech2, but edge-tts Andrew/Ava voices have
  naturally low H1/H2 (male voice, different phonation style) while SFM and
  cepstral are very clean. Individual gate fired incorrectly.
- Root cause: H1/H2 ratio is voice-type dependent, not just artifact-dependent.
- Fix: Removed individual H1/H2 gate. Lowered combined threshold 0.30 → 0.15
  instead. Clean model max: 0.073 (kokoro). Artifact model min: 0.179 (fastspeech2).
  Large gap; 0.15 gives ample margin on both sides.

### WARN_SILENCE_FLOOR incorrectly causing FAIL
- Problem: Both ERR_* and WARN_* codes were added to the same `error_flags` list,
  and any non-empty list → FAIL. WARN was never supposed to fail.
- Fix: Separate `error_flags` and `warn_flags` lists. Only `error_flags` drives FAIL.

---

## Threshold calibration summary (config.py)

```
ARTIFACT_HNR_ABS_THRESHOLD   = 8.0     # dB — ERR_VOICE_BUZZ
ARTIFACT_SILENCE_DETECT_DB   = -30.0   # dBFS — frame qualifies as silence
ARTIFACT_MIN_PAUSE_FRAMES    = 10      # ~160ms minimum pause
ARTIFACT_SILENCE_FAIL_DB     = -35.0   # dBFS — ERR_BACKGROUND_STATIC (hard FAIL)
ARTIFACT_SILENCE_WARN_DB     = -58.0   # dBFS — WARN_SILENCE_FLOOR (advisory)
ARTIFACT_H1H2_THRESHOLD      = 1.5     # reference only (individual gate removed)
ARTIFACT_CEP_MIDQ_THRESHOLD  = 0.022   # reference only
ARTIFACT_SFM_HF_THRESHOLD    = 0.16    # reference only
ARTIFACT_CLEAN_H1H2          = 1.9     # normalisation reference
ARTIFACT_CLEAN_SFM_HF        = 0.134   # normalisation reference
ARTIFACT_CLEAN_CEP            = 0.017   # normalisation reference
ARTIFACT_COMBINED_THRESHOLD  = 0.15    # ERR_SPECTRAL_ARTIFACT
```

---

## Known limitations

### Hindi STS subtle silence buzz — NOT detected
- Hindi STS pause floors at -85 to -88 dBFS are within the noise floor of
  the reference recordings themselves (gtts reference is at -84 dBFS clean).
- The artifact is real but indistinguishable from mic room noise without a
  matched same-session reference. The original plan had an SNR delta approach
  (TTS_SNR / REF_SNR ratio) to catch this cross-lingual. Not implemented.
- Impact: Hindi STS will PASS this gate even with subtle silence buzz.

### Subtle non-harmonic tonal artifacts below HNR threshold
- A pure tone injected at -20 dB SNR (clearly audible) does not drop HNR below 8 dB
  if it's wideband. HNR threshold at 8 dB is calibrated for severe vocoder artifacts.
- Inharmonicity metric (e.g., Praat's inharmonicity measure) would be needed to close this.

### Cross-lingual — spectral metrics valid, HNR valid
- Spectral artifact metrics are language-agnostic (voiced frame energy/harmonic analysis).
- HNR is absolute (no reference needed). Both valid for Hindi/Telugu STS output.

---

## Files

| File | Purpose |
|------|---------|
| `gates/gate_artifact.py` | Gate implementation |
| `config.py` | All thresholds (section: Artifact / Vocoder Buzz gate) |
| `generate_tts_samples.py` | Generate F5-TTS, Kokoro, Parler, edge-tts samples |
| `generate_poison.py` | Inject synthetic artifacts for threshold calibration testing |
| `run_poison_gate.py` | Run gate on poison directory |

---

## What to do when returning

1. Run `python gates/gate_artifact.py` to get current verdicts on all models in `data/models/`
2. If a new model is being evaluated, add its samples to `data/models/<model_name>/`
3. If gate verdict disagrees with perception, check which metric is the driver
   (Error_Flags column) and adjust the corresponding threshold in config.py
4. If the Hindi STS silent buzz issue matters, the SNR delta approach needs implementing:
   `delta_ratio = tts_snr_ratio / ref_snr_ratio`, threshold ~0.70, requires reference audio
