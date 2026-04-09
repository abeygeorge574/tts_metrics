# Next Phase: Missing Signal-Level Metrics

Current pipeline covers perceptual quality (NISQA, UTMOS), intelligibility (WER),
speaker identity (Speaker Sim), and content/prosody alignment (SER, AV, Pitch, Duration,
VAD, Amplitude, Accent, Artifact).

The following signal-level metrics are not yet implemented. They are especially useful
for measuring how much a TTS/STS system degrades the source audio at the waveform level,
which NISQA alone does not capture precisely.

---

## 1. SNR — Signal-to-Noise Ratio

**What it measures**: Power ratio of clean speech to background noise.

**Why it's missing here**: SNR requires a noise-free reference or a noise estimation step.
Our current references are Hindi recordings (possibly with room noise), not controlled
studio signals. A waveform-level SNR against Hindi would measure language difference, not noise.

**When to add**: When we have studio-clean English reference audio to compare against,
or when running a system that introduces measurable background noise (e.g., vocoders,
GAN artifacts). Also useful after speech enhancement passes (denoising).

**Python**: `pesq` library or manual energy ratio via librosa. Segmental SNR (framewise)
is more robust than global SNR for speech.

---

## 2. Log-Spectral Distance / MFCC Distance

**What it measures**: Frame-level distance between spectral envelopes of reference vs TTS.
Log-spectral distance (LSD) operates on power spectra; MFCC distance on cepstral coefficients.

**Why it matters**: NISQA captures perceived quality but is a neural network black box.
LSD/MFCC directly quantify how spectrally close the TTS output is to the reference —
useful for detecting coloration artifacts, formant drift, or codec distortions that NISQA
may smooth over.

**Limitation**: Requires time-aligned audio (same sentence, same timing). TTS outputs
can have different durations than the reference even for the same sentence. Needs DTW
(Dynamic Time Warping) alignment before distance computation.

**When to add**: When building a voice clone fidelity check (how closely does F5-TTS
match the source speaker's spectral envelope, beyond just speaker similarity cosine).

**Python**: `librosa.feature.mfcc` + scipy DTW or `fastdtw`. LSD via `np.log` on power spectra.

---

## 3. PESQ — Perceptual Evaluation of Speech Quality

**What it measures**: ITU-T P.862 standard MOS estimator for telephony/VoIP.
Range: -0.5 to 4.5.

**Why it matters**: PESQ is a well-established, interpretable MOS proxy with strong
correlation to human listening in the telephony domain. Unlike NISQA, PESQ is deterministic
and has published confidence intervals.

**Limitation for TTS**: PESQ was designed for wideband telephony degradation (codecs,
packet loss, noise). It treats the TTS output as a "degraded" version of a reference.
For TTS evaluation this is only meaningful if you have a studio-clean reference version
of exactly the same sentence. Cross-speaker or cross-language PESQ is not meaningful.

**When to add**: If the pipeline adds a speech enhancement or noise reduction pass,
PESQ is the standard way to measure before/after. Also useful if testing codec degradation
on TTS output before delivery.

**Python**: `pesq` (pip install pesq) — wraps the ITU reference implementation.
Requires 8 kHz (narrowband) or 16 kHz (wideband) audio.

---

## 4. STOI — Short-Time Objective Intelligibility

**What it measures**: ITU-T P.563-aligned intelligibility predictor. Range 0–1.
Higher = more intelligible under the target noise/degradation conditions.

**Relationship to WER gate**: STOI measures intelligibility without needing transcription.
It compares time-frequency envelopes of reference vs degraded signal — a good complement
to Whisper-based WER because it doesn't depend on the ASR model's language priors.

**Limitation**: Same as PESQ — requires a clean reference at the same sentence level.
Also assumes linear degradation models; neural vocoders can score low STOI even when
perceptually clear (because they synthesize differently from the reference waveform).

**When to add**: When testing TTS output under noisy playback conditions (e.g., in-car
TTS), or when evaluating a speech enhancement module. Not the right metric for
vanilla TTS quality assessment without a matched clean reference.

**Python**: `pystoi` (pip install pystoi).

---

## Priority order for next phase

1. **MFCC Distance with DTW** — most actionable now. Can quantify voice clone fidelity
   (F5-TTS vs enrollment speaker) in a way Speaker Sim cosine doesn't fully capture.
   No special reference requirements — just alignment.

2. **SNR (segmental)** — add as an amplitude gate extension once we have clean English
   reference audio. Catches cases where vocoders introduce noise floors.

3. **PESQ / STOI** — add only if the pipeline gets a speech enhancement / denoising
   pass, or if testing codec delivery pipelines. Not the right fit for bare TTS evaluation.

---

## Note on current gaps

The artifact gate (spectral flatness, HNR, ZCR, crest factor) partially fills the
SNR/LSD gap by detecting abnormal spectral content without needing a reference.
For most TTS quality gating, the combination of NISQA + Artifact gate is sufficient.

The MFCC Distance gate is the highest-value addition for the dubbing use case because
it directly measures speaker characteristic preservation — something ECAPA-TDNN cosine
similarity captures at the embedding level but not at the frame-level detail needed to
catch subtle coloration shifts.
