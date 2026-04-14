"""
Gate: Artifact / Vocoder Buzz Detector
Env : base (python 3.13)

Detects three types of audio artifacts in TTS/STS output:

1. Tonal / voiced artifacts  →  HNR (Harmonic-to-Noise Ratio)
   Metallic tonal buzz, vocoder overlay, non-harmonic resonance on voiced speech.
   Measured via parselmouth/Praat autocorrelation on voiced frames.
   Threshold: HNR_mean < ARTIFACT_HNR_ABS_THRESHOLD (8 dB) → ERR_VOICE_BUZZ
   Near-miss: HNR in [6.4, 8.0) dB → NEAR_MISS
   Catches: SpeechT5-HiFiGAN, Griffin-Lim, Telugu STS tonal artifacts.
   Works on any segment length. No reference needed.

2. Background static in silence  →  median dBFS of real pause frames
   Broadband noise floor audible during pauses (vocoder residual, codec hiss).
   Real pause = contiguous run of frames ≥ 160 ms all below −30 dBFS.
   Threshold: median_pause_db > ARTIFACT_SILENCE_FAIL_DB (−35 dBFS) → ERR_BACKGROUND_STATIC
   Catches: FastSpeech2 broadband constant noise (median ~−44 dBFS).
   N/A when no real pauses are found (short segment, continuous speech).

3. Spectral shape artifacts  →  combined score of three sub-metrics (librosa-based)
   Catches subtle flow-matching vocoder artifacts not detected by HNR or pause floor.
   Sub-metrics (all computed on voiced frames only, rms_db > −25):
     a) H1/H2 ratio: amplitude of first harmonic / second harmonic (via pyin F0).
        Low ratio = flattened harmonic slope → spectral shaping artifact.
        Threshold < ARTIFACT_H1H2_THRESHOLD (1.5)
     b) Cepstral mid-quefrency energy: normalised energy at pitch-period quefrencies.
        High value = excessive periodic structure in spectrum → tonal artifact.
        Threshold > ARTIFACT_CEP_MIDQ_THRESHOLD (0.022)
     c) SFM 4-8kHz (voiced): spectral flatness measure of HF band in voiced frames.
        High value = broadband noise in HF voiced region → broadband artifact.
        Threshold > ARTIFACT_SFM_HF_THRESHOLD (0.16)
   Combined score = avg of 3 normalised deviations from clean reference.
   Threshold: combined > ARTIFACT_COMBINED_THRESHOLD (0.15) → ERR_SPECTRAL_ARTIFACT
   Near-miss: combined in (0.15, 0.18] → NEAR_MISS
   Validated: f5tts=0.48 (FAIL), gtts/kokoro/samantha=−0.04–0.04 (PASS),
              kokoro_v1/parler_mini=−0.11–−0.06 (PASS), fastspeech2=0.32 (FAIL).

Voting:
  ERR_* → hard FAIL.
  WARN_SILENCE_FLOOR or near-miss thresholds (no ERR_*) → NEAR_MISS.
  Otherwise → PASS.

Reference: not used for gating. Thresholds are absolute and data-validated
against: gtts, kokoro, samantha (PASS), fastspeech2, HiFiGAN, Griffin-Lim,
Hindi STS, Telugu STS, f5tts (FAIL).

Known limitation: subtle silence buzz at levels comparable to mic room noise
(e.g. Hindi STS at ~−85 dBFS) is not detectable without a matched same-session
reference. This gate catches clearly audible artifacts only.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_SR        = 16000
_FRAME_SZ  = 512
_HOP_SZ    = 256

# Constants for spectral shape metrics
_SA_N_FFT  = 1024     # FFT size for spectral artifact metrics
_SA_HOP    = 160      # hop size (10 ms)
_SA_F_MIN  = 60       # minimum F0 (Hz) for pyin
_SA_F_MAX  = 400      # maximum F0 (Hz) for pyin


# ── HNR via parselmouth ────────────────────────────────────────────────────────

def compute_hnr(audio_path: str) -> dict:
    """
    Mean Harmonic-to-Noise Ratio on voiced frames (parselmouth autocorrelation).

    High HNR = clean periodic speech.
    Low HNR  = aperiodic noise or tonal overlay not at harmonic frequencies.

    Returns:
      hnr_mean        — mean HNR across voiced frames (dB)
      hnr_voiced_frac — fraction of analysis frames that Praat marks as voiced
      hnr_error       — set only on failure (parselmouth not installed, no voiced frames)
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {"hnr_mean": None, "hnr_voiced_frac": None,
                "hnr_error": "parselmouth not installed"}

    try:
        snd = parselmouth.Sound(audio_path)
        if snd.sampling_frequency != _SR:
            snd = snd.resample(_SR)

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        n_frames    = call(harmonicity, "Get number of frames")

        hnr_vals = []
        for i in range(1, n_frames + 1):
            v = call(harmonicity, "Get value in frame", i)
            if v == v and v > -200:      # not NaN and not undefined (unvoiced)
                hnr_vals.append(v)

        if not hnr_vals:
            return {"hnr_mean": None, "hnr_voiced_frac": 0.0,
                    "hnr_error": "no voiced frames"}

        return {
            "hnr_mean"       : round(float(np.mean(hnr_vals)), 2),
            "hnr_voiced_frac": round(len(hnr_vals) / n_frames, 3),
        }
    except Exception as e:
        return {"hnr_mean": None, "hnr_voiced_frac": None, "hnr_error": str(e)}


# ── Pause median dBFS ─────────────────────────────────────────────────────────

def compute_pause_median_db(audio_path: str) -> dict:
    """
    Median dBFS of real pause frames.

    Real pause = contiguous run of frames where rms_db < SILENCE_DB (−30 dBFS)
                 lasting at least MIN_PAUSE_FRAMES frames (≥ 160 ms).

    Uses MEDIAN (not mean) so a single breath sound or click in a long pause
    does not inflate the noise floor estimate.

    Returns:
      real_pause_frac — fraction of total frames inside real pauses
      n_real_pauses   — number of pause events ≥ 160 ms
      median_db       — median dBFS of pause frames (None if no real pauses)
      silence_na      — True when no real pauses found (metric is N/A)
    """
    y, sr = sf.read(audio_path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != _SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_SR)

    frames = librosa.util.frame(y, frame_length=_FRAME_SZ, hop_length=_HOP_SZ)
    rms    = np.sqrt(np.mean(frames**2, axis=0))
    rms_db = 20 * np.log10(rms + 1e-10)
    n      = len(rms)

    silence_db       = getattr(config, "ARTIFACT_SILENCE_DETECT_DB", -30.0)
    min_pause_frames = getattr(config, "ARTIFACT_MIN_PAUSE_FRAMES", 10)

    below = rms_db < silence_db

    # Find contiguous runs below threshold
    real_pause_mask = np.zeros(n, dtype=bool)
    n_real_pauses   = 0
    i = 0
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if (j - i) >= min_pause_frames:
                real_pause_mask[i:j] = True
                n_real_pauses += 1
            i = j
        else:
            i += 1

    n_pause = real_pause_mask.sum()
    real_pause_frac = float(n_pause / n)

    if n_pause == 0:
        return {
            "real_pause_frac": 0.0,
            "n_real_pauses"  : 0,
            "median_db"      : None,
            "silence_na"     : True,
        }

    pause_db  = rms_db[real_pause_mask]
    median_db = float(np.median(pause_db))

    return {
        "real_pause_frac": round(real_pause_frac, 3),
        "n_real_pauses"  : n_real_pauses,
        "median_db"      : round(median_db, 2),
        "silence_na"     : False,
    }


# ── Spectral shape artifact metrics ───────────────────────────────────────────

def _voiced_mask(y: np.ndarray, sr: int = _SR, threshold_db: float = -25.0) -> np.ndarray:
    """Return boolean mask of voiced frames (rms_db > threshold_db)."""
    rms = librosa.feature.rms(y=y, frame_length=_SA_N_FFT, hop_length=_SA_HOP)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    return rms_db > threshold_db


def compute_h1h2(y: np.ndarray, sr: int = _SR) -> float | None:
    """
    H1/H2 ratio: amplitude of first harmonic / second harmonic in voiced frames.

    Uses librosa.pyin for F0 detection, then finds harmonic peaks in the mean
    amplitude spectrum of pyin-voiced frames.

    Returns the H1/H2 ratio, or None if insufficient voiced frames.

    Interpretation: clean speech typically has H1/H2 ≈ 1.7–2.9.
    Artifact-bearing models show H1/H2 < 1.5 (flattened harmonic slope).
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=_SA_F_MIN, fmax=_SA_F_MAX, sr=sr,
            frame_length=_SA_N_FFT, hop_length=_SA_HOP,
        )
    except Exception:
        return None

    if voiced_flag is None or voiced_flag.sum() < 5:
        return None

    S_amp = np.abs(librosa.stft(y, n_fft=_SA_N_FFT, hop_length=_SA_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_SA_N_FFT)

    n_frames = min(S_amp.shape[1], len(voiced_flag))
    f0_n = f0[:n_frames]
    vf_n = voiced_flag[:n_frames]
    valid = vf_n & ~np.isnan(f0_n)

    if valid.sum() < 5:
        return None

    f0_med = float(np.median(f0_n[valid]))
    mean_spec = S_amp[:, :n_frames][:, valid].mean(axis=1)

    def _peak(target_hz: float) -> float:
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        lo = max(0, idx - 2)
        hi = min(len(mean_spec) - 1, idx + 2)
        return float(mean_spec[lo:hi + 1].max())

    h1 = _peak(f0_med)
    h2 = _peak(2 * f0_med)
    if h2 < 1e-10:
        return None
    return round(h1 / h2, 4)


def compute_cep_midq(y: np.ndarray, sr: int = _SR) -> float | None:
    """
    Cepstral mid-quefrency energy (voiced frames).

    For each voiced frame: compute log-power cepstrum, extract normalised energy
    in the quefrency range corresponding to pitch periods (sr/400 … sr/60 samples).
    Returns mean across voiced frames.

    Higher values indicate stronger periodic structure in the spectrum consistent
    with tonal/flow-matching vocoder artifacts.
    Clean speech: ≈ 0.015–0.019.  Artifact-bearing: ≥ 0.022.
    """
    frame_len = int(0.025 * sr)  # 25 ms
    hop_cep   = int(0.010 * sr)  # 10 ms

    rms    = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_cep)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)

    min_period = sr // _SA_F_MAX   # ~40 samples at 400 Hz
    max_period = sr // _SA_F_MIN   # ~267 samples at 60 Hz

    vals = []
    for i, db in enumerate(rms_db):
        if db < -25.0:
            continue
        start = i * hop_cep
        end   = start + frame_len
        if end > len(y):
            break
        frame = y[start:end] * np.hanning(frame_len)
        log_ps = np.log(np.abs(np.fft.rfft(frame, n=1024)) ** 2 + 1e-10)
        cep    = np.fft.irfft(log_ps)
        total  = float(np.sum(cep ** 2)) + 1e-10
        vals.append(float(np.sum(cep[min_period:max_period] ** 2)) / total)

    if not vals:
        return None
    return round(float(np.mean(vals)), 6)


def compute_sfm_hf(y: np.ndarray, sr: int = _SR) -> float | None:
    """
    Spectral Flatness Measure in 4-8 kHz, voiced frames only.

    Geometric mean / arithmetic mean of spectral power in the 4–8 kHz band,
    computed per voiced frame and then averaged.

    Higher SFM = more noise-like (flat spectrum) in the high-frequency region.
    Clean speech: ≈ 0.130–0.138.  Artifact-bearing (broadband HF noise): ≥ 0.16.
    """
    S     = np.abs(librosa.stft(y, n_fft=_SA_N_FFT, hop_length=_SA_HOP)) ** 2 + 1e-20
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_SA_N_FFT)
    vm    = _voiced_mask(y, sr)

    n_frames = min(S.shape[1], len(vm))
    S_v = S[:, :n_frames][:, vm[:n_frames]]

    if S_v.shape[1] < 5:
        return None

    band = (freqs >= 4000) & (freqs < 8000)
    if not band.any():
        return None

    S_band = S_v[band, :]
    sfm_per_frame = (
        np.exp(np.mean(np.log(S_band), axis=0))
        / (S_band.mean(axis=0) + 1e-20)
    )
    return round(float(np.mean(sfm_per_frame)), 6)


def compute_spectral_artifact_score(audio_path: str) -> dict:
    """
    Combined spectral artifact score from three sub-metrics.

    Returns a dict with keys:
      h1h2          — H1/H2 ratio (None if not computable)
      cep_midq      — cepstral mid-quefrency energy (None if not computable)
      sfm_hf        — SFM 4-8 kHz voiced (None if not computable)
      combined      — combined artefact score (float or None)
      sa_pass       — True = PASS, False = FAIL, None = not enough data
    """
    try:
        y, sr = librosa.load(audio_path, sr=_SR)
    except Exception as e:
        return {
            "h1h2": None, "cep_midq": None, "sfm_hf": None,
            "combined": None, "sa_pass": None,
            "sa_error": str(e),
        }

    h1h2     = compute_h1h2(y, sr)
    cep_midq = compute_cep_midq(y, sr)
    sfm_hf   = compute_sfm_hf(y, sr)

    ref_h1h2 = getattr(config, "ARTIFACT_CLEAN_H1H2",   1.9)
    ref_sfm  = getattr(config, "ARTIFACT_CLEAN_SFM_HF", 0.134)
    ref_cep  = getattr(config, "ARTIFACT_CLEAN_CEP",    0.017)

    scores = []
    if h1h2 is not None:
        scores.append((ref_h1h2 - h1h2) / ref_h1h2)
    if sfm_hf is not None:
        scores.append((sfm_hf - ref_sfm) / ref_sfm)
    if cep_midq is not None:
        scores.append((cep_midq - ref_cep) / ref_cep)

    if len(scores) >= 2:
        combined = round(float(np.mean(scores)), 4)
        thresh   = getattr(config, "ARTIFACT_COMBINED_THRESHOLD", 0.15)
        sa_pass  = combined <= thresh
    else:
        combined = None
        sa_pass  = None

    return {
        "h1h2"    : h1h2,
        "cep_midq": cep_midq,
        "sfm_hf"  : sfm_hf,
        "combined": combined,
        "sa_pass" : sa_pass,
    }


# ── Gate interface ─────────────────────────────────────────────────────────────

def load_model():
    return {}   # no ML model to load beyond parselmouth (imported on demand)


def run_gate(model_state=None):
    import pandas as pd

    MODELS_DIR    = (model_state or {}).get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = (model_state or {}).get("ref_dir")    or config.REFERENCE_DIR  # not used for gating

    HNR_THRESHOLD      = getattr(config, "ARTIFACT_HNR_ABS_THRESHOLD",    8.0)
    PAUSE_FAIL_THRESH  = getattr(config, "ARTIFACT_SILENCE_FAIL_DB",    -35.0)
    PAUSE_WARN_THRESH  = getattr(config, "ARTIFACT_SILENCE_WARN_DB",    -58.0)
    SA_THRESHOLD       = getattr(config, "ARTIFACT_COMBINED_THRESHOLD",   0.15)
    NM_MARGIN          = getattr(config, "ARTIFACT_NEAR_MISS_MARGIN",     0.20)
    MIN_SEG_DUR        = getattr(config, "MIN_SEGMENT_DURATION",           2.0)

    HNR_NM_LOWER    = HNR_THRESHOLD * (1 - NM_MARGIN)       # 6.4 dB
    SA_NM_UPPER     = SA_THRESHOLD  * (1 + NM_MARGIN)        # 0.18

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models found: {model_folders}")

    results = []

    for model in model_folders:
        print(f"\n{'='*55}\nModel: {model}\n{'='*55}")
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])

        for wav_file in wav_files:
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(model_path, wav_file)

            print(f"\n  Sample: {sample_name}")

            # Duration check
            try:
                info     = sf.info(tts_path)
                duration = info.duration
            except Exception:
                duration = 0.0
            flag = "SHORT" if duration < MIN_SEG_DUR else "—"

            # ── Step 1: HNR ──────────────────────────────────────────────────
            hnr_result = compute_hnr(tts_path)
            hnr_mean   = hnr_result.get("hnr_mean")
            hnr_error  = hnr_result.get("hnr_error")

            if hnr_mean is not None:
                if hnr_mean >= HNR_THRESHOLD:
                    hnr_verdict = "PASS"
                    hnr_label   = "PASS"
                elif hnr_mean >= HNR_NM_LOWER:
                    hnr_verdict = "NEAR_MISS"
                    hnr_label   = "NEAR_MISS"
                else:
                    hnr_verdict = "FAIL"
                    hnr_label   = "FAIL  ← ERR_VOICE_BUZZ"
                hnr_str  = f"{hnr_mean:.1f} dB (voiced={hnr_result['hnr_voiced_frac']:.0%})"
                print(f"  HNR    : {hnr_str}  → {hnr_label}")
            else:
                hnr_verdict = None
                print(f"  HNR    : unavailable ({hnr_error})")

            # ── Step 2 + 3: Pause existence + median dBFS ────────────────────
            pause_result = compute_pause_median_db(tts_path)

            if pause_result["silence_na"]:
                pause_verdict = None
                median_db     = None
                print(f"  Pause  : no real pauses ≥160ms detected  → N/A")
            else:
                median_db = pause_result["median_db"]
                if median_db > PAUSE_FAIL_THRESH:
                    pause_verdict = "FAIL"
                    pause_label   = "FAIL  ← ERR_BACKGROUND_STATIC"
                elif median_db > PAUSE_WARN_THRESH:
                    # WARN_SILENCE_FLOOR maps to NEAR_MISS in voting
                    pause_verdict = "NEAR_MISS"
                    pause_label   = "NEAR_MISS  ← WARN_SILENCE_FLOOR"
                else:
                    pause_verdict = "PASS"
                    pause_label   = "PASS"
                print(f"  Pause  : {pause_result['n_real_pauses']} pauses "
                      f"({pause_result['real_pause_frac']:.0%} of audio)  "
                      f"median={median_db:.1f} dBFS  → {pause_label}")

            # ── Step 4: Spectral shape artifact score ─────────────────────────
            sa_result   = compute_spectral_artifact_score(tts_path)
            sa_combined = sa_result.get("combined")
            h1h2        = sa_result.get("h1h2")
            cep_midq    = sa_result.get("cep_midq")
            sfm_hf      = sa_result.get("sfm_hf")

            if sa_combined is not None:
                if sa_combined <= SA_THRESHOLD:
                    sa_verdict = "PASS"
                    sa_label   = "PASS"
                elif sa_combined <= SA_NM_UPPER:
                    sa_verdict = "NEAR_MISS"
                    sa_label   = "NEAR_MISS"
                else:
                    sa_verdict = "FAIL"
                    sa_label   = "FAIL  ← ERR_SPECTRAL_ARTIFACT"
            else:
                sa_verdict = None
                sa_label   = "N/A"

            def _fmt(v, spec):
                return format(v, spec) if v is not None else "N/A"

            if sa_combined is not None:
                sa_str = (
                    f"H1H2={_fmt(h1h2, '.2f')}, CepMidQ={_fmt(cep_midq, '.4f')},"
                    f" SFM_HF={_fmt(sfm_hf, '.4f')}  combined={sa_combined:.3f}"
                )
                print(f"  SpectralShape: {sa_str}  → {sa_label}")
            else:
                print(f"  SpectralShape: insufficient data  → N/A")

            # ── Voting ────────────────────────────────────────────────────────
            # ERR_* (verdict==FAIL) → hard FAIL.
            # NEAR_MISS (WARN_* or near-miss thresholds) → NEAR_MISS if no ERR_*.
            # Otherwise → PASS.
            error_flags   = []
            near_miss_flags = []

            if hnr_verdict == "FAIL":
                error_flags.append("ERR_VOICE_BUZZ")
            elif hnr_verdict == "NEAR_MISS":
                near_miss_flags.append("WARN_HNR_LOW")

            if pause_verdict == "FAIL":
                error_flags.append("ERR_BACKGROUND_STATIC")
            elif pause_verdict == "NEAR_MISS":
                near_miss_flags.append("WARN_SILENCE_FLOOR")

            if sa_verdict == "FAIL":
                error_flags.append("ERR_SPECTRAL_ARTIFACT")
            elif sa_verdict == "NEAR_MISS":
                near_miss_flags.append("WARN_SA_BORDERLINE")

            if error_flags:
                result = "FAIL"
            elif near_miss_flags:
                result = "NEAR_MISS"
            elif hnr_verdict is None and pause_verdict is None and sa_verdict is None:
                result = "SKIP"      # no metrics could run
            else:
                result = "PASS"

            all_flags = error_flags + near_miss_flags
            flags_str = ",".join(all_flags) if all_flags else ""

            print(f"  Result : {result}" + (f"  [{flags_str}]" if flags_str else ""))

            results.append({
                "Model"                                                 : model,
                "Sample"                                                : sample_name,
                "Duration_s"                                            : round(duration, 2),
                # HNR
                "HNR Mean dB (threshold≥8.0dB, near_miss≥6.4dB)"       : hnr_mean,
                "HNR_Voiced_Frac"                                       : hnr_result.get("hnr_voiced_frac"),
                "HNR Pass (PASS/NEAR_MISS/FAIL)"                        : hnr_verdict if hnr_verdict is not None else "N/A",
                # Pause / median dBFS
                "Real_Pauses"                                           : pause_result["n_real_pauses"],
                "Pause_Frac"                                            : pause_result["real_pause_frac"],
                "Pause_Median_DB"                                       : median_db,
                "Pause Pass (PASS/NEAR_MISS/FAIL)"                      : pause_verdict if pause_verdict is not None else "N/A",
                # Spectral shape artifact score
                "SA_H1H2"                                               : h1h2,
                "SA_CepMidQ"                                            : cep_midq,
                "SA_SFM_HF"                                             : sfm_hf,
                "SA Combined Score (threshold≤0.15, near_miss≤0.18)"    : sa_combined,
                "SA Pass (PASS/NEAR_MISS/FAIL)"                         : sa_verdict if sa_verdict is not None else "N/A",
                # Overall
                "Error Flags (ERR_*=fail|WARN_*=near_miss)"             : flags_str,
                "Final Pass (PASS/NEAR_MISS/FAIL)"                      : result,
                "Flag (—=clean|SHORT=<2s)"                              : flag,
            })

    df = pd.DataFrame(results)

    # Summary
    fp_col  = "Final Pass (PASS/NEAR_MISS/FAIL)"
    hnr_col = "HNR Mean dB (threshold≥8.0dB, near_miss≥6.4dB)"
    sa_col  = "SA Combined Score (threshold≤0.15, near_miss≤0.18)"

    summary_rows = []
    for model in model_folders:
        mdf        = df[df["Model"] == model]
        pass_count = (mdf[fp_col] == "PASS").sum()
        nm_count   = (mdf[fp_col] == "NEAR_MISS").sum()
        fail_count = (mdf[fp_col] == "FAIL").sum()
        summary_rows.append({
            "Model"                          : model,
            "Pass Rate (PASS / total)"       : f"{pass_count}/{len(mdf)}",
            "Near Miss Rate (NEAR_MISS / total)": f"{nm_count}/{len(mdf)}",
            "Fail"                           : int(fail_count),
            "Mean_HNR_dB"                    : round(mdf[hnr_col].dropna().mean(), 2)
                                               if mdf[hnr_col].notna().any() else None,
            "Mean_Pause_dBFS"                : round(mdf["Pause_Median_DB"].dropna().mean(), 2)
                                               if mdf["Pause_Median_DB"].notna().any() else None,
            "Mean_SA_H1H2"                   : round(mdf["SA_H1H2"].dropna().mean(), 3)
                                               if mdf["SA_H1H2"].notna().any() else None,
            "Mean_SA_Combined"               : round(mdf[sa_col].dropna().mean(), 3)
                                               if mdf[sa_col].notna().any() else None,
            "Error_Flags"                    : "; ".join(
                                               f for f in mdf["Error Flags (ERR_*=fail|WARN_*=near_miss)"].dropna() if f
                                               ) or "—",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Fail", ascending=False)

    return df, summary_df


def print_results(df, summary_df):
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    fp_col   = "Final Pass (PASS/NEAR_MISS/FAIL)"
    hnr_col  = "HNR Mean dB (threshold≥8.0dB, near_miss≥6.4dB)"
    hnrp_col = "HNR Pass (PASS/NEAR_MISS/FAIL)"
    sa_col   = "SA Combined Score (threshold≤0.15, near_miss≤0.18)"
    sap_col  = "SA Pass (PASS/NEAR_MISS/FAIL)"
    pp_col   = "Pause Pass (PASS/NEAR_MISS/FAIL)"
    err_col  = "Error Flags (ERR_*=fail|WARN_*=near_miss)"
    flag_col = "Flag (—=clean|SHORT=<2s)"

    print("\n========== PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Duration_s",
        hnr_col, hnrp_col,
        "Real_Pauses", "Pause_Median_DB", pp_col,
        "SA_H1H2", "SA_CepMidQ", "SA_SFM_HF", sa_col, sap_col,
        err_col, fp_col, flag_col,
    ]].to_string(index=False))

    print("\n========== MODEL SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== GUIDE ==========")
    print(f"  HNR_Mean      < {getattr(config, 'ARTIFACT_HNR_ABS_THRESHOLD', 8.0):.0f} dB"
          f"   → ERR_VOICE_BUZZ       (tonal/metallic buzz on voiced speech)")
    print(f"  HNR_Mean in [{getattr(config, 'ARTIFACT_HNR_ABS_THRESHOLD', 8.0) * (1 - getattr(config, 'ARTIFACT_NEAR_MISS_MARGIN', 0.20)):.1f}, {getattr(config, 'ARTIFACT_HNR_ABS_THRESHOLD', 8.0):.0f}) → WARN_HNR_LOW (NEAR_MISS)")
    print(f"  Pause_Median  > {getattr(config, 'ARTIFACT_SILENCE_FAIL_DB', -45.0):.0f} dBFS"
          f"  → ERR_BACKGROUND_STATIC (hard FAIL — clearly audible broadband noise in pauses)")
    print(f"  Pause_Median  > {getattr(config, 'ARTIFACT_SILENCE_WARN_DB', -58.0):.0f} dBFS"
          f"  → WARN_SILENCE_FLOOR    (NEAR_MISS — present but subtle)")
    print(f"  Pause_Pass=N/A → no real pauses ≥160ms (short segment or continuous speech)")
    print(f"  SA_Combined   > {getattr(config, 'ARTIFACT_COMBINED_THRESHOLD', 0.15):.2f}"
          f"       → ERR_SPECTRAL_ARTIFACT (H1/H2 + cepstral + SFM_HF composite)")
    print(f"  SA_Combined in ({getattr(config, 'ARTIFACT_COMBINED_THRESHOLD', 0.15):.2f}, {getattr(config, 'ARTIFACT_COMBINED_THRESHOLD', 0.15) * (1 + getattr(config, 'ARTIFACT_NEAR_MISS_MARGIN', 0.20)):.2f}] → WARN_SA_BORDERLINE (NEAR_MISS)")


def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact / Vocoder Buzz Detector gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "artifact"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = load_model()
    if model_state is None:
        model_state = {}
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)

    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
