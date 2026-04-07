"""
generate_poison.py
------------------
Inject known artifacts at controlled severity levels into clean TTS audio.

Sources: data/models/gtts/sample_1.wav (primary)
Output:  data/models_poison/<model_name>/sample_1.wav

Artifact types:
  1. H2_BOOST       — parametric EQ boost at 2*F0 (flatten harmonic slope)
  2. HF_NOISE_VOICED — broadband noise 4-8 kHz in voiced frames
  3. SILENCE_HISS   — broadband noise in silence frames
  4. TONAL_BUZZ     — periodic non-harmonic tone at 1.5*F0 in voiced frames
"""

import os
import sys
import shutil
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import iirpeak, sosfilt, butter, sosfiltfilt

ROOT        = "/Users/abey/Documents/tts_metrics"
SOURCE_WAV  = os.path.join(ROOT, "data", "models", "gtts", "sample_1.wav")
OUTPUT_BASE = os.path.join(ROOT, "data", "models_poison")
SR          = 16000


# ── helpers ────────────────────────────────────────────────────────────────────

def load_source() -> tuple[np.ndarray, int]:
    y, sr = librosa.load(SOURCE_WAV, sr=SR)
    print(f"Loaded source: {SOURCE_WAV}  ({len(y)/SR:.2f}s, sr={SR})")
    return y, sr


def save_wav(y: np.ndarray, sr: int, model_name: str) -> str:
    out_dir  = os.path.join(OUTPUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_1.wav")
    # Clip to prevent clipping distortion
    y_out = np.clip(y, -1.0, 1.0)
    sf.write(out_path, y_out, sr)
    return out_path


def get_median_f0(y: np.ndarray, sr: int) -> float:
    """Estimate median fundamental frequency using pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=60, fmax=400, sr=sr,
        frame_length=1024, hop_length=160,
    )
    valid = voiced_flag & ~np.isnan(f0)
    if valid.sum() < 5:
        return 150.0  # fallback
    return float(np.median(f0[valid]))


def voiced_frame_mask(y: np.ndarray, sr: int,
                      frame_sz: int = 512, hop_sz: int = 256,
                      rms_thresh_db: float = -25.0) -> np.ndarray:
    """Return per-sample boolean mask: True where rms_db > threshold (voiced)."""
    frames = librosa.util.frame(y, frame_length=frame_sz, hop_length=hop_sz)
    rms    = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db = 20 * np.log10(rms + 1e-10)
    voiced = rms_db > rms_thresh_db

    # Expand per-frame mask back to per-sample
    mask = np.zeros(len(y), dtype=bool)
    for i, v in enumerate(voiced):
        start = i * hop_sz
        end   = min(start + frame_sz, len(y))
        if v:
            mask[start:end] = True
    return mask


def silence_frame_mask(y: np.ndarray, sr: int,
                       frame_sz: int = 512, hop_sz: int = 256,
                       rms_thresh_db: float = -30.0) -> np.ndarray:
    """Return per-sample boolean mask: True where rms_db < threshold (silence)."""
    frames = librosa.util.frame(y, frame_length=frame_sz, hop_length=hop_sz)
    rms    = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db = 20 * np.log10(rms + 1e-10)
    silent = rms_db < rms_thresh_db

    mask = np.zeros(len(y), dtype=bool)
    for i, v in enumerate(silent):
        start = i * hop_sz
        end   = min(start + frame_sz, len(y))
        if v:
            mask[start:end] = True
    return mask


# ── Artifact 1: H2_BOOST ───────────────────────────────────────────────────────

def peaking_eq(y: np.ndarray, sr: int, center_hz: float, gain_db: float,
               Q: float = 5.0) -> np.ndarray:
    """Apply an IIR peaking EQ filter (gain in dB) at center_hz."""
    # iirpeak returns (b, a) but we need second-order sections for numerical stability
    w0    = center_hz / (sr / 2)          # normalized angular freq (0..1)
    gain  = 10 ** (gain_db / 20)
    # iirpeak is a notch; for boost we use a standard parametric shelf approach:
    # biquad peaking EQ coefficients (Audio EQ Cookbook)
    A     = gain
    w0_r  = 2 * np.pi * center_hz / sr   # radians/sample
    alpha = np.sin(w0_r) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0_r)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0_r)
    a2 = 1 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0,     a1 / a0, a2 / a0])

    from scipy.signal import lfilter
    return lfilter(b, a, y).astype(np.float32)


def inject_h2_boost(y: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    """Boost 2nd harmonic (2*F0) by gain_db dB using a peaking EQ filter."""
    f0  = get_median_f0(y, sr)
    f2  = 2 * f0
    print(f"  Median F0={f0:.1f} Hz  →  H2 at {f2:.1f} Hz, boost={gain_db:+.0f} dB")
    return peaking_eq(y, sr, center_hz=f2, gain_db=gain_db, Q=5.0)


# ── Artifact 2: HF_NOISE_VOICED ───────────────────────────────────────────────

def bandpass_noise(n_samples: int, sr: int,
                   lo: float = 4000, hi: float = 7800) -> np.ndarray:
    """Generate white noise bandpass-filtered to [lo, hi] Hz.
    Upper cutoff is 7800 Hz to stay safely below Nyquist at 16 kHz (fs/2=8000).
    """
    noise = np.random.randn(n_samples).astype(np.float32)
    sos   = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, noise).astype(np.float32)


def inject_hf_noise_voiced(y: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    """
    Add bandpass (4-8 kHz) noise to voiced frames at specified SNR.
    snr_db < 0 means noise is louder: e.g. -20 dB → noise is 10× quieter than signal
    but still very audible.
    """
    voiced = voiced_frame_mask(y, sr)
    noise  = bandpass_noise(len(y), sr)

    # Scale noise so SNR = snr_db relative to voiced signal power
    y_voiced = y[voiced]
    if len(y_voiced) == 0:
        return y.copy()

    sig_rms   = np.sqrt(np.mean(y_voiced ** 2))
    noise_rms = np.sqrt(np.mean(noise[voiced] ** 2)) + 1e-10
    # target_noise_rms = sig_rms * 10^(snr_db/20)
    # snr_db = -20 → noise at 0.1 × signal power = much lower level but in HF band
    target_rms = sig_rms * (10 ** (snr_db / 20))
    noise_scaled = noise * (target_rms / noise_rms)

    out = y.copy()
    out[voiced] = out[voiced] + noise_scaled[voiced]
    print(f"  Voiced SNR={snr_db:.0f} dB  sig_rms={sig_rms:.5f}  "
          f"noise_rms={target_rms:.5f}  voiced_frac={voiced.mean():.2f}")
    return out


# ── Artifact 3: SILENCE_HISS ─────────────────────────────────────────────────

def inject_silence_hiss(y: np.ndarray, sr: int, level_dbfs: float) -> np.ndarray:
    """
    Add white noise at a fixed absolute level (dBFS) to silence frames.
    level_dbfs = -40 means noise RMS = 10^(-40/20) = 0.01 (full scale = 1.0)
    """
    silent = silence_frame_mask(y, sr, rms_thresh_db=-30.0)

    noise     = np.random.randn(len(y)).astype(np.float32)
    # Normalise noise to target dBFS
    target_rms = 10 ** (level_dbfs / 20)
    noise_rms  = np.sqrt(np.mean(noise ** 2)) + 1e-10
    noise_scaled = noise * (target_rms / noise_rms)

    out = y.copy()
    out[silent] = out[silent] + noise_scaled[silent]
    print(f"  Silence hiss level={level_dbfs:.0f} dBFS  "
          f"target_rms={target_rms:.5f}  silence_frac={silent.mean():.2f}")
    return out


# ── Artifact 4: TONAL_BUZZ ─────────────────────────────────────────────────────

def inject_tonal_buzz(y: np.ndarray, sr: int, rel_db: float) -> np.ndarray:
    """
    Add a pure tone at 1.5 * F0 (non-harmonic frequency) during voiced frames.
    rel_db is relative to voiced signal RMS: e.g. -20 dB → tone is 10× quieter.
    """
    f0     = get_median_f0(y, sr)
    f_buzz = 1.5 * f0
    print(f"  Median F0={f0:.1f} Hz  →  buzz at {f_buzz:.1f} Hz, rel={rel_db:+.0f} dB")

    voiced = voiced_frame_mask(y, sr)
    t      = np.arange(len(y), dtype=np.float32) / sr
    tone   = np.sin(2 * np.pi * f_buzz * t).astype(np.float32)

    # Scale tone relative to voiced signal RMS
    y_voiced = y[voiced]
    if len(y_voiced) == 0:
        return y.copy()
    sig_rms   = np.sqrt(np.mean(y_voiced ** 2))
    target_rms = sig_rms * (10 ** (rel_db / 20))
    tone_rms   = np.sqrt(np.mean(tone[voiced] ** 2)) + 1e-10
    tone_scaled = tone * (target_rms / tone_rms)

    out = y.copy()
    out[voiced] = out[voiced] + tone_scaled[voiced]
    print(f"  Tone RMS (target)={target_rms:.5f}  voiced_frac={voiced.mean():.2f}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    y, sr = load_source()

    # ── Baseline: copy clean gtts/sample_1.wav ──────────────────────────────
    print("\n[Baseline] Copying clean audio as gtts_clean/sample_1.wav")
    baseline_dir = os.path.join(OUTPUT_BASE, "gtts_clean")
    os.makedirs(baseline_dir, exist_ok=True)
    shutil.copy2(SOURCE_WAV, os.path.join(baseline_dir, "sample_1.wav"))
    print(f"  Saved → {baseline_dir}/sample_1.wav")

    # ── H2_BOOST ─────────────────────────────────────────────────────────────
    for gain_db, name in [(3, "h2boost_3db"), (6, "h2boost_6db"),
                          (9, "h2boost_9db"), (12, "h2boost_12db")]:
        print(f"\n[H2_BOOST] {name}")
        y_out = inject_h2_boost(y, sr, gain_db=gain_db)
        path  = save_wav(y_out, sr, name)
        print(f"  Saved → {path}")

    # ── HF_NOISE_VOICED ───────────────────────────────────────────────────────
    for snr_db, name in [(-20, "hf_noise_m20db"), (-30, "hf_noise_m30db"),
                         (-40, "hf_noise_m40db"), (-50, "hf_noise_m50db")]:
        print(f"\n[HF_NOISE_VOICED] {name}")
        y_out = inject_hf_noise_voiced(y, sr, snr_db=snr_db)
        path  = save_wav(y_out, sr, name)
        print(f"  Saved → {path}")

    # ── SILENCE_HISS ─────────────────────────────────────────────────────────
    for level_dbfs, name in [(-40, "sil_hiss_m40"), (-45, "sil_hiss_m45"),
                              (-50, "sil_hiss_m50"), (-55, "sil_hiss_m55"),
                              (-60, "sil_hiss_m60")]:
        print(f"\n[SILENCE_HISS] {name}")
        y_out = inject_silence_hiss(y, sr, level_dbfs=level_dbfs)
        path  = save_wav(y_out, sr, name)
        print(f"  Saved → {path}")

    # ── TONAL_BUZZ ────────────────────────────────────────────────────────────
    for rel_db, name in [(-20, "tonal_m20db"), (-30, "tonal_m30db"),
                         (-40, "tonal_m40db"), (-50, "tonal_m50db")]:
        print(f"\n[TONAL_BUZZ] {name}")
        y_out = inject_tonal_buzz(y, sr, rel_db=rel_db)
        path  = save_wav(y_out, sr, name)
        print(f"  Saved → {path}")

    print("\n\nAll poisoned samples generated successfully.")
    models = os.listdir(OUTPUT_BASE)
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Models generated ({len(models)}): {sorted(models)}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
