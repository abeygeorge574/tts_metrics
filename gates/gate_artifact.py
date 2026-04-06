"""
Gate: Artifact / Vocoder Buzz Detector
Env : base (python 3.13)

Detects two types of audio artifacts in TTS/STS output:

1. Tonal / voiced artifacts  →  HNR (Harmonic-to-Noise Ratio)
   Metallic tonal buzz, vocoder overlay, non-harmonic resonance on voiced speech.
   Measured via parselmouth/Praat autocorrelation on voiced frames.
   Threshold: HNR_mean < ARTIFACT_HNR_ABS_THRESHOLD (8 dB) → ERR_VOICE_BUZZ
   Catches: SpeechT5-HiFiGAN, Griffin-Lim, Telugu STS tonal artifacts.
   Works on any segment length. No reference needed.

2. Background static in silence  →  median dBFS of real pause frames
   Broadband noise floor audible during pauses (vocoder residual, codec hiss).
   Real pause = contiguous run of frames ≥ 160 ms all below −30 dBFS.
   Threshold: median_pause_db > ARTIFACT_SILENCE_MEDIAN_DB (−55 dBFS) → ERR_BACKGROUND_STATIC
   Catches: FastSpeech2 broadband constant noise (median ~−44 dBFS).
   N/A when no real pauses are found (short segment, continuous speech).

Voting: any error code → FAIL.  No errors → PASS.
NEAR_MISS is not used — artifact presence is binary.

Reference: not used for gating. Thresholds are absolute and data-validated
against: gtts, kokoro, samantha (PASS), fastspeech2, HiFiGAN, Griffin-Lim,
Hindi STS, Telugu STS (FAIL).

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


# ── Gate interface ─────────────────────────────────────────────────────────────

def load_model():
    return {}   # no ML model to load beyond parselmouth (imported on demand)


def run_gate(model_state=None):
    import pandas as pd

    MODELS_DIR    = config.MODELS_DIR
    REFERENCE_DIR = config.REFERENCE_DIR   # not used for gating — kept for CSV info

    HNR_THRESHOLD    = getattr(config, "ARTIFACT_HNR_ABS_THRESHOLD",   8.0)
    PAUSE_DB_THRESH  = getattr(config, "ARTIFACT_SILENCE_MEDIAN_DB", -55.0)
    MIN_SEG_DUR      = getattr(config, "MIN_SEGMENT_DURATION",          2.0)

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
            flag = "SHORT_SEGMENT" if duration < MIN_SEG_DUR else ""

            # ── Step 1: HNR ──────────────────────────────────────────────────
            hnr_result = compute_hnr(tts_path)
            hnr_mean   = hnr_result.get("hnr_mean")
            hnr_error  = hnr_result.get("hnr_error")

            if hnr_mean is not None:
                hnr_pass = hnr_mean >= HNR_THRESHOLD
                hnr_str  = f"{hnr_mean:.1f} dB (voiced={hnr_result['hnr_voiced_frac']:.0%})"
                print(f"  HNR    : {hnr_str}  → {'PASS' if hnr_pass else 'FAIL  ← ERR_VOICE_BUZZ'}")
            else:
                hnr_pass = None
                print(f"  HNR    : unavailable ({hnr_error})")

            # ── Step 2 + 3: Pause existence + median dBFS ────────────────────
            pause_result = compute_pause_median_db(tts_path)

            if pause_result["silence_na"]:
                pause_pass = None
                median_db  = None
                print(f"  Pause  : no real pauses ≥160ms detected  → N/A")
            else:
                median_db  = pause_result["median_db"]
                pause_pass = median_db <= PAUSE_DB_THRESH
                print(f"  Pause  : {pause_result['n_real_pauses']} pauses "
                      f"({pause_result['real_pause_frac']:.0%} of audio)  "
                      f"median={median_db:.1f} dBFS  "
                      f"→ {'PASS' if pause_pass else 'FAIL  ← ERR_BACKGROUND_STATIC'}")

            # ── Voting ────────────────────────────────────────────────────────
            error_flags = []
            if hnr_mean is not None and not hnr_pass:
                error_flags.append("ERR_VOICE_BUZZ")
            if pause_pass is not None and not pause_pass:
                error_flags.append("ERR_BACKGROUND_STATIC")

            if error_flags:
                result = "FAIL"
            elif hnr_mean is None and pause_pass is None:
                result = "SKIP"      # no metrics could run
            else:
                result = "PASS"

            flags_str = ",".join(error_flags) if error_flags else ""
            if flag:
                flags_str = f"{flag},{flags_str}" if flags_str else flag

            print(f"  Result : {result}" + (f"  [{flags_str}]" if flags_str else ""))

            results.append({
                "Model"           : model,
                "Sample"          : sample_name,
                "Duration_s"      : round(duration, 2),
                # HNR
                "HNR_Mean"        : hnr_mean,
                "HNR_Voiced_Frac" : hnr_result.get("hnr_voiced_frac"),
                "HNR_Pass"        : ("PASS" if hnr_pass else "FAIL") if hnr_pass is not None else "N/A",
                # Pause / median dBFS
                "Real_Pauses"     : pause_result["n_real_pauses"],
                "Pause_Frac"      : pause_result["real_pause_frac"],
                "Pause_Median_DB" : median_db,
                "Pause_Pass"      : ("PASS" if pause_pass else "FAIL") if pause_pass is not None else "N/A",
                # Overall
                "Error_Flags"     : flags_str,
                "Pass"            : result,
                "Flag"            : flag,
            })

    df = pd.DataFrame(results)

    # Summary
    summary_rows = []
    for model in model_folders:
        mdf = df[df["Model"] == model]
        pass_count = (mdf["Pass"] == "PASS").sum()
        fail_count = (mdf["Pass"] == "FAIL").sum()
        summary_rows.append({
            "Model"           : model,
            "Pass_Rate"       : f"{pass_count}/{len(mdf)}",
            "Fail"            : int(fail_count),
            "Mean_HNR_dB"     : round(mdf["HNR_Mean"].dropna().mean(), 2)
                                 if mdf["HNR_Mean"].notna().any() else None,
            "Mean_Pause_dBFS" : round(mdf["Pause_Median_DB"].dropna().mean(), 2)
                                 if mdf["Pause_Median_DB"].notna().any() else None,
            "Error_Flags"     : "; ".join(
                                 f for f in mdf["Error_Flags"].dropna() if f
                                 ) or "—",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Fail", ascending=False)

    return df, summary_df


def print_results(df, summary_df):
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n========== PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Duration_s",
        "HNR_Mean", "HNR_Pass",
        "Real_Pauses", "Pause_Median_DB", "Pause_Pass",
        "Error_Flags", "Pass", "Flag",
    ]].to_string(index=False))

    print("\n========== MODEL SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== GUIDE ==========")
    print(f"  HNR_Mean      < {getattr(config, 'ARTIFACT_HNR_ABS_THRESHOLD', 8.0):.0f} dB"
          f"   → ERR_VOICE_BUZZ       (tonal/metallic buzz on voiced speech)")
    print(f"  Pause_Median  > {getattr(config, 'ARTIFACT_SILENCE_MEDIAN_DB', -55.0):.0f} dBFS"
          f"  → ERR_BACKGROUND_STATIC (broadband noise floor in pauses)")
    print(f"  Pause_Pass=N/A → no real pauses ≥160ms (short segment or continuous speech)")


def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artifact / Vocoder Buzz Detector gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "artifact"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
