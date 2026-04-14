"""
Gate: Pitch (median register + expressiveness)
Env : base (python 3.13)
Uses PRAAT (via parselmouth) to extract F0. Checks:
  1. Absolute std floor (expressiveness minimum, no reference needed)
  2. Std ratio vs reference (relative expressiveness)
  3. Median delta vs reference (pitch register / zone)
Segments where reference voiced ratio < 0.2 are flagged as degraded.

Why PRAAT over pyin:
  Isolated comparison (notebooks/pitch_estimator_comparison.py) showed PRAAT
  gives deltas most consistent with perceptual listening. pyin has octave errors
  on expressive voices (chatterbox: 45 Hz FAIL → PRAAT: 21 Hz PASS, matches ear).
  pyin+voiced_probs filter produces absurd values (1500-2000 Hz) on high-std voices.
  CREPE agrees with PRAAT on stable segments but diverges on ambiguous references.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    try:
        import parselmouth  # noqa: F401 — confirm available at startup
        print("Pitch gate ready (PRAAT via parselmouth).")
    except ImportError:
        print("WARNING: parselmouth not installed. Install with: pip install praat-parselmouth")
    return None


# ── Pitch computation ──────────────────────────────────────────────────────────
def compute_pitch(audio_path):
    """
    Returns (pitch_median_hz, pitch_std_hz, voiced_ratio) using PRAAT.
    voiced_ratio = fraction of total frames where PRAAT detected pitch.
    Falls back to (None, None, None) on error.
    """
    try:
        import parselmouth

        snd   = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch(
            time_step=0.01,    # 10 ms frames
            pitch_floor=60.0,  # Hz — covers low male voices
            pitch_ceiling=600.0,  # Hz — covers high female voices
        )

        f0_values    = pitch.selected_array["frequency"]  # 0 = unvoiced frame
        total_frames = len(f0_values)
        voiced_f0    = f0_values[f0_values > 0]

        if len(voiced_f0) == 0:
            print(f"  No voiced frames: {os.path.basename(audio_path)}")
            return None, None, 0.0

        # Mean (not median) is used because PRAAT's autocorrelation algorithm
        # handles octave ambiguity internally — voiced frames are clean.
        # Mean captures genuine sustained pitch elevation that median would centre away.
        pitch_mean   = round(float(np.mean(voiced_f0)), 2)
        pitch_std    = round(float(np.std(voiced_f0)), 2)
        pitch_range  = round(float(np.max(voiced_f0) - np.min(voiced_f0)), 2)
        voiced_ratio = round(len(voiced_f0) / total_frames, 3)

        return pitch_mean, pitch_std, pitch_range, voiced_ratio

    except Exception as e:
        print(f"  Pitch error: {e}")
        return None, None, None


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = {}
    MODELS_DIR    = model_state.get("models_dir")  or config.MODELS_DIR
    REFERENCE_DIR = model_state.get("ref_dir")     or config.REFERENCE_DIR

    PITCH_MEDIAN_THRESHOLD    = config.PITCH_MEDIAN_THRESHOLD
    PITCH_STD_ABS_THRESHOLD   = config.PITCH_STD_ABS_THRESHOLD
    PITCH_STD_RATIO_THRESHOLD = config.PITCH_STD_RATIO_THRESHOLD

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    reference_available = os.path.exists(REFERENCE_DIR)
    if reference_available:
        ref_files = sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".wav")])
        print(f"Reference folder found: {len(ref_files)} files")
    else:
        print("No reference folder — median delta and std ratio disabled.")
        print("Only absolute std floor check will run.")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models found: {model_folders}")

    model_samples = {}
    for model in model_folders:
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])
        model_samples[model] = wav_files
        print(f"   {model}: {len(wav_files)} samples")

    reference_filenames = set(model_samples[model_folders[0]])
    for model in model_folders[1:]:
        current_filenames = set(model_samples[model])
        if current_filenames != reference_filenames:
            missing = reference_filenames - current_filenames
            extra   = current_filenames - reference_filenames
            raise ValueError(
                f"Model '{model}' has mismatched filenames.\n"
                f"  Missing : {missing}\n"
                f"  Extra   : {extra}"
            )
    print("All models have identical filenames.")

    sample_names = model_samples[model_folders[0]]
    total        = len(model_folders) * len(sample_names)
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = {total} evaluations")
    print(f"Reference: {'available' if reference_available else 'not available'}")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)

            duration = sf.info(tts_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            if is_short:
                print(f"\n  Sample : {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample : {sample_name}")

            tts_mean, tts_std, tts_range, tts_voiced_ratio = compute_pitch(tts_path)
            print(f"  TTS    : mean={tts_mean}Hz | std={tts_std}Hz | range={tts_range}Hz | voiced={tts_voiced_ratio}")

            ref_mean         = None
            ref_std          = None
            ref_range        = None
            ref_voiced_ratio = None
            mean_delta       = None
            std_ratio        = None
            range_ratio      = None
            mean_pass        = None
            std_ratio_pass   = None
            ref_flag         = "NO_REF"
            is_degraded      = False

            if reference_available:
                ref_path = os.path.join(REFERENCE_DIR, wav_file)
                if os.path.exists(ref_path):
                    ref_mean, ref_std, ref_range, ref_voiced_ratio = compute_pitch(ref_path)
                    print(f"  Ref    : mean={ref_mean}Hz | std={ref_std}Hz | range={ref_range}Hz | voiced={ref_voiced_ratio}")

                    if ref_voiced_ratio is not None and ref_voiced_ratio < 0.2:
                        ref_flag    = "REF_UNVOICED"
                        is_degraded = True
                        print(f"  Reference voiced ratio < 0.2 — delta checks unreliable")
                    else:
                        ref_flag = "—"

                        if ref_mean is not None and tts_mean is not None:
                            mean_delta = round(abs(ref_mean - tts_mean), 2)
                            mean_pass  = mean_delta <= PITCH_MEDIAN_THRESHOLD

                        if ref_std is not None and tts_std is not None and ref_std > 0:
                            std_ratio      = round(tts_std / ref_std, 3)
                            std_ratio_pass = std_ratio >= PITCH_STD_RATIO_THRESHOLD

                        if ref_range is not None and tts_range is not None and ref_range > 0:
                            range_ratio = round(tts_range / ref_range, 3)

            # absolute std floor — always runs, no reference needed
            if tts_std is not None:
                std_abs_pass = tts_std >= PITCH_STD_ABS_THRESHOLD
            else:
                std_abs_pass = None

            # final pass/fail
            if tts_mean is None:
                final_pass = "ERROR"
            else:
                failures = []
                if std_abs_pass is False:
                    failures.append("Flat (abs)")
                if std_ratio_pass is False:
                    failures.append("Flat (vs ref)")
                if mean_pass is False:
                    failures.append("Register")
                final_pass = "PASS" if not failures else f"FAIL ({', '.join(failures)})"

            print(f"  Result : {final_pass} | Mean Δ: {mean_delta} | "
                  f"Std Ratio: {std_ratio} | Degraded: {is_degraded}")

            results.append({
                "Model"         : model,
                "Sample"        : sample_name,
                "TTS Mean"      : tts_mean,
                "TTS Std"       : tts_std,
                "TTS Range"     : tts_range,
                "TTS Voiced"    : tts_voiced_ratio,
                "Ref Mean"      : ref_mean,
                "Ref Std"       : ref_std,
                "Ref Range"     : ref_range,
                "Ref Voiced"    : ref_voiced_ratio,
                "Mean Delta"    : mean_delta,
                "Std Ratio"     : std_ratio,
                "Range Ratio"   : range_ratio,
                "Std Abs Pass"  : "PASS" if std_abs_pass else "FAIL" if std_abs_pass is not None else "—",
                "Std Ratio Pass": "PASS" if std_ratio_pass else "FAIL" if std_ratio_pass is not None else "—",
                "Mean Pass"     : "PASS" if mean_pass else "FAIL" if mean_pass is not None else "—",
                "Final Pass"    : final_pass,
                "Ref Flag"      : ref_flag,
                "_is_degraded"  : is_degraded or is_short,
                "Flag"          : "SHORT_SEGMENT" if is_short else ref_flag,
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"]]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total = len(clean_df)
        clean_pass  = (clean_df["Final Pass"] == "PASS").sum()

        deg_total   = len(degraded_df)
        deg_pass    = (degraded_df["Final Pass"] == "PASS").sum()

        flat_abs_count   = model_df["Final Pass"].str.contains("Flat \\(abs\\)").sum()
        flat_ratio_count = model_df["Final Pass"].str.contains("Flat \\(vs ref\\)").sum()
        register_count   = model_df["Final Pass"].str.contains("Register").sum()

        deltas = model_df["Mean Delta"].dropna()
        med_tts_std   = round(model_df["TTS Std"].dropna().median(), 2)
        med_tts_range = round(model_df["TTS Range"].dropna().median(), 2) if "TTS Range" in model_df else None
        med_std_ratio = (
            round(model_df["Std Ratio"].dropna().median(), 3)
            if model_df["Std Ratio"].notna().any() else None
        )
        med_range_ratio = (
            round(model_df["Range Ratio"].dropna().median(), 3)
            if "Range Ratio" in model_df and model_df["Range Ratio"].notna().any() else None
        )
        med_delta = round(deltas.median(), 2) if len(deltas) > 0 else None
        max_delta = round(deltas.max(), 2)    if len(deltas) > 0 else None

        summary_rows.append({
            "Model"             : model,
            "Total Segments"    : total,
            "Clean Segments"    : clean_total,
            "Clean Pass Rate"   : f"{clean_pass}/{clean_total}"  if clean_total > 0 else "—",
            "Degraded Segments" : deg_total,
            "Degraded Pass Rate": f"{deg_pass}/{deg_total}"      if deg_total > 0 else "—",
            "Flat Abs Fails"    : flat_abs_count,
            "Flat Ratio Fails"  : flat_ratio_count,
            "Register Fails"    : register_count,
            "Median TTS Std"    : med_tts_std,
            "Median TTS Range"  : med_tts_range,
            "Median Std Ratio"  : med_std_ratio,
            "Median Range Ratio": med_range_ratio,
            "Median Δ (mean/seg)": med_delta,
            "Max Δ"             : max_delta,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df["_med_std_ratio"]     = summary_df["Median Std Ratio"].fillna(-999)
    summary_df["_med_delta"]         = summary_df["Median Δ (mean/seg)"].fillna(9999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num", "_med_std_ratio", "_med_delta"],
        ascending=[False, False, False, True]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num", "_med_std_ratio", "_med_delta"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "TTS Mean", "TTS Std", "TTS Range", "TTS Voiced",
        "Ref Mean", "Ref Std", "Ref Range", "Ref Voiced",
        "Mean Delta", "Std Ratio", "Range Ratio",
        "Std Abs Pass", "Std Ratio Pass", "Mean Pass",
        "Final Pass", "Ref Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Degraded Pass Rate",
        "Flat Abs Fails", "Flat Ratio Fails", "Register Fails",
        "Median TTS Std", "Median TTS Range",
        "Median Std Ratio", "Median Range Ratio",
        "Median Δ (mean/seg)", "Max Δ"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate      → primary ranking — ref voiced ratio >= 0.2 only")
    print("Flat Abs Fails       → TTS std below floor — robotic regardless of reference")
    print("Flat Ratio Fails     → TTS flat relative to reference (std ratio < 0.5×)")
    print("Register Fails       → TTS pitch zone wrong vs reference (mean delta > 30 Hz)")
    print("Median TTS Range     → F0 max−min across voiced frames: higher = more expressive pitch")
    print("Median Range Ratio   → TTS range / ref range: < 1 = TTS narrower pitch excursion than ref")
    print("Median Δ (mean/seg)  → median across segments of per-segment mean F0 delta")
    print("Max Δ                → worst-case segment delta")
    print(f"\nEstimator: PRAAT (parselmouth). Mean F0 per segment, median across segments.")
    print(f"Thresholds: Mean Δ <= {config.PITCH_MEDIAN_THRESHOLD}Hz | "
          f"Std abs >= {config.PITCH_STD_ABS_THRESHOLD}Hz | "
          f"Std ratio >= {config.PITCH_STD_RATIO_THRESHOLD}x ref")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "pitch"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = {}
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)

    load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
