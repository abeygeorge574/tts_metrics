"""
Gate: Pitch (median register + expressiveness)
Env : base (python 3.13)
Uses librosa pyin to extract pitch. Checks:
  1. Absolute std floor (expressiveness minimum, no reference needed)
  2. Std ratio vs reference (relative expressiveness)
  3. Median delta vs reference (pitch register / zone)
Segments where reference voiced ratio < 0.2 are flagged as degraded.
"""

import os
import sys
import argparse

import numpy as np
import librosa
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    # No model to load — librosa is a library function
    print("Pitch gate ready (librosa pyin, no model to load).")
    return None


# ── Pitch computation ──────────────────────────────────────────────────────────
def compute_pitch(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        voiced_f0 = f0[voiced_flag]

        if len(voiced_f0) == 0:
            print(f"  No voiced frames: {os.path.basename(audio_path)}")
            return None, None, 0.0

        pitch_median = round(float(np.median(voiced_f0)), 2)
        pitch_std    = round(float(np.std(voiced_f0)), 2)
        voiced_ratio = round(float(np.sum(voiced_flag) / len(voiced_flag)), 3)

        return pitch_median, pitch_std, voiced_ratio

    except Exception as e:
        print(f"  Pitch error: {e}")
        return None, None, None


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    BASE_DIR      = config.PITCH_BASE_DIR
    REFERENCE_DIR = os.path.join(BASE_DIR, "reference")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")

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
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)

            print(f"\n  Sample : {sample_name}")

            tts_median, tts_std, tts_voiced_ratio = compute_pitch(tts_path)
            print(f"  TTS    : median={tts_median}Hz | std={tts_std}Hz | voiced={tts_voiced_ratio}")

            ref_median       = None
            ref_std          = None
            ref_voiced_ratio = None
            median_delta     = None
            std_ratio        = None
            median_pass      = None
            std_ratio_pass   = None
            ref_flag         = "NO_REF"
            is_degraded      = False

            if reference_available:
                ref_path = os.path.join(REFERENCE_DIR, wav_file)
                if os.path.exists(ref_path):
                    ref_median, ref_std, ref_voiced_ratio = compute_pitch(ref_path)
                    print(f"  Ref    : median={ref_median}Hz | std={ref_std}Hz | voiced={ref_voiced_ratio}")

                    if ref_voiced_ratio is not None and ref_voiced_ratio < 0.2:
                        ref_flag    = "REF_UNVOICED"
                        is_degraded = True
                        print(f"  Reference voiced ratio < 0.2 — delta checks unreliable")
                    else:
                        ref_flag = "—"

                        if ref_median is not None and tts_median is not None:
                            median_delta = round(abs(ref_median - tts_median), 2)
                            median_pass  = median_delta <= PITCH_MEDIAN_THRESHOLD

                        if ref_std is not None and tts_std is not None and ref_std > 0:
                            std_ratio      = round(tts_std / ref_std, 3)
                            std_ratio_pass = std_ratio >= PITCH_STD_RATIO_THRESHOLD

            # absolute std floor — always runs, no reference needed
            if tts_std is not None:
                std_abs_pass = tts_std >= PITCH_STD_ABS_THRESHOLD
            else:
                std_abs_pass = None

            # final pass/fail
            if tts_median is None:
                final_pass = "ERROR"
            else:
                failures = []
                if std_abs_pass is False:
                    failures.append("Flat (abs)")
                if std_ratio_pass is False:
                    failures.append("Flat (vs ref)")
                if median_pass is False:
                    failures.append("Register")
                final_pass = "PASS" if not failures else f"FAIL ({', '.join(failures)})"

            print(f"  Result : {final_pass} | Median Δ: {median_delta} | "
                  f"Std Ratio: {std_ratio} | Degraded: {is_degraded}")

            results.append({
                "Model"         : model,
                "Sample"        : sample_name,
                "TTS Median"    : tts_median,
                "TTS Std"       : tts_std,
                "TTS Voiced"    : tts_voiced_ratio,
                "Ref Median"    : ref_median,
                "Ref Std"       : ref_std,
                "Ref Voiced"    : ref_voiced_ratio,
                "Median Delta"  : median_delta,
                "Std Ratio"     : std_ratio,
                "Std Abs Pass"  : "PASS" if std_abs_pass else "FAIL" if std_abs_pass is not None else "—",
                "Std Ratio Pass": "PASS" if std_ratio_pass else "FAIL" if std_ratio_pass is not None else "—",
                "Median Pass"   : "PASS" if median_pass else "FAIL" if median_pass is not None else "—",
                "Final Pass"    : final_pass,
                "Ref Flag"      : ref_flag,
                "_is_degraded"  : is_degraded,
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

        mean_tts_std   = round(model_df["TTS Std"].dropna().mean(), 2)
        mean_std_ratio = (
            round(model_df["Std Ratio"].dropna().mean(), 3)
            if model_df["Std Ratio"].notna().any() else None
        )
        mean_delta = (
            round(model_df["Median Delta"].dropna().mean(), 2)
            if model_df["Median Delta"].notna().any() else None
        )

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
            "Mean TTS Std"      : mean_tts_std,
            "Mean Std Ratio"    : mean_std_ratio,
            "Mean Median Delta" : mean_delta,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df["_mean_std_ratio"]    = summary_df["Mean Std Ratio"].fillna(-999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num", "_mean_std_ratio"],
        ascending=[False, False, False]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num", "_mean_std_ratio"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "TTS Median", "TTS Std", "TTS Voiced",
        "Ref Median", "Ref Std", "Ref Voiced",
        "Median Delta", "Std Ratio",
        "Std Abs Pass", "Std Ratio Pass", "Median Pass",
        "Final Pass", "Ref Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Degraded Pass Rate",
        "Flat Abs Fails", "Flat Ratio Fails", "Register Fails",
        "Mean TTS Std", "Mean Std Ratio", "Mean Median Delta"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate   → primary ranking — ref voiced ratio >= 0.2 only")
    print("Flat Abs Fails    → TTS std below floor — robotic regardless of reference")
    print("Register Fails    → TTS pitch zone wrong — fix voice clone")
    print(f"\nThresholds: Median Δ <= {config.PITCH_MEDIAN_THRESHOLD}Hz | "
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
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "pitch"))
    args = parser.parse_args()

    load_model()
    df, summary_df = run_gate()
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
