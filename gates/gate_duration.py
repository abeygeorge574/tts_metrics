"""
Gate: Duration Ratio
Env : base (python 3.13)
Checks that TTS duration is within ±10% of the reference duration.
Uses soundfile header reading (no decode).
"""

import os
import sys
import argparse

import soundfile as sf
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    # No model — soundfile reads WAV headers
    print("Duration gate ready (soundfile header read, no model to load).")
    return None


# ── Duration extraction ────────────────────────────────────────────────────────
def get_duration(file_path):
    """Fast duration extraction — reads WAV header only, no decode."""
    with sf.SoundFile(file_path) as f:
        return f.frames / f.samplerate


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    BASE_DIR      = config.DURATION_BASE_DIR
    REFERENCE_DIR = os.path.join(BASE_DIR, "reference")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")

    DURATION_TOLERANCE = config.DURATION_TOLERANCE
    lower_bound        = 1.0 - DURATION_TOLERANCE
    upper_bound        = 1.0 + DURATION_TOLERANCE

    for folder in [BASE_DIR, REFERENCE_DIR, MODELS_DIR]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
    print("Top level folders found.")

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
    for wav_file in sample_names:
        ref_path = os.path.join(REFERENCE_DIR, wav_file)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Missing reference for {wav_file} — expected: {ref_path}")
    print("All reference files found.")

    total = len(model_folders) * len(sample_names)
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = {total} evaluations")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            print(f"\n  Sample : {sample_name}")

            try:
                ref_dur = get_duration(ref_path)
                tts_dur = get_duration(tts_path)
                ratio   = round(tts_dur / ref_dur, 4) if ref_dur > 0 else None

                if ratio is None:
                    final_pass   = "ERROR"
                    failure_type = "Zero ref duration"
                elif ratio < lower_bound:
                    final_pass   = "FAIL"
                    failure_type = "Too Short"
                elif ratio > upper_bound:
                    final_pass   = "FAIL"
                    failure_type = "Too Long"
                else:
                    final_pass   = "PASS"
                    failure_type = "—"

                print(f"  Ref    : {round(ref_dur, 3)}s | TTS: {round(tts_dur, 3)}s | "
                      f"Ratio: {ratio} → {final_pass}")

                results.append({
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Ref Dur"     : round(ref_dur, 3),
                    "TTS Dur"     : round(tts_dur, 3),
                    "Ratio"       : ratio,
                    "Final Pass"  : final_pass,
                    "Failure Type": failure_type,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Ref Dur"     : None,
                    "TTS Dur"     : None,
                    "Ratio"       : None,
                    "Final Pass"  : "ERROR",
                    "Failure Type": str(e),
                })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df     = df[df["Model"] == model]
        total        = len(model_df)
        pass_count   = (model_df["Final Pass"] == "PASS").sum()
        short_count  = (model_df["Failure Type"] == "Too Short").sum()
        long_count   = (model_df["Failure Type"] == "Too Long").sum()
        error_count  = (model_df["Final Pass"] == "ERROR").sum()
        valid_ratios = model_df["Ratio"].dropna()

        summary_rows.append({
            "Model"       : model,
            "Segments"    : total,
            "Pass Rate"   : f"{pass_count}/{total}",
            "Too Short"   : short_count,
            "Too Long"    : long_count,
            "Errors"      : error_count,
            "Median Ratio": round(valid_ratios.median(), 4) if len(valid_ratios) > 0 else None,
            "Mean Ratio"  : round(valid_ratios.mean(), 4)   if len(valid_ratios) > 0 else None,
            "Min Ratio"   : round(valid_ratios.min(), 4)    if len(valid_ratios) > 0 else None,
            "Max Ratio"   : round(valid_ratios.max(), 4)    if len(valid_ratios) > 0 else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"]  = summary_df["Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df["_ratio_dev"] = summary_df["Median Ratio"].apply(
        lambda x: abs(x - 1.0) if x is not None else 999
    )

    summary_df = summary_df.sort_values(
        by=["_pass_num", "_ratio_dev", "Too Short"],
        ascending=[False, True, True]
    ).drop(columns=["_pass_num", "_ratio_dev"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Ref Dur", "TTS Dur", "Ratio", "Final Pass", "Failure Type"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Pass Rate", "Median Ratio", "Mean Ratio",
        "Too Short", "Too Long", "Errors"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Pass Rate    → % of segments within tolerance")
    print("Median Ratio → 1.0 = perfect | <0.9 too short | >1.1 too long")
    print("Too Short    → dialogue cut off — worse than too long for dubbing")
    print(f"\nTolerance: ±{int(config.DURATION_TOLERANCE * 100)}%")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duration ratio gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "duration"))
    args = parser.parse_args()

    load_model()
    df, summary_df = run_gate()
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
