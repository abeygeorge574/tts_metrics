"""
Gate: Amplitude / Loudness
Env : base (python 3.13)
Uses pyloudnorm + librosa to measure LUFS, LRA, spectral centroid, and true peak.
Checks:
  - True peak absolute (always, no reference needed) — clipping
  - LUFS delta, LRA delta, spectral centroid delta (only on clean references)
Segments where reference LUFS is outside [-40, -5] are flagged as degraded.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import librosa
import pyloudnorm as pyln
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    # No ML model — pyloudnorm + librosa
    print("Amplitude gate ready (pyloudnorm + librosa, no model to load).")
    return None


# ── Audio analysis ─────────────────────────────────────────────────────────────
def analyze_audio(file_path):
    """
    Returns (LUFS, LRA, spectral_centroid, true_peak_dBFS).
    librosa loads as (channels, samples) — must transpose to
    (samples, channels) for pyloudnorm.
    """
    data, sr = librosa.load(file_path, sr=None, mono=False)

    if data.ndim == 1:
        data_pln = data.reshape(-1, 1)
    else:
        data_pln = data.T  # (channels, samples) → (samples, channels)

    meter = pyln.Meter(sr)
    lufs  = meter.integrated_loudness(data_pln)
    lra   = meter.loudness_range(data_pln)

    mono     = data if data.ndim == 1 else librosa.to_mono(data)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

    peak_amp = float(np.max(np.abs(data_pln)))
    peak_db  = 20 * np.log10(peak_amp) if peak_amp > 0 else -100.0

    return round(lufs, 3), round(lra, 3), round(centroid, 2), round(peak_db, 3)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    BASE_DIR      = config.AMPLITUDE_BASE_DIR
    REFERENCE_DIR = os.path.join(BASE_DIR, "reference")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")

    LUFS_TOLERANCE     = config.LUFS_TOLERANCE
    LRA_TOLERANCE      = config.LRA_TOLERANCE
    CENTROID_TOLERANCE = config.CENTROID_TOLERANCE
    PEAK_LIMIT         = config.PEAK_LIMIT
    REF_LUFS_MIN       = config.REF_LUFS_MIN
    REF_LUFS_MAX       = config.REF_LUFS_MAX

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
                ref_lufs, ref_lra, ref_cent, ref_peak = analyze_audio(ref_path)
                tts_lufs, tts_lra, tts_cent, tts_peak = analyze_audio(tts_path)

                print(f"  Ref    : LUFS={ref_lufs} | LRA={ref_lra} | "
                      f"Cent={ref_cent} | Peak={ref_peak}")
                print(f"  TTS    : LUFS={tts_lufs} | LRA={tts_lra} | "
                      f"Cent={tts_cent} | Peak={tts_peak}")

                ref_lufs_degraded = not (REF_LUFS_MIN <= ref_lufs <= REF_LUFS_MAX)
                is_degraded       = ref_lufs_degraded
                ref_flag          = "REF_LUFS_DEGRADED" if ref_lufs_degraded else "—"

                tts_clipping = tts_peak >= PEAK_LIMIT
                ref_clipping = ref_peak >= PEAK_LIMIT

                if tts_clipping and ref_clipping:
                    peak_flag = "REF_ALSO_CLIPPED"
                elif tts_clipping:
                    peak_flag = "TTS_CLIPPING"
                elif ref_clipping:
                    peak_flag = "REF_CLIPPED"
                else:
                    peak_flag = "—"

                lufs_diff = round(abs(ref_lufs - tts_lufs), 3)
                lra_diff  = round(abs(ref_lra  - tts_lra),  3)
                cent_diff = round(abs(ref_cent - tts_cent),  2)

                failures = []

                if tts_clipping:
                    failures.append("Clipping")

                if not is_degraded:
                    if lufs_diff > LUFS_TOLERANCE:
                        failures.append("Volume")
                    if lra_diff > LRA_TOLERANCE:
                        failures.append("Dynamics")
                    if cent_diff > CENTROID_TOLERANCE:
                        failures.append("EQ")

                final_pass = "PASS" if not failures else f"FAIL ({', '.join(failures)})"

                print(f"  Result : {final_pass} | Degraded: {is_degraded} | Peak: {peak_flag}")

                results.append({
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Ref LUFS"    : ref_lufs,
                    "TTS LUFS"    : tts_lufs,
                    "LUFS Delta"  : lufs_diff,
                    "Ref LRA"     : ref_lra,
                    "TTS LRA"     : tts_lra,
                    "LRA Delta"   : lra_diff,
                    "Ref Cent"    : ref_cent,
                    "TTS Cent"    : tts_cent,
                    "Cent Delta"  : cent_diff,
                    "Ref Peak"    : ref_peak,
                    "TTS Peak"    : tts_peak,
                    "Peak Flag"   : peak_flag,
                    "Final Pass"  : final_pass,
                    "Ref Flag"    : ref_flag,
                    "_is_degraded": is_degraded,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Ref LUFS"    : None,
                    "TTS LUFS"    : None,
                    "LUFS Delta"  : None,
                    "Ref LRA"     : None,
                    "TTS LRA"     : None,
                    "LRA Delta"   : None,
                    "Ref Cent"    : None,
                    "TTS Cent"    : None,
                    "Cent Delta"  : None,
                    "Ref Peak"    : None,
                    "TTS Peak"    : None,
                    "Peak Flag"   : "ERROR",
                    "Final Pass"  : "ERROR",
                    "Ref Flag"    : "ERROR",
                    "_is_degraded": False,
                })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"] & (model_df["Final Pass"] != "ERROR")]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total = len(clean_df)
        clean_pass  = (clean_df["Final Pass"] == "PASS").sum()

        deg_total   = len(degraded_df)
        deg_pass    = (degraded_df["Final Pass"] == "PASS").sum()

        clipping_count = model_df["Final Pass"].str.contains("Clipping").sum()
        volume_count   = model_df["Final Pass"].str.contains("Volume").sum()
        dynamics_count = model_df["Final Pass"].str.contains("Dynamics").sum()
        eq_count       = model_df["Final Pass"].str.contains("EQ").sum()
        error_count    = (model_df["Final Pass"] == "ERROR").sum()

        summary_rows.append({
            "Model"             : model,
            "Total Segments"    : total,
            "Clean Segments"    : clean_total,
            "Clean Pass Rate"   : f"{clean_pass}/{clean_total}"  if clean_total > 0 else "—",
            "Degraded Segments" : deg_total,
            "Degraded Pass Rate": f"{deg_pass}/{deg_total}"      if deg_total > 0 else "—",
            "Clipping Fails"    : clipping_count,
            "Volume Fails"      : volume_count,
            "Dynamics Fails"    : dynamics_count,
            "EQ Fails"          : eq_count,
            "Errors"            : error_count,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num", "Clipping Fails"],
        ascending=[False, False, True]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "Ref LUFS", "TTS LUFS", "LUFS Delta",
        "Ref LRA",  "TTS LRA",  "LRA Delta",
        "Ref Cent", "TTS Cent", "Cent Delta",
        "Ref Peak", "TTS Peak", "Peak Flag",
        "Final Pass", "Ref Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Degraded Pass Rate",
        "Clipping Fails", "Volume Fails", "Dynamics Fails", "EQ Fails"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate → primary ranking — ref LUFS within -40 to -5 only")
    print("Clipping Fails  → TTS peak >= -1.0 dBFS — always fails regardless of ref")
    print("Volume Fails    → LUFS delta > 4.0")
    print("Dynamics Fails  → LRA delta > 3.0")
    print("EQ Fails        → centroid delta > 500Hz")
    print(f"\nThresholds: LUFS ±{config.LUFS_TOLERANCE} | LRA ±{config.LRA_TOLERANCE} | "
          f"Centroid ±{config.CENTROID_TOLERANCE}Hz | Peak < {config.PEAK_LIMIT}dBFS")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amplitude gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "amplitude"))
    args = parser.parse_args()

    load_model()
    df, summary_df = run_gate()
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
