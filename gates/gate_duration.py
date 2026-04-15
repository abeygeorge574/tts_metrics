"""
Gate: Duration Ratio
Env : base (python 3.13)
Checks that TTS duration is within ±10% of the reference duration.
Uses soundfile header reading (no decode).

Pass band: [0.90, 1.10]
FAIL: ratio outside pass band (Too Short or Too Long)
NO_REF: reference file missing — ratio check skipped, TTS duration recorded only
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
    print("Duration gate ready (soundfile header read, no model to load).")
    return None


# ── Duration extraction ────────────────────────────────────────────────────────
def get_duration(file_path):
    """Fast duration extraction — reads WAV header only, no decode."""
    with sf.SoundFile(file_path) as f:
        return f.frames / f.samplerate


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    MODELS_DIR    = (model_state or {}).get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = (model_state or {}).get("ref_dir")    or config.REFERENCE_DIR

    lower_bound = 1.0 - config.DURATION_TOLERANCE   # 0.90
    upper_bound = 1.0 + config.DURATION_TOLERANCE   # 1.10

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Folder not found: {MODELS_DIR}")
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

    ref_available = os.path.exists(REFERENCE_DIR)
    if not ref_available:
        print("WARNING: Reference directory not found — all segments will be NO_REF")

    total = len(model_folders) * len(model_samples[model_folders[0]])
    print(f"\nReady: {len(model_folders)} models × {len(model_samples[model_folders[0]])} samples = {total} evaluations")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file) if ref_available else None

            print(f"\n  Sample : {sample_name}")

            try:
                tts_dur = get_duration(tts_path)
            except Exception as e:
                print(f"  ERROR reading TTS: {e}")
                results.append({
                    "Model"                              : model,
                    "Sample"                             : sample_name,
                    "Ref Dur s"                          : None,
                    "TTS Dur s"                          : None,
                    "Ratio TTS/Ref (pass=0.90-1.10)"     : None,
                    "Final Pass (PASS/FAIL)"             : "ERROR",
                    "Failure Type (Too Short|Too Long|—)" : str(e),
                })
                continue

            # ── Reference ─────────────────────────────────────────────────────
            ref_dur = None
            ratio   = None

            if ref_available and ref_path and os.path.exists(ref_path):
                try:
                    ref_dur = get_duration(ref_path)
                    ratio   = round(tts_dur / ref_dur, 4) if ref_dur > 0 else None
                    print(f"  Ref: {round(ref_dur, 3)}s | TTS: {round(tts_dur, 3)}s | Ratio: {ratio}")
                except Exception as e:
                    print(f"  Ref read error: {e} — NO_REF")
            else:
                print(f"  TTS: {round(tts_dur, 3)}s | NO_REF")

            # ── Verdict ────────────────────────────────────────────────────────
            if ratio is None:
                final_pass   = "NO_REF"
                failure_type = "—"
            elif lower_bound <= ratio <= upper_bound:
                final_pass   = "PASS"
                failure_type = "—"
            elif ratio < lower_bound:
                final_pass   = "FAIL"
                failure_type = "Too Short"
            else:
                final_pass   = "FAIL"
                failure_type = "Too Long"

            print(f"  Result : {final_pass} | Ratio: {ratio}")

            results.append({
                "Model"                              : model,
                "Sample"                             : sample_name,
                "Ref Dur s"                          : round(ref_dur, 3) if ref_dur is not None else None,
                "TTS Dur s"                          : round(tts_dur, 3),
                "Ratio TTS/Ref (pass=0.90-1.10)"     : ratio,
                "Final Pass (PASS/FAIL)"             : final_pass,
                "Failure Type (Too Short|Too Long|—)" : failure_type,
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    FINAL_COL = "Final Pass (PASS/FAIL)"
    RATIO_COL = "Ratio TTS/Ref (pass=0.90-1.10)"
    FT_COL    = "Failure Type (Too Short|Too Long|—)"

    summary_rows = []
    for model in model_folders:
        model_df = df[df["Model"] == model]
        total    = len(model_df)

        scored_df    = model_df[model_df[FINAL_COL].isin(["PASS", "FAIL"])]
        scored_total = len(scored_df)
        pass_count   = (scored_df[FINAL_COL] == "PASS").sum()
        short_count  = scored_df[FT_COL].str.contains("Too Short", na=False).sum()
        long_count   = scored_df[FT_COL].str.contains("Too Long",  na=False).sum()
        no_ref_count = (model_df[FINAL_COL] == "NO_REF").sum()
        error_count  = (model_df[FINAL_COL] == "ERROR").sum()

        valid_ratios = model_df[RATIO_COL].dropna()

        summary_rows.append({
            "Model"                          : model,
            "Total Segments"                 : total,
            "Pass Rate (PASS / scored)"      : f"{pass_count}/{scored_total}" if scored_total > 0 else "—",
            "Too Short Fails"                : short_count,
            "Too Long Fails"                 : long_count,
            "No Ref"                         : no_ref_count,
            "Errors"                         : error_count,
            "Median Ratio"                   : round(valid_ratios.median(), 4) if len(valid_ratios) > 0 else None,
            "Mean Ratio"                     : round(valid_ratios.mean(),   4) if len(valid_ratios) > 0 else None,
            "Min Ratio"                      : round(valid_ratios.min(),    4) if len(valid_ratios) > 0 else None,
            "Max Ratio"                      : round(valid_ratios.max(),    4) if len(valid_ratios) > 0 else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"]  = summary_df["Pass Rate (PASS / scored)"].apply(
        lambda x: int(x.split("/")[0]) if x != "—" else -1
    )
    summary_df["_ratio_dev"] = summary_df["Median Ratio"].apply(
        lambda x: abs(x - 1.0) if x is not None else 999
    )
    summary_df = summary_df.sort_values(
        by=["_pass_num", "Too Short Fails", "_ratio_dev"],
        ascending=[False, True, True]
    ).drop(columns=["_pass_num", "_ratio_dev"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df.to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Pass Rate      → segments within ±10% of reference duration")
    print("Too Short      → TTS cut off or speaking too fast vs reference")
    print("Too Long       → TTS too slow or padded vs reference")
    print("No Ref         → no reference file — ratio check skipped")
    print("Median Ratio   → 1.0 = perfect | < 0.9 too short | > 1.1 too long")
    tol = config.DURATION_TOLERANCE
    print(f"\nTolerance: ±{int(tol * 100)}%  (pass band: {round(1.0 - tol, 2)} – {round(1.0 + tol, 2)})")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duration ratio gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "duration"))
    parser.add_argument("--models-dir",  default=None)
    parser.add_argument("--ref-dir",     default=None)
    args = parser.parse_args()

    state = {}
    if args.models_dir:
        state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        state["ref_dir"] = os.path.abspath(args.ref_dir)

    load_model()
    df, summary_df = run_gate(state or None)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
