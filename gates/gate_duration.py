"""
Gate: Duration Ratio
Env : base (python 3.13)
Checks that TTS duration is within ±10% of the reference duration.
Uses soundfile header reading (no decode).

NEAR_MISS: ratio within 20% of tolerance boundary → NEAR_MISS not FAIL
  Pass band:       [0.90, 1.10]
  Near-miss band:  [0.88, 0.90) or (1.10, 1.12]
DEGRADED : ref_dur < 0.50s → skip ratio check, run TTS absolute bounds only
REVIEW   : ref degraded but TTS duration in [0.5s, 30.0s] → needs human check
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
    MODELS_DIR    = (model_state or {}).get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = (model_state or {}).get("ref_dir")    or config.REFERENCE_DIR

    DURATION_TOLERANCE  = config.DURATION_TOLERANCE        # 0.10 → ±10%
    NEAR_MISS_MARGIN    = config.DURATION_NEAR_MISS_MARGIN  # 0.20

    lower_bound = 1.0 - DURATION_TOLERANCE                  # 0.90
    upper_bound = 1.0 + DURATION_TOLERANCE                  # 1.10
    nm_delta    = DURATION_TOLERANCE * NEAR_MISS_MARGIN     # 0.02
    nm_lower    = lower_bound - nm_delta                    # 0.88
    nm_upper    = upper_bound + nm_delta                    # 1.12

    for folder in [MODELS_DIR]:
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

    # Check reference availability (warn, do not crash on missing)
    ref_available = os.path.exists(REFERENCE_DIR)
    if ref_available:
        sample_names = model_samples[model_folders[0]]
        missing_refs = [w for w in sample_names if not os.path.exists(os.path.join(REFERENCE_DIR, w))]
        if missing_refs:
            print(f"WARNING: {len(missing_refs)} reference files missing — those segments will get NO_REF")
    else:
        print("WARNING: Reference directory not found — all segments will be NO_REF")
        sample_names = model_samples[model_folders[0]]
        missing_refs = list(sample_names)

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
                    "Model"                                                            : model,
                    "Sample"                                                           : sample_name,
                    "Ref Dur s"                                                        : None,
                    "TTS Dur s"                                                        : None,
                    "Ratio TTS/Ref (pass=0.90-1.10|near_miss=0.88-0.90 or 1.10-1.12)" : None,
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                          : "ERROR",
                    "Failure Type (Too Short|Too Long|—)"                              : str(e),
                    "Ref Flag (—=clean|REF_SHORT=ref<0.50s|NO_REF)"                   : "—",
                    "_is_degraded"                                                     : True,
                })
                continue

            # ── Reference handling ─────────────────────────────────────────────
            ref_dur     = None
            ratio       = None
            ref_flag    = "NO_REF"
            is_degraded = False

            if ref_available and ref_path and os.path.exists(ref_path):
                try:
                    ref_dur = get_duration(ref_path)
                    if ref_dur < config.REF_DUR_MIN:
                        ref_flag    = "REF_SHORT"
                        is_degraded = True
                        print(f"  Ref    : {round(ref_dur, 3)}s [SHORT — degraded]")
                    else:
                        ref_flag = "—"
                        ratio    = round(tts_dur / ref_dur, 4) if ref_dur > 0 else None
                        print(f"  Ref    : {round(ref_dur, 3)}s | TTS: {round(tts_dur, 3)}s | Ratio: {ratio}")
                except Exception as e:
                    print(f"  Ref read error: {e} — treating as NO_REF")
                    ref_flag    = "NO_REF"
                    is_degraded = True
            else:
                is_degraded = True
                print(f"  TTS    : {round(tts_dur, 3)}s | NO_REF")

            # ── Final verdict ──────────────────────────────────────────────────
            if is_degraded:
                # No usable reference — run absolute sanity bounds on TTS only
                if tts_dur < config.TTS_DUR_ABS_MIN:
                    final_pass   = "FAIL"
                    failure_type = "Near_Silent_Abs"
                elif tts_dur > config.TTS_DUR_ABS_MAX:
                    final_pass   = "FAIL"
                    failure_type = "Too_Long_Abs"
                else:
                    final_pass   = "REVIEW"
                    failure_type = "—"

            elif ratio is None:
                final_pass   = "ERROR"
                failure_type = "Zero ref duration"

            else:
                if lower_bound <= ratio <= upper_bound:
                    final_pass   = "PASS"
                    failure_type = "—"
                elif nm_lower <= ratio < lower_bound:
                    final_pass   = "NEAR_MISS"
                    failure_type = "Too Short"
                elif upper_bound < ratio <= nm_upper:
                    final_pass   = "NEAR_MISS"
                    failure_type = "Too Long"
                elif ratio < nm_lower:
                    final_pass   = "FAIL"
                    failure_type = "Too Short"
                else:
                    final_pass   = "FAIL"
                    failure_type = "Too Long"

            print(f"  Result : {final_pass} | Ratio: {ratio} | {ref_flag}")

            results.append({
                "Model"                                                            : model,
                "Sample"                                                           : sample_name,
                "Ref Dur s"                                                        : round(ref_dur, 3) if ref_dur is not None else None,
                "TTS Dur s"                                                        : round(tts_dur, 3),
                "Ratio TTS/Ref (pass=0.90-1.10|near_miss=0.88-0.90 or 1.10-1.12)" : ratio,
                "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                          : final_pass,
                "Failure Type (Too Short|Too Long|—)"                              : failure_type,
                "Ref Flag (—=clean|REF_SHORT=ref<0.50s|NO_REF)"                   : ref_flag,
                "_is_degraded"                                                     : is_degraded,
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    FINAL_COL = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"
    RATIO_COL = "Ratio TTS/Ref (pass=0.90-1.10|near_miss=0.88-0.90 or 1.10-1.12)"

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"]]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total  = len(clean_df)
        clean_pass   = clean_df[FINAL_COL].str.startswith("PASS").sum()
        clean_nm     = clean_df[FINAL_COL].str.startswith("NEAR_MISS").sum()

        deg_total  = len(degraded_df)
        deg_review = (degraded_df[FINAL_COL] == "REVIEW").sum()

        short_count = model_df[FINAL_COL].str.contains("Too Short").sum()
        long_count  = model_df[FINAL_COL].str.contains("Too Long").sum()
        error_count = (model_df[FINAL_COL] == "ERROR").sum()

        valid_ratios = clean_df[RATIO_COL].dropna()

        summary_rows.append({
            "Model"                       : model,
            "Total Segments"              : total,
            "Clean Segments"              : clean_total,
            "Clean Pass Rate (PASS only)" : f"{clean_pass}/{clean_total}" if clean_total > 0 else "—",
            "Near Miss"                   : clean_nm,
            "Degraded Segments"           : deg_total,
            "Degraded Review"             : deg_review,
            "Too Short Fails"             : short_count,
            "Too Long Fails"              : long_count,
            "Errors"                      : error_count,
            "Median Ratio"                : round(valid_ratios.median(), 4) if len(valid_ratios) > 0 else None,
            "Mean Ratio"                  : round(valid_ratios.mean(), 4)   if len(valid_ratios) > 0 else None,
            "Min Ratio"                   : round(valid_ratios.min(), 4)    if len(valid_ratios) > 0 else None,
            "Max Ratio"                   : round(valid_ratios.max(), 4)    if len(valid_ratios) > 0 else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"]  = summary_df["Clean Pass Rate (PASS only)"].apply(
        lambda x: int(x.split("/")[0]) if x != "—" else -1
    )
    summary_df["_nm"]        = summary_df["Near Miss"]
    summary_df["_ratio_dev"] = summary_df["Median Ratio"].apply(
        lambda x: abs(x - 1.0) if x is not None else 999
    )

    summary_df = summary_df.sort_values(
        by=["_pass_num", "_nm", "_ratio_dev", "Too Short Fails"],
        ascending=[False, True, True, True]
    ).drop(columns=["_pass_num", "_nm", "_ratio_dev"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    display_cols = [c for c in df.columns if not c.startswith("_")]
    print(df[display_cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate (PASS only)", "Near Miss",
        "Degraded Segments", "Degraded Review",
        "Too Short Fails", "Too Long Fails", "Errors",
        "Median Ratio", "Mean Ratio", "Min Ratio", "Max Ratio"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate  → % of segments within ±10% tolerance (clean reference only)")
    print("Near Miss        → ratio in [0.88,0.90) or (1.10,1.12] — marginally outside tolerance")
    print("Degraded Review  → ref < 0.5s or missing — can't compare, TTS abs bounds checked only")
    print("Too Short Fails  → dialogue cut off — worse than too long for dubbing")
    print("Median Ratio     → 1.0 = perfect | < 0.9 too short | > 1.1 too long")
    print(f"\nTolerance: ±{int(config.DURATION_TOLERANCE * 100)}% "
          f"(NEAR_MISS ±{int((config.DURATION_TOLERANCE + config.DURATION_TOLERANCE * config.DURATION_NEAR_MISS_MARGIN) * 100)}%)")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duration ratio gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "duration"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
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
