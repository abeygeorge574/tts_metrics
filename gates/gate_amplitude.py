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
def analyze_audio(file_path, clip_limit_db=-1.0):
    """
    Returns (LUFS, LRA, spectral_centroid, true_peak_dBFS, clip_rate).
    clip_rate = fraction of samples at or above clip_limit_db.
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

    peak_amp  = float(np.max(np.abs(data_pln)))
    peak_db   = 20 * np.log10(peak_amp) if peak_amp > 0 else -100.0

    clip_amp  = 10 ** (clip_limit_db / 20)
    clip_rate = float(np.mean(np.abs(data_pln) >= clip_amp))

    return round(lufs, 3), round(lra, 3), round(centroid, 2), round(peak_db, 3), round(clip_rate, 6)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    MODELS_DIR    = (model_state or {}).get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = (model_state or {}).get("ref_dir")    or config.REFERENCE_DIR

    LUFS_TOLERANCE     = config.LUFS_TOLERANCE
    LRA_TOLERANCE      = config.LRA_TOLERANCE
    CENTROID_TOLERANCE = config.CENTROID_TOLERANCE
    PEAK_LIMIT         = config.PEAK_LIMIT
    REF_LUFS_MIN       = config.REF_LUFS_MIN
    REF_LUFS_MAX       = config.REF_LUFS_MAX
    NEAR_MISS_MARGIN   = config.AMPLITUDE_NEAR_MISS_MARGIN
    CLIP_RATE_WARN     = config.CLIP_RATE_WARN
    TTS_LUFS_ABS_MIN   = config.TTS_LUFS_ABS_MIN
    TTS_LUFS_ABS_MAX   = config.TTS_LUFS_ABS_MAX
    TTS_LRA_ABS_MIN    = config.TTS_LRA_ABS_MIN
    TTS_LRA_ABS_MAX    = config.TTS_LRA_ABS_MAX

    for folder in [REFERENCE_DIR, MODELS_DIR]:
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
    missing_refs = [f for f in sample_names if not os.path.exists(os.path.join(REFERENCE_DIR, f))]
    if missing_refs:
        print(f"  Warning: {len(missing_refs)} samples have no reference — will skip delta checks: {missing_refs}")
    else:
        print("All reference files found.")

    total = len(model_folders) * len(sample_names)
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = {total} evaluations")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            duration = sf.info(tts_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            if is_short:
                print(f"\n  Sample : {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample : {sample_name}")

            try:
                ref_lufs, ref_lra, ref_cent, ref_peak, ref_clip_rate = analyze_audio(ref_path, PEAK_LIMIT)
                tts_lufs, tts_lra, tts_cent, tts_peak, tts_clip_rate = analyze_audio(tts_path, PEAK_LIMIT)

                print(f"  Ref    : LUFS={ref_lufs} | LRA={ref_lra} | "
                      f"Cent={ref_cent} | Peak={ref_peak} | ClipRate={ref_clip_rate}")
                print(f"  TTS    : LUFS={tts_lufs} | LRA={tts_lra} | "
                      f"Cent={tts_cent} | Peak={tts_peak} | ClipRate={tts_clip_rate}")

                # ── Clipping classification ────────────────────────────────────
                ref_clipping     = ref_clip_rate >= CLIP_RATE_WARN
                tts_hard_clip    = tts_clip_rate >= CLIP_RATE_WARN
                tts_near_clip    = 0 < tts_clip_rate < CLIP_RATE_WARN

                if tts_hard_clip and ref_clipping:
                    peak_flag = "REF_ALSO_CLIPPED"
                elif tts_hard_clip:
                    peak_flag = "CLIPPING"
                elif tts_near_clip:
                    peak_flag = "NEAR_CLIP"
                elif ref_clipping:
                    peak_flag = "REF_CLIPPED"
                else:
                    peak_flag = "—"

                # ── Degraded: ref quality makes delta comparison unreliable ───
                ref_lufs_degraded = not (REF_LUFS_MIN <= ref_lufs <= REF_LUFS_MAX)
                is_degraded       = ref_lufs_degraded or is_short or ref_clipping
                ref_flag          = (
                    "SHORT_SEGMENT"     if is_short          else
                    "REF_CLIPPED"       if ref_clipping      else
                    "REF_LUFS_DEGRADED" if ref_lufs_degraded else "—"
                )

                lufs_diff = round(abs(ref_lufs - tts_lufs), 3)
                lra_diff  = round(abs(ref_lra  - tts_lra),  3)
                cent_diff = round(abs(ref_cent - tts_cent),  2)

                # ── Verdict ────────────────────────────────────────────────────
                def _check(delta, tolerance):
                    """PASS / NEAR_MISS / FAIL based on delta vs threshold + margin."""
                    if delta <= tolerance:
                        return "PASS"
                    elif delta <= tolerance * (1 + NEAR_MISS_MARGIN):
                        return "NEAR_MISS"
                    return "FAIL"

                # Absolute TTS sanity check — always runs regardless of ref quality
                abs_fails = []
                if not (TTS_LUFS_ABS_MIN <= tts_lufs <= TTS_LUFS_ABS_MAX):
                    abs_fails.append("Volume_Abs")
                if not (TTS_LRA_ABS_MIN <= tts_lra <= TTS_LRA_ABS_MAX):
                    abs_fails.append("Dynamics_Abs")

                if tts_hard_clip:
                    final_pass = "FAIL (Clipping)"

                elif abs_fails:
                    final_pass = f"FAIL ({', '.join(abs_fails)})"

                elif is_degraded:
                    # Delta checks skipped — ref was unreliable, TTS passes abs bounds
                    final_pass = "REVIEW"

                else:
                    lufs_v = _check(lufs_diff, LUFS_TOLERANCE)
                    lra_v  = _check(lra_diff,  LRA_TOLERANCE)
                    cent_v = _check(cent_diff, CENTROID_TOLERANCE)

                    hard_fails  = [n for n, v in [("Volume_Delta", lufs_v), ("Dynamics_Delta", lra_v), ("EQ_Delta", cent_v)] if v == "FAIL"]
                    near_misses = [n for n, v in [("Volume_Delta", lufs_v), ("Dynamics_Delta", lra_v), ("EQ_Delta", cent_v)] if v == "NEAR_MISS"]

                    if hard_fails:
                        final_pass = f"FAIL ({', '.join(hard_fails)})"
                    elif near_misses:
                        final_pass = f"NEAR_MISS ({', '.join(near_misses)})"
                    else:
                        final_pass = "PASS"

                    # NEAR_CLIP warning appended to any non-FAIL verdict
                    if tts_near_clip and not final_pass.startswith("FAIL"):
                        final_pass += " +NEAR_CLIP_Abs"

                print(f"  Result : {final_pass} | Degraded: {is_degraded} | Peak: {peak_flag}")

                results.append({
                    "Model"                                                                       : model,
                    "Sample"                                                                      : sample_name,
                    "Ref LUFS"                                                                    : ref_lufs,
                    "TTS LUFS"                                                                    : tts_lufs,
                    "LUFS Delta (threshold<=6.5)"                                                 : lufs_diff,
                    "Ref LRA"                                                                     : ref_lra,
                    "TTS LRA"                                                                     : tts_lra,
                    "LRA Delta (threshold<=3.0)"                                                  : lra_diff,
                    "Ref Cent"                                                                    : ref_cent,
                    "TTS Cent"                                                                    : tts_cent,
                    "Cent Delta (threshold<=500Hz)"                                               : cent_diff,
                    "Ref Peak"                                                                    : ref_peak,
                    "TTS Peak"                                                                    : tts_peak,
                    "TTS Clip Rate % (NEAR_CLIP=0-0.1%|CLIPPING>=0.1%)"                          : round(tts_clip_rate * 100, 4),
                    "Peak Flag"                                                                   : peak_flag,
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                                     : final_pass,
                    "Ref Flag (—=clean|REF_CLIPPED|REF_LUFS_DEGRADED|SHORT_SEGMENT)"              : ref_flag,
                    "_is_degraded"                                                                : is_degraded,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"                                                                       : model,
                    "Sample"                                                                      : sample_name,
                    "Ref LUFS"                                                                    : None,
                    "TTS LUFS"                                                                    : None,
                    "LUFS Delta (threshold<=6.5)"                                                 : None,
                    "Ref LRA"                                                                     : None,
                    "TTS LRA"                                                                     : None,
                    "LRA Delta (threshold<=3.0)"                                                  : None,
                    "Ref Cent"                                                                    : None,
                    "TTS Cent"                                                                    : None,
                    "Cent Delta (threshold<=500Hz)"                                               : None,
                    "Ref Peak"                                                                    : None,
                    "TTS Peak"                                                                    : None,
                    "TTS Clip Rate % (NEAR_CLIP=0-0.1%|CLIPPING>=0.1%)"                          : None,
                    "Peak Flag"                                                                   : "ERROR",
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                                     : "ERROR",
                    "Ref Flag (—=clean|REF_CLIPPED|REF_LUFS_DEGRADED|SHORT_SEGMENT)"              : "ERROR",
                    "_is_degraded"                                                                : False,
                })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"] & (model_df["Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"] != "ERROR")]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total = len(clean_df)
        # PASS +NEAR_CLIP is still a pass — only the warning suffix differs
        clean_pass  = clean_df["Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"].str.startswith("PASS").sum()

        deg_total   = len(degraded_df)

        FINAL_COL = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"
        fp = model_df[FINAL_COL]
        clipping_count  = (fp.str.startswith("FAIL") & fp.str.contains("Clipping",       na=False)).sum()
        volume_count    = (fp.str.startswith("FAIL") & fp.str.contains("Volume",         na=False)).sum()
        dynamics_count  = (fp.str.startswith("FAIL") & fp.str.contains("Dynamics",       na=False)).sum()
        eq_count        = (fp.str.startswith("FAIL") & fp.str.contains("EQ_Delta",       na=False)).sum()
        near_miss_count = fp.str.startswith("NEAR_MISS", na=False).sum()
        review_count    = (fp == "REVIEW").sum()
        near_clip_count = model_df["Peak Flag"].eq("NEAR_CLIP").sum()
        error_count     = (fp == "ERROR").sum()

        summary_rows.append({
            "Model"                                   : model,
            "Total Segments"                          : total,
            "Clean Segments"                          : clean_total,
            "Clean Pass Rate (PASS only)"             : f"{clean_pass}/{clean_total}" if clean_total > 0 else "—",
            "Degraded Segments"                       : deg_total,
            "Review (degraded ref, TTS sane)"         : review_count,
            "Near Miss (delta only)"                  : near_miss_count,
            "Near Clip (abs only)"                    : near_clip_count,
            "Clipping Fails (abs)"                    : clipping_count,
            "Volume Fails (delta+abs)"                : volume_count,
            "Dynamics Fails (delta+abs)"              : dynamics_count,
            "EQ Fails (delta only)"                   : eq_count,
            "Errors"                                  : error_count,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"] = summary_df["Clean Pass Rate (PASS only)"].apply(parse_rate)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num",
            "Clipping Fails (abs)", "Volume Fails (delta+abs)", "Dynamics Fails (delta+abs)", "EQ Fails (delta only)",
            "Near Miss (delta only)", "Near Clip (abs only)", "Errors"],
        ascending=[False, True, True, True, True, True, True, True]
    ).drop(columns=["_clean_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    display_cols = [c for c in df.columns if not c.startswith("_")]
    print(df[display_cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate (PASS only)", "Degraded Segments", "Review (degraded ref, TTS sane)",
        "Near Miss (delta only)", "Near Clip (abs only)",
        "Clipping Fails (abs)", "Volume Fails (delta+abs)", "Dynamics Fails (delta+abs)", "EQ Fails (delta only)"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate          → primary ranking")
    print("Near Miss (delta only)   → marginal delta, not hard fail — ref was clean")
    print("Review (degraded ref)    → ref unreliable, TTS sane on abs — needs human listen")
    print("Near Clip (abs only)     → <0.1% samples near peak — warn only, not fail")
    print("Clipping Fails (abs)     → >=0.1% samples clipping — always absolute, no ref needed")
    print("Volume Fails (delta+abs) → LUFS_Delta > 6.5 OR TTS LUFS outside [-40,-5]")
    print("Dynamics Fails (delta+abs)→ LRA_Delta > 3.0 OR TTS LRA outside [0.5,20]")
    print("EQ Fails (delta only)    → centroid delta > 500Hz — delta only, no abs check")
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
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "amplitude"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = {}
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)

    load_model()
    df, summary_df = run_gate(model_state or None)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
