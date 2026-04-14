"""
Gate: Pitch (median register + expressiveness)
Env : base (python 3.13)
Uses PRAAT (via parselmouth) to extract F0. Checks:
  1. Absolute std floor (expressiveness minimum, no reference needed)
  2. Std ratio vs reference (relative expressiveness)
  3. Median delta vs reference (pitch register / zone)

NEAR_MISS: metric within 20% of threshold boundary → NEAR_MISS not FAIL
  Std abs:   PASS >= 20 Hz | NEAR_MISS >= 16 Hz | FAIL < 16 Hz
  Std ratio: PASS >= 0.50  | NEAR_MISS >= 0.40  | FAIL < 0.40
  Register:  PASS <= 30 Hz | NEAR_MISS <= 36 Hz | FAIL > 36 Hz
DEGRADED : ref voiced ratio < 0.20 → skip delta/ratio checks
REVIEW   : ref degraded but TTS passes absolute sanity bounds
             (voiced_ratio >= 0.10, F0 std >= 5.0 Hz)

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
    Returns (pitch_mean_hz, pitch_std_hz, pitch_range_hz, voiced_ratio) using PRAAT.
    voiced_ratio = fraction of total frames where PRAAT detected pitch.
    Falls back to (None, None, None, None) on error.
    """
    try:
        import parselmouth

        snd   = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch(
            time_step=0.01,       # 10 ms frames
            pitch_floor=60.0,     # Hz — covers low male voices
            pitch_ceiling=600.0,  # Hz — covers high female voices
        )

        f0_values    = pitch.selected_array["frequency"]  # 0 = unvoiced frame
        total_frames = len(f0_values)
        voiced_f0    = f0_values[f0_values > 0]

        if len(voiced_f0) == 0:
            print(f"  No voiced frames: {os.path.basename(audio_path)}")
            return None, None, None, 0.0

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
        return None, None, None, None


# ── Near-miss helpers ──────────────────────────────────────────────────────────
def _check_high(value, threshold, margin):
    """Higher-is-better. Returns PASS / NEAR_MISS / FAIL."""
    if value >= threshold:
        return "PASS"
    if value >= threshold * (1.0 - margin):
        return "NEAR_MISS"
    return "FAIL"


def _check_low(value, threshold, margin):
    """Lower-is-better. Returns PASS / NEAR_MISS / FAIL."""
    if value <= threshold:
        return "PASS"
    if value <= threshold * (1.0 + margin):
        return "NEAR_MISS"
    return "FAIL"


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = {}
    MODELS_DIR    = model_state.get("models_dir")  or config.MODELS_DIR
    REFERENCE_DIR = model_state.get("ref_dir")     or config.REFERENCE_DIR

    PITCH_MEDIAN_THRESHOLD    = config.PITCH_MEDIAN_THRESHOLD
    PITCH_STD_ABS_THRESHOLD   = config.PITCH_STD_ABS_THRESHOLD
    PITCH_STD_RATIO_THRESHOLD = config.PITCH_STD_RATIO_THRESHOLD
    NEAR_MISS_MARGIN          = config.PITCH_NEAR_MISS_MARGIN

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
            std_abs_r        = None
            std_ratio_r      = None
            mean_r           = None
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
                        print("  Reference voiced ratio < 0.2 — delta checks unreliable")
                    else:
                        ref_flag = "—"

                        if ref_mean is not None and tts_mean is not None:
                            mean_delta = round(abs(ref_mean - tts_mean), 2)
                            mean_r     = _check_low(mean_delta, PITCH_MEDIAN_THRESHOLD, NEAR_MISS_MARGIN)

                        if ref_std is not None and tts_std is not None and ref_std > 0:
                            std_ratio   = round(tts_std / ref_std, 3)
                            std_ratio_r = _check_high(std_ratio, PITCH_STD_RATIO_THRESHOLD, NEAR_MISS_MARGIN)

                        if ref_range is not None and tts_range is not None and ref_range > 0:
                            range_ratio = round(tts_range / ref_range, 3)

            # Absolute std floor — always runs, no reference needed
            if tts_std is not None:
                std_abs_r = _check_high(tts_std, PITCH_STD_ABS_THRESHOLD, NEAR_MISS_MARGIN)

            # ── Final verdict ──────────────────────────────────────────────────
            if tts_mean is None:
                final_pass = "ERROR"

            elif is_degraded:
                # Reference unvoiced — skip delta/ratio, run absolute sanity only
                abs_ok = (
                    tts_voiced_ratio is not None and tts_voiced_ratio >= config.TTS_VOICED_ABS_MIN and
                    tts_std is not None and tts_std >= config.TTS_PITCH_STD_ABS_MIN
                )
                if abs_ok:
                    final_pass = "REVIEW"
                else:
                    abs_fails = []
                    if tts_voiced_ratio is None or tts_voiced_ratio < config.TTS_VOICED_ABS_MIN:
                        abs_fails.append("Unvoiced_Abs")
                    if tts_std is None or tts_std < config.TTS_PITCH_STD_ABS_MIN:
                        abs_fails.append("Flat_Abs")
                    final_pass = f"FAIL ({', '.join(abs_fails)})"

            else:
                fail_reasons      = []
                near_miss_reasons = []
                for result, label in [
                    (std_abs_r,   "Flat_Abs"),
                    (std_ratio_r, "Flat_Ratio"),
                    (mean_r,      "Register"),
                ]:
                    if result == "FAIL":
                        fail_reasons.append(label)
                    elif result == "NEAR_MISS":
                        near_miss_reasons.append(label)

                if fail_reasons:
                    final_pass = f"FAIL ({', '.join(fail_reasons)})"
                elif near_miss_reasons:
                    final_pass = f"NEAR_MISS ({', '.join(near_miss_reasons)})"
                else:
                    final_pass = "PASS"

            print(f"  Result : {final_pass} | Mean Δ: {mean_delta} | "
                  f"Std Ratio: {std_ratio} | Degraded: {is_degraded}")

            results.append({
                "Model"                                                              : model,
                "Sample"                                                             : sample_name,
                "TTS Mean Hz"                                                        : tts_mean,
                "TTS Std Hz (expressiveness; abs floor>=20Hz)"                       : tts_std,
                "TTS Range Hz (F0 max-min)"                                          : tts_range,
                "TTS Voiced Ratio (voiced frames/total)"                             : tts_voiced_ratio,
                "Ref Mean Hz"                                                        : ref_mean,
                "Ref Std Hz"                                                         : ref_std,
                "Ref Range Hz"                                                       : ref_range,
                "Ref Voiced Ratio (degraded if<0.20)"                                : ref_voiced_ratio,
                "Mean Delta Hz (threshold<=30Hz|NEAR_MISS<=36Hz)"                   : mean_delta,
                "Std Ratio TTS/Ref (threshold>=0.50|NEAR_MISS>=0.40)"               : std_ratio,
                "Range Ratio TTS/Ref"                                                : range_ratio,
                "Std Abs (PASS>=20Hz|NEAR_MISS>=16Hz|FAIL<16Hz)"                    : std_abs_r or "—",
                "Std Ratio (PASS>=0.50|NEAR_MISS>=0.40|FAIL<0.40)"                  : std_ratio_r or "—",
                "Register (PASS<=30Hz|NEAR_MISS<=36Hz|FAIL>36Hz)"                   : mean_r or "—",
                "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                            : final_pass,
                "Ref Flag (—=clean|REF_UNVOICED=voiced<0.20|NO_REF)"                : ref_flag,
                "_is_degraded"                                                       : is_degraded or is_short,
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    FINAL_COL = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"]]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total  = len(clean_df)
        clean_pass   = clean_df[FINAL_COL].str.startswith("PASS").sum()
        clean_nm     = clean_df[FINAL_COL].str.startswith("NEAR_MISS").sum()
        clean_review = (clean_df[FINAL_COL] == "REVIEW").sum()

        deg_total  = len(degraded_df)
        deg_review = (degraded_df[FINAL_COL] == "REVIEW").sum()

        flat_abs_count   = (model_df[FINAL_COL].str.startswith("FAIL") & model_df[FINAL_COL].str.contains("Flat_Abs",   na=False)).sum()
        flat_ratio_count = (model_df[FINAL_COL].str.startswith("FAIL") & model_df[FINAL_COL].str.contains("Flat_Ratio", na=False)).sum()
        register_count   = (model_df[FINAL_COL].str.startswith("FAIL") & model_df[FINAL_COL].str.contains("Register",   na=False)).sum()

        std_col   = "TTS Std Hz (expressiveness; abs floor>=20Hz)"
        range_col = "TTS Range Hz (F0 max-min)"
        ratio_col = "Std Ratio TTS/Ref (threshold>=0.50|NEAR_MISS>=0.40)"
        rr_col    = "Range Ratio TTS/Ref"
        delta_col = "Mean Delta Hz (threshold<=30Hz|NEAR_MISS<=36Hz)"

        med_tts_std    = round(model_df[std_col].dropna().median(), 2)
        med_tts_range  = round(model_df[range_col].dropna().median(), 2)
        med_std_ratio  = round(model_df[ratio_col].dropna().median(), 3)  if model_df[ratio_col].notna().any() else None
        med_range_ratio = round(model_df[rr_col].dropna().median(), 3)   if model_df[rr_col].notna().any()    else None
        deltas         = model_df[delta_col].dropna()
        med_delta      = round(deltas.median(), 2) if len(deltas) > 0 else None
        max_delta      = round(deltas.max(), 2)    if len(deltas) > 0 else None

        summary_rows.append({
            "Model"                       : model,
            "Total Segments"              : total,
            "Clean Segments"              : clean_total,
            "Clean Pass Rate (PASS only)" : f"{clean_pass}/{clean_total}" if clean_total > 0 else "—",
            "Near Miss"                   : clean_nm,
            "Review (clean)"              : clean_review,
            "Degraded Segments"           : deg_total,
            "Degraded Review"             : deg_review,
            "Flat Abs Fails"              : flat_abs_count,
            "Flat Ratio Fails"            : flat_ratio_count,
            "Register Fails"              : register_count,
            "Median TTS Std Hz"           : med_tts_std,
            "Median TTS Range Hz"         : med_tts_range,
            "Median Std Ratio"            : med_std_ratio,
            "Median Range Ratio"          : med_range_ratio,
            "Median Mean Delta Hz"        : med_delta,
            "Max Mean Delta Hz"           : max_delta,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"] = summary_df["Clean Pass Rate (PASS only)"].apply(parse_rate)
    summary_df["_nm"]             = summary_df["Near Miss"]
    summary_df["_med_std_ratio"]  = summary_df["Median Std Ratio"].fillna(-999)
    summary_df["_med_delta"]      = summary_df["Median Mean Delta Hz"].fillna(9999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_nm", "_med_std_ratio", "_med_delta"],
        ascending=[False, True, False, True]
    ).drop(columns=["_clean_pass_num", "_nm", "_med_std_ratio", "_med_delta"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    display_cols = [c for c in df.columns if not c.startswith("_")]
    print(df[display_cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate (PASS only)", "Near Miss", "Review (clean)",
        "Degraded Segments", "Degraded Review",
        "Flat Abs Fails", "Flat Ratio Fails", "Register Fails",
        "Median TTS Std Hz", "Median TTS Range Hz",
        "Median Std Ratio", "Median Range Ratio",
        "Median Mean Delta Hz", "Max Mean Delta Hz"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate     → primary ranking — ref voiced ratio >= 0.2 only")
    print("Near Miss           → metric within 20% of threshold — borderline, listen before failing")
    print("Review (clean)      → ref degraded but TTS has voice — can't compare, needs listen")
    print("Degraded Review     → same but from short/noisy segments")
    print("Flat Abs Fails      → TTS std below 20 Hz floor — robotic delivery regardless of reference")
    print("Flat Ratio Fails    → TTS flatter than reference (std ratio < 0.5×)")
    print("Register Fails      → TTS pitch zone wrong vs reference (mean delta > 30 Hz)")
    print("Median TTS Range Hz → F0 max−min across voiced frames: higher = more expressive")
    print("Median Range Ratio  → TTS range / ref range: < 1.0 = TTS narrower pitch excursion than ref")
    print(f"\nEstimator: PRAAT (parselmouth). Mean F0 per segment, median across segments.")
    print(f"Thresholds: Mean Δ ≤ {config.PITCH_MEDIAN_THRESHOLD}Hz (NM ≤ "
          f"{round(config.PITCH_MEDIAN_THRESHOLD * (1 + config.PITCH_NEAR_MISS_MARGIN), 1)}Hz) | "
          f"Std ≥ {config.PITCH_STD_ABS_THRESHOLD}Hz (NM ≥ "
          f"{round(config.PITCH_STD_ABS_THRESHOLD * (1 - config.PITCH_NEAR_MISS_MARGIN), 1)}Hz) | "
          f"Std ratio ≥ {config.PITCH_STD_RATIO_THRESHOLD}× ref (NM ≥ "
          f"{round(config.PITCH_STD_RATIO_THRESHOLD * (1 - config.PITCH_NEAR_MISS_MARGIN), 2)}×)")


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
