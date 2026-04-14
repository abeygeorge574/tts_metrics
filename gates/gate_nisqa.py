"""
Gate: NISQA (MOS, Noisiness, Discontinuity, Coloration, Loudness)
Env : base (python 3.13)
Scores TTS audio with the NISQA model. Supports absolute thresholds
and delta-vs-reference thresholds with hybrid pass/fail logic.

NEAR_MISS: absolute barely fails — all failing dimensions within 20% of their threshold.
  e.g. MOS threshold=3.75 → NEAR_MISS if MOS ∈ [3.0, 3.75) and all other dims borderline.
REVIEW   : absolute fails but delta vs reference passes — TTS is like reference, both borderline.
           Needs human listen before failing the segment.
DEGRADED : reference fails NISQA_REF_THRESHOLDS — delta comparison skipped (REF_QUALITY flag).

Files longer than ~2.5 min are automatically chunked into 30 s clips,
each clip is scored, and the five metrics are averaged across chunks.
"""

import os
import sys
import argparse
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_NISQA_CHUNK_S = 15   # training distribution ~5-15s


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    NISQA_REPO = config.NISQA_REPO
    if NISQA_REPO not in sys.path:
        sys.path.insert(0, NISQA_REPO)

    from nisqa.NISQA_model import nisqaModel

    NISQA_WEIGHT = config.NISQA_WEIGHT
    if not os.path.exists(NISQA_WEIGHT):
        raise FileNotFoundError(f"NISQA weights not found: {NISQA_WEIGHT}")

    print(f"NISQA repo  : {NISQA_REPO}")
    print(f"NISQA weight: {NISQA_WEIGHT}")
    print("NISQA model ready (instantiated per file call).")

    return {"nisqa_weight": NISQA_WEIGHT}


# ── Score single file (raw — no chunking) ─────────────────────────────────────
def _score_file_raw(audio_path, nisqa_weight):
    """Call NISQA directly on one file. Raises ValueError if file is too long."""
    from nisqa.NISQA_model import nisqaModel

    args = {
        "mode"            : "predict_file",
        "pretrained_model": nisqa_weight,
        "deg"             : audio_path,
        "data_dir"        : None,
        "output_dir"      : None,
        "csv_file"        : None,
        "csv_deg"         : None,
        "tr_bs_val"       : 1,
        "tr_num_workers"  : 0,
        "ms_channel"      : None,
    }
    nisqa   = nisqaModel(args)
    df_pred = nisqa.predict()
    row     = df_pred.iloc[0]

    return {
        "MOS"          : round(float(row["mos_pred"]),  3),
        "Noisiness"    : round(float(row["noi_pred"]),  3),
        "Discontinuity": round(float(row["dis_pred"]),  3),
        "Coloration"   : round(float(row["col_pred"]),  3),
        "Loudness"     : round(float(row["loud_pred"]), 3),
    }


# ── Score with automatic chunking ─────────────────────────────────────────────
def score_single_file(audio_path, nisqa_weight):
    """
    Score one audio file with NISQA.

    If the file is ≤ _NISQA_CHUNK_S seconds it is scored directly.
    Longer files are split into _NISQA_CHUNK_S-second temp WAV clips,
    each clip is scored, and the five metrics are averaged across chunks.
    Temp files are deleted immediately after scoring.
    """
    import soundfile as sf

    # Fast path — try scoring as-is first
    try:
        return _score_file_raw(audio_path, nisqa_weight)
    except ValueError as e:
        if "max_length" not in str(e) and "n_wins" not in str(e):
            raise
        print(f"  [NISQA] File too long — chunking into {_NISQA_CHUNK_S}s clips …")

    # Load full waveform for chunking
    import math
    data, sr = sf.read(audio_path, always_2d=False)
    total_frames = len(data)
    max_chunk_frames = int(_NISQA_CHUNK_S * sr)
    n_chunks         = math.ceil(total_frames / max_chunk_frames)
    chunk_size       = total_frames / n_chunks   # equal-length, float → round at boundaries
    print(f"  [NISQA] {total_frames/sr:.1f}s → {n_chunks} equal chunks (~{total_frames/sr/n_chunks:.1f}s each)")

    all_scores = []
    tmp_dir = tempfile.mkdtemp(prefix="nisqa_chunk_")
    try:
        for i in range(n_chunks):
            start = round(i       * chunk_size)
            end   = round((i + 1) * chunk_size)
            chunk = data[start:end]

            # Skip chunks shorter than 0.5 s — NISQA can't handle them
            if (end - start) / sr < 0.5:
                print(f"  [NISQA] chunk {i+1}/{n_chunks}: too short, skipping")
                continue

            tmp_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
            sf.write(tmp_path, chunk, sr)

            try:
                scores = _score_file_raw(tmp_path, nisqa_weight)
                all_scores.append(scores)
                print(f"  [NISQA] chunk {i+1}/{n_chunks}: MOS={scores['MOS']}")
            except ValueError as e:
                print(f"  [NISQA] chunk {i+1}/{n_chunks} still too long — skipping ({e})")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if not all_scores:
        raise RuntimeError(f"NISQA: no usable chunks from {audio_path}")

    # Average each metric across chunks
    keys = all_scores[0].keys()
    return {
        k: round(sum(s[k] for s in all_scores) / len(all_scores), 3)
        for k in keys
    }


# ── Pass/fail helpers ──────────────────────────────────────────────────────────
def get_absolute_pass(scores):
    return all(scores[k] >= config.NISQA_THRESHOLDS[k] for k in config.NISQA_THRESHOLDS)

def get_delta_pass(deltas):
    return all(deltas[k] >= config.NISQA_DELTA_THRESHOLDS[k] for k in config.NISQA_DELTA_THRESHOLDS)

def get_absolute_near_miss(scores):
    """
    Returns True if all failing absolute dimensions are within NEAR_MISS_MARGIN of threshold.
    Used to distinguish NEAR_MISS (borderline) from hard FAIL.
    """
    near_miss_margin = 0.20
    for k, threshold in config.NISQA_THRESHOLDS.items():
        if scores[k] < threshold:
            near_miss_floor = threshold * (1.0 - near_miss_margin)
            if scores[k] < near_miss_floor:
                return False  # This dimension fails badly — not near-miss
    return True  # All failing dims are borderline

def get_primary_failure_with_delta(deltas):
    return min(deltas, key=deltas.get)

def get_primary_failure_without_delta(scores):
    distances = {k: scores[k] - config.NISQA_THRESHOLDS[k] for k in config.NISQA_THRESHOLDS}
    return min(distances, key=distances.get)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    nisqa_weight = model_state["nisqa_weight"]

    MODELS_DIR    = model_state.get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = model_state.get("ref_dir")    or config.REFERENCE_DIR

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

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

    reference_available = os.path.exists(REFERENCE_DIR)
    if reference_available:
        ref_files = sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".wav")])
        print(f"Reference folder found: {len(ref_files)} files")
    else:
        print("No reference folder — absolute thresholds only")

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
                print(f"\n  Sample: {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample: {sample_name}")

            try:
                tts_scores = score_single_file(tts_path, nisqa_weight)
            except Exception as e:
                print(f"  SKIP: NISQA failed on {sample_name} ({e})")
                results.append({
                    "Model": model, "Sample": sample_name,
                    "MOS": None, "Noisiness": None, "Discontinuity": None,
                    "Coloration": None, "Loudness": None,
                    "ΔMOS": None, "ΔNoisiness": None, "ΔDiscontinuity": None,
                    "ΔColoration": None, "ΔLoudness": None,
                    "Absolute Pass (PASS=all≥threshold|FAIL)": "SKIP",
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)": "SKIP",
                    "Primary Failure Dimension": "ERROR",
                    "Ref Flag (—=clean|REF_QUALITY=ref_below_floor|NO_REF|SHORT_SEGMENT)": "ERROR",
                    "_is_clean": False,
                })
                continue

            absolute_pass = get_absolute_pass(tts_scores)
            print(f"  TTS    → MOS: {tts_scores['MOS']} | Noi: {tts_scores['Noisiness']} | "
                  f"Dis: {tts_scores['Discontinuity']} | Col: {tts_scores['Coloration']} | "
                  f"Lou: {tts_scores['Loudness']}")

            deltas     = None
            delta_pass = None
            ref_scores = None
            ref_flag   = None

            if reference_available:
                ref_path = os.path.join(REFERENCE_DIR, wav_file)

                if not os.path.exists(ref_path):
                    ref_flag = "NO_REF"
                    print(f"  No reference file found for {wav_file}")
                else:
                    try:
                        ref_scores = score_single_file(ref_path, nisqa_weight)
                    except Exception as e:
                        print(f"  Reference NISQA failed — falling back to absolute only ({e})")
                        ref_flag   = "REF_ERROR"
                        ref_scores = None

                    if ref_scores is not None:
                        print(f"  REF    → MOS: {ref_scores['MOS']} | Noi: {ref_scores['Noisiness']} |"
                              f" Dis: {ref_scores['Discontinuity']} | Col: {ref_scores['Coloration']} |"
                              f" Lou: {ref_scores['Loudness']}")

                    ref_quality_fail = (
                        ref_scores is not None and
                        any(ref_scores[k] < config.NISQA_REF_THRESHOLDS[k]
                            for k in config.NISQA_REF_THRESHOLDS)
                    )
                    if ref_quality_fail:
                        failed_dims = [k for k in config.NISQA_REF_THRESHOLDS
                                       if ref_scores[k] < config.NISQA_REF_THRESHOLDS[k]]
                        ref_flag = "REF_QUALITY"
                        print(f"  Reference below quality floor on {failed_dims} — skipping delta")
                    elif ref_scores is not None:
                        deltas = {
                            k: round(tts_scores[k] - ref_scores[k], 3)
                            for k in tts_scores
                        }
                        delta_pass = get_delta_pass(deltas)
                        print(f"  DELTA  → MOS: {deltas['MOS']} | Noi: {deltas['Noisiness']} | "
                              f"Dis: {deltas['Discontinuity']} | Col: {deltas['Coloration']} | "
                              f"Lou: {deltas['Loudness']}")
            else:
                ref_flag = "NO_REF"

            # ── Final verdict ──────────────────────────────────────────────────
            # PASS:      absolute passes (delta is bonus info only)
            # NEAR_MISS: absolute barely fails, delta also fails (or unavailable)
            # REVIEW:    absolute fails but delta passes — TTS similar to reference
            # FAIL:      absolute fails and delta also fails (or unavailable), not near-miss
            if deltas is not None:
                if absolute_pass and delta_pass:
                    final_result    = "PASS"
                    primary_failure = "—"
                elif absolute_pass and not delta_pass:
                    final_result    = "PASS"  # hybrid: absolute OK even if degraded vs ref
                    primary_failure = get_primary_failure_with_delta(deltas)
                elif not absolute_pass and delta_pass:
                    final_result    = "REVIEW"  # absolute fails but delta is fine — listen before failing
                    primary_failure = get_primary_failure_with_delta(deltas)
                else:  # both fail
                    if get_absolute_near_miss(tts_scores):
                        final_result = "NEAR_MISS"
                    else:
                        final_result = "FAIL"
                    primary_failure = get_primary_failure_with_delta(deltas)
            else:
                if absolute_pass:
                    final_result    = "PASS"
                    primary_failure = "—"
                else:
                    if get_absolute_near_miss(tts_scores):
                        final_result = "NEAR_MISS"
                    else:
                        final_result = "FAIL"
                    primary_failure = get_primary_failure_without_delta(tts_scores)

            flag_val = "SHORT_SEGMENT" if is_short else (ref_flag or "—")
            is_clean = flag_val not in ("REF_QUALITY", "SHORT_SEGMENT", "NO_REF", "REF_ERROR")

            print(f"  Result : {final_result} | Primary failure: {primary_failure} | Flag: {flag_val}")

            row = {
                "Model"                                                               : model,
                "Sample"                                                              : sample_name,
                "MOS"                                                                 : tts_scores["MOS"],
                "Noisiness"                                                           : tts_scores["Noisiness"],
                "Discontinuity"                                                       : tts_scores["Discontinuity"],
                "Coloration"                                                          : tts_scores["Coloration"],
                "Loudness"                                                            : tts_scores["Loudness"],
                "ΔMOS"                                                                : deltas["MOS"]           if deltas else None,
                "ΔNoisiness"                                                          : deltas["Noisiness"]     if deltas else None,
                "ΔDiscontinuity"                                                      : deltas["Discontinuity"] if deltas else None,
                "ΔColoration"                                                         : deltas["Coloration"]    if deltas else None,
                "ΔLoudness"                                                           : deltas["Loudness"]      if deltas else None,
                "Absolute Pass (PASS=all≥threshold|FAIL)"                             : "PASS" if absolute_pass else "FAIL",
                "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"                             : final_result,
                "Primary Failure Dimension"                                           : primary_failure,
                "Ref Flag (—=clean|REF_QUALITY=ref_below_floor|NO_REF|SHORT_SEGMENT)" : flag_val,
                "_is_clean"                                                           : is_clean,
            }
            results.append(row)

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    FINAL_COL = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[model_df["_is_clean"]]
        degraded_df = model_df[~model_df["_is_clean"]]

        mos_values   = model_df["MOS"]
        delta_values = model_df["ΔMOS"].dropna()
        total        = len(model_df)
        clean_total  = len(clean_df)
        deg_total    = len(degraded_df)

        clean_pass   = clean_df[FINAL_COL].str.startswith("PASS").sum()
        clean_nm     = clean_df[FINAL_COL].str.startswith("NEAR_MISS").sum()
        clean_review = (clean_df[FINAL_COL] == "REVIEW").sum()
        clean_fail   = clean_df[FINAL_COL].str.startswith("FAIL").sum()

        deg_pass     = degraded_df[FINAL_COL].str.startswith("PASS").sum()
        deg_review   = (degraded_df[FINAL_COL] == "REVIEW").sum()
        deg_fail     = degraded_df[FINAL_COL].str.startswith("FAIL").sum()

        failure_counts      = model_df[model_df["Primary Failure Dimension"] != "—"]["Primary Failure Dimension"].value_counts()
        most_common_failure = failure_counts.index[0] if len(failure_counts) > 0 else "—"

        summary_rows.append({
            "Model"                       : model,
            "Total Segments"              : total,
            "Clean Segments"              : clean_total,
            "Clean Pass Rate (PASS only)" : f"{clean_pass}/{clean_total}"   if clean_total > 0 else "—",
            "Near Miss"                   : clean_nm,
            "Review (clean)"              : clean_review,
            "Clean Fail Rate"             : f"{clean_fail}/{clean_total}"   if clean_total > 0 else "—",
            "Degraded Segments"           : deg_total,
            "Degraded Pass Rate"          : f"{deg_pass}/{deg_total}"       if deg_total > 0 else "—",
            "Degraded Review"             : deg_review,
            "Median MOS"                  : round(mos_values.median(), 3),
            "Mean ΔMOS"                   : round(delta_values.mean(), 3)   if len(delta_values) > 0 else None,
            "Top Failure Mode"            : most_common_failure,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate (PASS only)"].apply(parse_rate)
    summary_df["_nm"]                = summary_df["Near Miss"]
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df["_mean_delta_mos"]    = summary_df["Mean ΔMOS"].fillna(-999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_nm", "_degraded_pass_num", "_mean_delta_mos", "Median MOS"],
        ascending=[False, True, False, False, False]
    ).drop(columns=["_clean_pass_num", "_nm", "_degraded_pass_num", "_mean_delta_mos"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    display_cols = [c for c in df.columns if not c.startswith("_")]
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[display_cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate (PASS only)", "Near Miss", "Review (clean)", "Clean Fail Rate",
        "Degraded Segments", "Degraded Pass Rate",
        "Median MOS", "Mean ΔMOS", "Top Failure Mode"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate    → primary ranking — trustworthy ground truth comparison")
    print("Near Miss          → absolute barely fails (within 20% of threshold) — borderline, listen first")
    print("Review (clean)     → absolute fails but delta passes — TTS no worse than reference, needs listen")
    print("Top Failure Mode   → which NISQA dimension this model fails most")
    print("Ref Flag REF_QUALITY → ref below NISQA_REF_THRESHOLDS — delta skipped, counts as degraded")
    print("Ref Flag NO_REF    → no reference file — absolute threshold only")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_clean"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NISQA gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "nisqa"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = load_model()
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
