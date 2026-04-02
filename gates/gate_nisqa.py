"""
Gate: NISQA (MOS, Noisiness, Discontinuity, Coloration, Loudness)
Env : base (python 3.13)
Scores TTS audio with the NISQA model. Supports absolute thresholds
and delta-vs-reference thresholds with hybrid pass/fail logic.
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


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


# ── Score single file ──────────────────────────────────────────────────────────
def score_single_file(audio_path, nisqa_weight):
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


# ── Pass/fail helpers ──────────────────────────────────────────────────────────
def get_absolute_pass(scores):
    return all(scores[k] >= config.NISQA_THRESHOLDS[k] for k in config.NISQA_THRESHOLDS)

def get_delta_pass(deltas):
    return all(deltas[k] >= config.NISQA_DELTA_THRESHOLDS[k] for k in config.NISQA_DELTA_THRESHOLDS)

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

    MODELS_DIR    = config.MODELS_DIR
    REFERENCE_DIR = config.REFERENCE_DIR

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
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)

            print(f"\n  Sample: {sample_name}")

            try:
                tts_scores = score_single_file(tts_path, nisqa_weight)
            except ValueError as e:
                if "max_length" in str(e) or "n_wins" in str(e):
                    print(f"  SKIP: file too long for NISQA ({e})")
                    results.append({
                        "Model": model, "Sample": sample_name,
                        "MOS": None, "Noisiness": None, "Discontinuity": None,
                        "Coloration": None, "Loudness": None,
                        "ΔMOS": None, "ΔNoisiness": None, "ΔDiscontinuity": None,
                        "ΔColoration": None, "ΔLoudness": None,
                        "Absolute": "SKIP", "Final": "SKIP",
                        "Primary Failure": "TOO_LONG", "Flag": "TOO_LONG",
                    })
                    continue
                raise

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
                    except ValueError as e:
                        if "max_length" in str(e) or "n_wins" in str(e):
                            print(f"  Reference too long for NISQA — falling back to absolute only")
                            ref_flag = "REF_TOO_LONG"
                            ref_scores = None
                        else:
                            raise

                    if ref_scores is not None:
                        print(f"  REF    → MOS: {ref_scores['MOS']} | Noi: {ref_scores['Noisiness']} | "
                              f"Dis: {ref_scores['Discontinuity']} | Col: {ref_scores['Coloration']} | "
                              f"Lou: {ref_scores['Loudness']}")

                    if ref_scores is not None and ref_scores["MOS"] < 3.0:
                        ref_flag = "REF_QUALITY"
                        print(f"  Reference MOS below 3.0 — skipping delta")
                    else:
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

            if deltas is not None:
                if absolute_pass and delta_pass:
                    final_result    = "PASS"
                    primary_failure = "—"
                elif absolute_pass and not delta_pass:
                    final_result    = "PASS"
                    primary_failure = get_primary_failure_with_delta(deltas)
                elif not absolute_pass and delta_pass:
                    final_result    = "REVIEW"
                    primary_failure = get_primary_failure_with_delta(deltas)
                else:
                    final_result    = "FAIL"
                    primary_failure = get_primary_failure_with_delta(deltas)
            else:
                if absolute_pass:
                    final_result    = "PASS"
                    primary_failure = "—"
                else:
                    final_result    = "FAIL"
                    primary_failure = get_primary_failure_without_delta(tts_scores)

            print(f"  Result : {final_result} | Primary failure: {primary_failure} | Flag: {ref_flag or '—'}")

            row = {
                "Model"          : model,
                "Sample"         : sample_name,
                "MOS"            : tts_scores["MOS"],
                "Noisiness"      : tts_scores["Noisiness"],
                "Discontinuity"  : tts_scores["Discontinuity"],
                "Coloration"     : tts_scores["Coloration"],
                "Loudness"       : tts_scores["Loudness"],
                "ΔMOS"           : deltas["MOS"]           if deltas else None,
                "ΔNoisiness"     : deltas["Noisiness"]     if deltas else None,
                "ΔDiscontinuity" : deltas["Discontinuity"] if deltas else None,
                "ΔColoration"    : deltas["Coloration"]    if deltas else None,
                "ΔLoudness"      : deltas["Loudness"]      if deltas else None,
                "Absolute"       : "PASS" if absolute_pass else "FAIL",
                "Final"          : final_result,
                "Primary Failure": primary_failure,
                "Flag"           : ref_flag or "—",
            }
            results.append(row)

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)
    df["_is_clean"] = df["Flag"].apply(lambda x: x != "REF_QUALITY")

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

        clean_pass   = (clean_df["Final"] == "PASS").sum()
        clean_review = (clean_df["Final"] == "REVIEW").sum()
        clean_fail   = (clean_df["Final"] == "FAIL").sum()

        deg_pass     = (degraded_df["Final"] == "PASS").sum()
        deg_review   = (degraded_df["Final"] == "REVIEW").sum()
        deg_fail     = (degraded_df["Final"] == "FAIL").sum()

        failure_counts      = model_df[model_df["Primary Failure"] != "—"]["Primary Failure"].value_counts()
        most_common_failure = failure_counts.index[0] if len(failure_counts) > 0 else "—"

        summary_rows.append({
            "Model"              : model,
            "Total Segments"     : total,
            "Clean Segments"     : clean_total,
            "Clean Pass Rate"    : f"{clean_pass}/{clean_total}"   if clean_total > 0 else "—",
            "Clean Review Rate"  : f"{clean_review}/{clean_total}" if clean_total > 0 else "—",
            "Clean Fail Rate"    : f"{clean_fail}/{clean_total}"   if clean_total > 0 else "—",
            "Degraded Segments"  : deg_total,
            "Degraded Pass Rate" : f"{deg_pass}/{deg_total}"       if deg_total > 0 else "—",
            "Degraded Review Rate": f"{deg_review}/{deg_total}"    if deg_total > 0 else "—",
            "Degraded Fail Rate" : f"{deg_fail}/{deg_total}"       if deg_total > 0 else "—",
            "Median MOS"         : round(mos_values.median(), 3),
            "Mean ΔMOS"          : round(delta_values.mean(), 3)   if len(delta_values) > 0 else None,
            "Top Failure Mode"   : most_common_failure,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df["_mean_delta_mos"]    = summary_df["Mean ΔMOS"].fillna(-999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num", "_mean_delta_mos"],
        ascending=[False, False, False]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num", "_mean_delta_mos"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    display_cols = [
        "Model", "Sample",
        "MOS", "Noisiness", "Discontinuity", "Coloration", "Loudness",
        "ΔMOS", "ΔNoisiness", "ΔDiscontinuity", "ΔColoration", "ΔLoudness",
        "Absolute", "Final", "Primary Failure", "Flag"
    ]
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[display_cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Clean Review Rate", "Clean Fail Rate",
        "Degraded Pass Rate", "Median MOS", "Mean ΔMOS", "Top Failure Mode"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate    → primary ranking — trustworthy ground truth comparison")
    print("REVIEW segments    → absolute fail but delta small — listen before deciding")
    print("Top Failure Mode   → which artifact type this model produces most")
    print("Flag REF_QUALITY   → reference MOS < 3.0 — delta skipped, counts as degraded")
    print("Flag NO_REF        → no reference file — absolute threshold only, counts as degraded")


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
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "nisqa"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
