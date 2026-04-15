"""
Gate: VAD / Pause Alignment
Env : base (python 3.13)
Uses ffmpeg silencedetect to find pause intervals, then matches reference
pauses to TTS pauses with the Hungarian algorithm (weighted position + duration cost).
Segments where reference pauses/sec > threshold are flagged as degraded.
"""

import os
import sys
import re
import subprocess
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    # No ML model — uses ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("ffmpeg found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise FileNotFoundError(
            "ffmpeg not found. Install: brew install ffmpeg"
        )
    return None


# ── Pause detection ────────────────────────────────────────────────────────────
def get_pauses(audio_path):
    """
    Uses ffmpeg silencedetect to find pause intervals.
    Returns (list of pause dicts, audio_duration_seconds).
    """
    SILENCE_DB           = config.SILENCE_DB
    MIN_SILENCE_DURATION = config.MIN_SILENCE_DURATION

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={MIN_SILENCE_DURATION}",
        "-f", "null", "-"
    ]

    try:
        res    = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=30)
        output = res.stderr
    except subprocess.TimeoutExpired:
        print(f"  ffmpeg timeout: {os.path.basename(audio_path)}")
        return [], 0.0
    except Exception as e:
        print(f"  ffmpeg error: {e}")
        return [], 0.0

    audio_duration = 0.0
    dur_match = re.search(r"Duration: (\d+):(\d+):([\d\.]+)", output)
    if dur_match:
        h, m, s        = dur_match.groups()
        audio_duration = int(h) * 3600 + int(m) * 60 + float(s)

    starts, ends = [], []
    for line in output.split("\n"):
        if "silence_start" in line:
            m = re.search(r"silence_start: ([\d\.]+)", line)
            if m:
                starts.append(float(m.group(1)))
        elif "silence_end" in line:
            m = re.search(r"silence_end: ([\d\.]+)", line)
            if m:
                ends.append(float(m.group(1)))

    # handle case where audio starts with silence
    if len(ends) > len(starts):
        ends = ends[1:]

    pauses = []
    for s, e in zip(starts, ends):
        pauses.append({
            "start"   : round(s, 3),
            "end"     : round(e, 3),
            "duration": round(e - s, 3)
        })

    return pauses, audio_duration


# ── Pause comparison (Hungarian algorithm) ────────────────────────────────────
def compare_pauses(ref_pauses, tts_pauses):
    """
    Matches ref pauses to TTS pauses using the Hungarian algorithm with
    a weighted position + duration cost. Pairs beyond POSITION_HARD_LIMIT
    are assigned infinite cost and never matched.
    """
    POSITION_HARD_LIMIT       = config.POSITION_HARD_LIMIT
    POSITION_WEIGHT           = config.POSITION_WEIGHT
    DURATION_WEIGHT           = config.DURATION_WEIGHT
    POSITION_SCALE            = config.POSITION_SCALE
    PAUSE_COUNT_THRESHOLD     = config.PAUSE_COUNT_THRESHOLD
    POSITION_OFFSET_THRESHOLD = config.POSITION_OFFSET_THRESHOLD
    DURATION_RATIO_MIN        = config.VAD_DURATION_RATIO_MIN
    DURATION_RATIO_MAX        = config.VAD_DURATION_RATIO_MAX

    ref_count = len(ref_pauses)
    tts_count = len(tts_pauses)

    if ref_count == 0 or tts_count == 0:
        return {
            "ref_count"          : ref_count,
            "tts_count"          : tts_count,
            "matched_count"      : 0,
            "unmatched_ref"      : ref_count,
            "unmatched_tts"      : tts_count,
            "count_delta"        : abs(ref_count - tts_count),
            "med_position_offset": None,
            "med_duration_ratio" : None,
            "count_pass"         : abs(ref_count - tts_count) <= PAUSE_COUNT_THRESHOLD,
            "position_pass"      : None,
            "duration_pass"      : None,
        }

    INF         = 1e9
    cost_matrix = np.full((ref_count, tts_count), INF)

    for i, ref_p in enumerate(ref_pauses):
        ref_mid = (ref_p["start"] + ref_p["end"]) / 2
        for j, tts_p in enumerate(tts_pauses):
            tts_mid       = (tts_p["start"] + tts_p["end"]) / 2
            position_diff = abs(ref_mid - tts_mid)

            if position_diff > POSITION_HARD_LIMIT:
                continue  # leave as INF — never assigned

            duration_diff     = abs(ref_p["duration"] - tts_p["duration"])
            position_cost     = position_diff / POSITION_SCALE
            duration_cost     = duration_diff / ref_p["duration"] if ref_p["duration"] > 0 else 0.0
            cost_matrix[i, j] = (
                POSITION_WEIGHT * position_cost +
                DURATION_WEIGHT * duration_cost
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    position_offsets = []
    duration_ratios  = []
    matched_ref      = set()
    matched_tts      = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= INF:
            continue

        ref_p   = ref_pauses[r]
        tts_p   = tts_pauses[c]
        ref_mid = (ref_p["start"] + ref_p["end"]) / 2
        tts_mid = (tts_p["start"] + tts_p["end"]) / 2

        position_offsets.append(abs(ref_mid - tts_mid))
        if ref_p["duration"] > 0:
            duration_ratios.append(tts_p["duration"] / ref_p["duration"])

        matched_ref.add(r)
        matched_tts.add(c)

    matched_count  = len(matched_ref)
    unmatched_ref  = ref_count - matched_count
    unmatched_tts  = tts_count - len(matched_tts)
    count_delta    = unmatched_ref + unmatched_tts

    med_position_offset = round(float(np.median(position_offsets)), 3) if position_offsets else None
    med_duration_ratio  = round(float(np.median(duration_ratios)),  3) if duration_ratios  else None

    count_pass    = count_delta <= PAUSE_COUNT_THRESHOLD
    position_pass = (med_position_offset <= POSITION_OFFSET_THRESHOLD) if med_position_offset is not None else None
    duration_pass = (
        (DURATION_RATIO_MIN <= med_duration_ratio <= DURATION_RATIO_MAX)
        if med_duration_ratio is not None else None
    )

    return {
        "ref_count"          : ref_count,
        "tts_count"          : tts_count,
        "matched_count"      : matched_count,
        "unmatched_ref"      : unmatched_ref,
        "unmatched_tts"      : unmatched_tts,
        "count_delta"        : count_delta,
        "med_position_offset": med_position_offset,
        "med_duration_ratio" : med_duration_ratio,
        "count_pass"         : count_pass,
        "position_pass"      : position_pass,
        "duration_pass"      : duration_pass,
    }


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    MODELS_DIR    = (model_state or {}).get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = (model_state or {}).get("ref_dir")    or config.REFERENCE_DIR

    REF_PAUSES_PER_SECOND_LIMIT = config.REF_PAUSES_PER_SECOND_LIMIT
    VAD_NEAR_MISS_MARGIN        = config.VAD_NEAR_MISS_MARGIN
    TTS_PAUSES_PER_SEC_MAX      = config.TTS_PAUSES_PER_SEC_MAX
    POSITION_OFFSET_THRESHOLD   = config.POSITION_OFFSET_THRESHOLD
    DURATION_RATIO_MIN          = config.VAD_DURATION_RATIO_MIN
    DURATION_RATIO_MAX          = config.VAD_DURATION_RATIO_MAX

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
        print(f"  Warning: {len(missing_refs)} samples have no reference — will skip pause comparison: {missing_refs}")
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
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            print(f"\n  Sample : {sample_name}")

            if not os.path.exists(ref_path):
                print(f"  Skipping {sample_name} — no reference file")
                results.append({
                    "Model"                                           : model,
                    "Sample"                                          : sample_name,
                    "Ref Pauses"                                      : None,
                    "TTS Pauses"                                      : None,
                    "Matched"                                         : None,
                    "Unmatched Ref"                                   : None,
                    "Unmatched TTS"                                   : None,
                    "Count Delta (threshold≤20)"                      : None,
                    "Med Pos Offset s (threshold≤0.20s)"              : None,
                    "Med Dur Ratio (pass band 0.75–1.25)"             : None,
                    "Count Pass"                                      : "—",
                    "Position Pass"                                   : "—",
                    "Duration Pass"                                   : "—",
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"         : "NO_REF",
                    "Ref Flag (—=clean|REF_DENSE=pauses/sec>1.0)"    : "NO_REF",
                    "_is_degraded"                                    : True,
                })
                continue

            try:
                ref_pauses, ref_duration = get_pauses(ref_path)
                tts_pauses, _            = get_pauses(tts_path)

                print(f"  Ref    : {len(ref_pauses)} pauses | duration {round(ref_duration, 2)}s")
                print(f"  TTS    : {len(tts_pauses)} pauses")

                ref_pauses_per_sec = len(ref_pauses) / ref_duration if ref_duration > 0 else 0
                is_degraded        = ref_pauses_per_sec > REF_PAUSES_PER_SECOND_LIMIT
                ref_flag           = "REF_DENSE" if is_degraded else "—"

                if is_degraded:
                    print(f"  Ref pauses/sec={round(ref_pauses_per_sec, 2)} — degraded")

                cmp = compare_pauses(ref_pauses, tts_pauses)

                if is_degraded:
                    # REVIEW if TTS rate is within absolute bounds, else FAIL
                    tts_pauses_per_sec = len(tts_pauses) / ref_duration if ref_duration > 0 else 0
                    tts_rate_ok        = tts_pauses_per_sec <= TTS_PAUSES_PER_SEC_MAX
                    if tts_rate_ok:
                        final_pass = "REVIEW"
                    else:
                        final_pass = "FAIL (Dense_Abs)"
                else:
                    # Near-miss logic: collect near_misses and hard_fails separately
                    near_misses = []
                    hard_fails  = []

                    # Count check
                    if not cmp["count_pass"]:
                        hard_fails.append("Count")

                    # Position near-miss
                    pos_offset = cmp["med_position_offset"]
                    pos_nm_upper = POSITION_OFFSET_THRESHOLD * (1 + VAD_NEAR_MISS_MARGIN)
                    if pos_offset is not None:
                        if pos_offset > POSITION_OFFSET_THRESHOLD:
                            if pos_offset <= pos_nm_upper:
                                near_misses.append("Position")
                            else:
                                hard_fails.append("Position")
                    # position_pass=None means no matched pairs → not a failure

                    # Duration near-miss
                    dur_ratio  = cmp["med_duration_ratio"]
                    dur_nm_min = DURATION_RATIO_MIN * (1 - VAD_NEAR_MISS_MARGIN)
                    dur_nm_max = DURATION_RATIO_MAX * (1 + VAD_NEAR_MISS_MARGIN)
                    if dur_ratio is not None:
                        in_pass_band = DURATION_RATIO_MIN <= dur_ratio <= DURATION_RATIO_MAX
                        in_nm_band   = (dur_nm_min <= dur_ratio < DURATION_RATIO_MIN) or \
                                       (DURATION_RATIO_MAX < dur_ratio <= dur_nm_max)
                        if not in_pass_band:
                            if in_nm_band:
                                near_misses.append("Duration")
                            else:
                                hard_fails.append("Duration")

                    if hard_fails:
                        final_pass = f"FAIL ({', '.join(hard_fails)})"
                    elif near_misses:
                        final_pass = f"NEAR_MISS ({', '.join(near_misses)})"
                    else:
                        final_pass = "PASS"

                print(f"  Matched: {cmp['matched_count']} | "
                      f"Unmatched ref: {cmp['unmatched_ref']} | "
                      f"Unmatched tts: {cmp['unmatched_tts']}")
                print(f"  Result : {final_pass} | Count Δ: {cmp['count_delta']} | "
                      f"Pos: {cmp['med_position_offset']}s | Dur ratio: {cmp['med_duration_ratio']}")

                results.append({
                    "Model"                                          : model,
                    "Sample"                                         : sample_name,
                    "Ref Pauses"                                     : cmp["ref_count"],
                    "TTS Pauses"                                     : cmp["tts_count"],
                    "Matched"                                        : cmp["matched_count"],
                    "Unmatched Ref"                                  : cmp["unmatched_ref"],
                    "Unmatched TTS"                                  : cmp["unmatched_tts"],
                    "Count Delta (threshold≤20)"                     : cmp["count_delta"],
                    "Med Pos Offset s (threshold≤0.20s)"             : cmp["med_position_offset"],
                    "Med Dur Ratio (pass band 0.75–1.25)"            : cmp["med_duration_ratio"],
                    "Count Pass"                                     : "PASS" if cmp["count_pass"] else "FAIL",
                    "Position Pass"                                  : "PASS" if cmp["position_pass"] else "FAIL" if cmp["position_pass"] is not None else "—",
                    "Duration Pass"                                  : "PASS" if cmp["duration_pass"] else "FAIL" if cmp["duration_pass"] is not None else "—",
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"        : final_pass,
                    "Ref Flag (—=clean|REF_DENSE=pauses/sec>1.0)"   : ref_flag,
                    "_is_degraded"                                   : is_degraded,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"                                          : model,
                    "Sample"                                         : sample_name,
                    "Ref Pauses"                                     : None,
                    "TTS Pauses"                                     : None,
                    "Matched"                                        : None,
                    "Unmatched Ref"                                  : None,
                    "Unmatched TTS"                                  : None,
                    "Count Delta (threshold≤20)"                     : None,
                    "Med Pos Offset s (threshold≤0.20s)"             : None,
                    "Med Dur Ratio (pass band 0.75–1.25)"            : None,
                    "Count Pass"                                     : "—",
                    "Position Pass"                                  : "—",
                    "Duration Pass"                                  : "—",
                    "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"        : "ERROR",
                    "Ref Flag (—=clean|REF_DENSE=pauses/sec>1.0)"   : "ERROR",
                    "_is_degraded"                                   : False,
                })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    fp_col  = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"
    pos_col = "Med Pos Offset s (threshold≤0.20s)"
    dur_col = "Med Dur Ratio (pass band 0.75–1.25)"

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[~model_df["_is_degraded"] & (model_df[fp_col] != "ERROR")]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total  = len(clean_df)
        clean_pass   = (clean_df[fp_col] == "PASS").sum()
        clean_nm     = clean_df[fp_col].str.startswith("NEAR_MISS").sum()
        # Clean Pass Rate = PASS + REVIEW / non-degraded (REVIEW only in degraded so = PASS here)
        clean_pass_review = clean_pass  # REVIEW only appears in degraded segments

        # REVIEW count = degraded segments that got REVIEW verdict
        review_count = (degraded_df[fp_col] == "REVIEW").sum()

        near_miss_count = int(clean_nm)

        count_fails    = (model_df[fp_col].str.startswith("FAIL") & model_df[fp_col].str.contains("Count",    na=False)).sum()
        position_fails = (model_df[fp_col].str.startswith("FAIL") & model_df[fp_col].str.contains("Position", na=False)).sum()
        duration_fails = (model_df[fp_col].str.startswith("FAIL") & model_df[fp_col].str.contains("Duration", na=False)).sum()
        error_count    = (model_df[fp_col] == "ERROR").sum()

        valid    = model_df[model_df[pos_col].notna()]
        med_pos  = round(valid[pos_col].median(), 3) if len(valid) > 0 else None
        med_dur  = round(valid[dur_col].median(), 3)  if len(valid) > 0 else None

        summary_rows.append({
            "Model"                                      : model,
            "Total Segments"                             : total,
            "Clean Segments"                             : clean_total,
            "Clean Pass Rate (PASS / non-degraded)": f"{clean_pass_review}/{clean_total}" if clean_total > 0 else "—",
            "Near Miss (threshold exceeded ≤20% margin)"        : near_miss_count,
            "Review (ref dense, TTS rate OK)"            : int(review_count),
            "Degraded Segments"                          : len(degraded_df),
            "Count Fails"                                : count_fails,
            "Position Fails"                             : position_fails,
            "Duration Fails"                             : duration_fails,
            "Errors"                                     : error_count,
            "Median Pos Offset"                          : med_pos,
            "Median Dur Ratio"                           : med_dur,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"] = summary_df["Clean Pass Rate (PASS / non-degraded)"].apply(parse_rate)
    summary_df["_review_num"]     = summary_df["Review (ref dense, TTS rate OK)"]
    summary_df["_med_pos"]        = summary_df["Median Pos Offset"].fillna(999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_review_num", "_med_pos"],
        ascending=[False, False, True]
    ).drop(columns=["_clean_pass_num", "_review_num", "_med_pos"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    fp_col  = "Final Pass (PASS/NEAR_MISS/REVIEW/FAIL)"
    pos_col = "Med Pos Offset s (threshold≤0.20s)"
    dur_col = "Med Dur Ratio (pass band 0.75–1.25)"
    cnt_col = "Count Delta (threshold≤20)"
    flag_col = "Ref Flag (—=clean|REF_DENSE=pauses/sec>1.0)"

    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "Ref Pauses", "TTS Pauses", "Matched",
        "Unmatched Ref", "Unmatched TTS", cnt_col,
        pos_col, dur_col,
        "Count Pass", "Position Pass", "Duration Pass",
        fp_col, flag_col
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model",
        "Clean Pass Rate (PASS / non-degraded)",
        "Near Miss (threshold exceeded ≤20% margin)",
        "Review (ref dense, TTS rate OK)",
        "Count Fails", "Position Fails", "Duration Fails",
        "Median Pos Offset", "Median Dur Ratio"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate  → primary ranking — ref pauses/sec <= 1.0 only")
    print("NEAR_MISS        → within 20% beyond threshold — marginal failure")
    print("REVIEW           → ref too dense but TTS rate plausible")
    print("Count Fails      → TTS has wrong number of pauses")
    print("Position Fails   → pauses in wrong places — dramatic beats misaligned")
    print("Duration Fails   → pauses too short or too long")
    print(f"\nThresholds: Count ±{config.PAUSE_COUNT_THRESHOLD} | "
          f"Position <= {config.POSITION_OFFSET_THRESHOLD}s | "
          f"Duration ratio {config.VAD_DURATION_RATIO_MIN}–{config.VAD_DURATION_RATIO_MAX}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAD / Pause alignment gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "vad"))
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
