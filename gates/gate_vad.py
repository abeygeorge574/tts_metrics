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
    BASE_DIR      = config.VAD_BASE_DIR
    REFERENCE_DIR = os.path.join(BASE_DIR, "reference")
    MODELS_DIR    = os.path.join(BASE_DIR, "models")

    REF_PAUSES_PER_SECOND_LIMIT = config.REF_PAUSES_PER_SECOND_LIMIT

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

                failures = []
                if not cmp["count_pass"]:
                    failures.append("Count")
                if cmp["position_pass"] is False:
                    failures.append("Position")
                if cmp["duration_pass"] is False:
                    failures.append("Duration")

                final_pass = "PASS" if not failures else f"FAIL ({', '.join(failures)})"

                print(f"  Matched: {cmp['matched_count']} | "
                      f"Unmatched ref: {cmp['unmatched_ref']} | "
                      f"Unmatched tts: {cmp['unmatched_tts']}")
                print(f"  Result : {final_pass} | Count Δ: {cmp['count_delta']} | "
                      f"Pos: {cmp['med_position_offset']}s | Dur ratio: {cmp['med_duration_ratio']}")

                results.append({
                    "Model"          : model,
                    "Sample"         : sample_name,
                    "Ref Pauses"     : cmp["ref_count"],
                    "TTS Pauses"     : cmp["tts_count"],
                    "Matched"        : cmp["matched_count"],
                    "Unmatched Ref"  : cmp["unmatched_ref"],
                    "Unmatched TTS"  : cmp["unmatched_tts"],
                    "Count Delta"    : cmp["count_delta"],
                    "Med Pos Offset" : cmp["med_position_offset"],
                    "Med Dur Ratio"  : cmp["med_duration_ratio"],
                    "Count Pass"     : "PASS" if cmp["count_pass"] else "FAIL",
                    "Position Pass"  : "PASS" if cmp["position_pass"] else "FAIL" if cmp["position_pass"] is not None else "—",
                    "Duration Pass"  : "PASS" if cmp["duration_pass"] else "FAIL" if cmp["duration_pass"] is not None else "—",
                    "Final Pass"     : final_pass,
                    "Ref Flag"       : ref_flag,
                    "_is_degraded"   : is_degraded,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"         : model,
                    "Sample"        : sample_name,
                    "Ref Pauses"    : None,
                    "TTS Pauses"    : None,
                    "Matched"       : None,
                    "Unmatched Ref" : None,
                    "Unmatched TTS" : None,
                    "Count Delta"   : None,
                    "Med Pos Offset": None,
                    "Med Dur Ratio" : None,
                    "Count Pass"    : "—",
                    "Position Pass" : "—",
                    "Duration Pass" : "—",
                    "Final Pass"    : "ERROR",
                    "Ref Flag"      : "ERROR",
                    "_is_degraded"  : False,
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

        count_fails    = model_df["Final Pass"].str.contains("Count").sum()
        position_fails = model_df["Final Pass"].str.contains("Position").sum()
        duration_fails = model_df["Final Pass"].str.contains("Duration").sum()
        error_count    = (model_df["Final Pass"] == "ERROR").sum()

        valid    = model_df[model_df["Med Pos Offset"].notna()]
        med_pos  = round(valid["Med Pos Offset"].median(), 3) if len(valid) > 0 else None
        med_dur  = round(valid["Med Dur Ratio"].median(), 3)  if len(valid) > 0 else None

        summary_rows.append({
            "Model"             : model,
            "Total Segments"    : total,
            "Clean Segments"    : clean_total,
            "Clean Pass Rate"   : f"{clean_pass}/{clean_total}"  if clean_total > 0 else "—",
            "Degraded Segments" : deg_total,
            "Degraded Pass Rate": f"{deg_pass}/{deg_total}"      if deg_total > 0 else "—",
            "Count Fails"       : count_fails,
            "Position Fails"    : position_fails,
            "Duration Fails"    : duration_fails,
            "Errors"            : error_count,
            "Median Pos Offset" : med_pos,
            "Median Dur Ratio"  : med_dur,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df["_med_pos"]           = summary_df["Median Pos Offset"].fillna(999)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num", "_med_pos"],
        ascending=[False, False, True]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num", "_med_pos"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "Ref Pauses", "TTS Pauses", "Matched",
        "Unmatched Ref", "Unmatched TTS", "Count Delta",
        "Med Pos Offset", "Med Dur Ratio",
        "Count Pass", "Position Pass", "Duration Pass",
        "Final Pass", "Ref Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Degraded Pass Rate",
        "Count Fails", "Position Fails", "Duration Fails",
        "Median Pos Offset", "Median Dur Ratio"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate  → primary ranking — ref pauses/sec <= 1.0 only")
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
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "vad"))
    args = parser.parse_args()

    load_model()
    df, summary_df = run_gate()
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
