"""
STS Gate: VAD / Pause Alignment (soft gate — WARN, not hard FAIL)
Compares output pause structure vs input pause structure.
Same language: pauses should be preserved closely for lip sync.

Tighter than TTS thresholds:
  Position offset: <= 0.15s (vs TTS 0.20s)
  Duration ratio:  0.80 – 1.20 (vs TTS 0.75 – 1.25)

Verdict: PASS / NEAR_MISS / WARN (not FAIL — VAD is diagnostic for STS).
Segments where input has too many pauses/sec → DEGRADED_INPUT flag.
"""

from __future__ import annotations

import os
import sys
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_STS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

import config_sts as config


def _get_pauses(audio_path: str) -> tuple[list, float]:
    """ffmpeg silencedetect → list of pause dicts + audio duration."""
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", f"silencedetect=noise={config.SILENCE_DB}dB:d={config.MIN_SILENCE_DURATION}",
        "-f", "null", "-"
    ]
    try:
        res    = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=30)
        output = res.stderr
    except Exception as e:
        return [], 0.0

    duration = 0.0
    m = re.search(r"Duration: (\d+):(\d+):([\d\.]+)", output)
    if m:
        h, mn, s = m.groups()
        duration  = int(h) * 3600 + int(mn) * 60 + float(s)

    starts, ends = [], []
    for line in output.split("\n"):
        if "silence_start" in line:
            ms = re.search(r"silence_start: ([\d\.]+)", line)
            if ms:
                starts.append(float(ms.group(1)))
        elif "silence_end" in line:
            me = re.search(r"silence_end: ([\d\.]+)", line)
            if me:
                ends.append(float(me.group(1)))

    if len(ends) > len(starts):
        ends = ends[1:]

    pauses = [{"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)}
              for s, e in zip(starts, ends)]
    return pauses, duration


def _compare_pauses(ref_pauses: list, out_pauses: list) -> dict:
    """Hungarian algorithm matching with STS-specific thresholds."""
    rc = len(ref_pauses)
    oc = len(out_pauses)

    if rc == 0 or oc == 0:
        return {
            "ref_count": rc, "out_count": oc, "count_delta": abs(rc - oc),
            "matched_pairs": [], "position_offsets": [], "duration_ratios": [],
        }

    # Build cost matrix
    cost = np.full((rc, oc), np.inf)
    for i, rp in enumerate(ref_pauses):
        for j, op in enumerate(out_pauses):
            pos_diff = abs(rp["start"] - op["start"])
            if pos_diff > config.POSITION_HARD_LIMIT:
                continue
            pos_cost = (pos_diff / config.POSITION_SCALE) * config.POSITION_WEIGHT
            dur_ratio = op["duration"] / rp["duration"] if rp["duration"] > 0 else 1.0
            dur_cost  = abs(dur_ratio - 1.0) * config.DURATION_WEIGHT
            cost[i, j] = pos_cost + dur_cost

    # Hungarian assignment — replace inf with large sentinel so the matrix is
    # always solvable; filter out sentinel-assigned pairs after.
    cost_finite = np.where(np.isfinite(cost), cost, 1e9)
    row_ind, col_ind = linear_sum_assignment(cost_finite)
    pairs = [(r, c) for r, c in zip(row_ind, col_ind) if np.isfinite(cost[r, c])]

    pos_offsets  = [round(abs(ref_pauses[r]["start"] - out_pauses[c]["start"]), 3) for r, c in pairs]
    dur_ratios   = [round(out_pauses[c]["duration"] / ref_pauses[r]["duration"], 3)
                    if ref_pauses[r]["duration"] > 0 else 1.0 for r, c in pairs]

    return {
        "ref_count"      : rc,
        "out_count"      : oc,
        "count_delta"    : abs(rc - oc),
        "matched_pairs"  : pairs,
        "position_offsets": pos_offsets,
        "duration_ratios" : dur_ratios,
    }


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str) -> tuple:
    import soundfile as sf
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    results   = []

    NM = config.VAD_NEAR_MISS_MARGIN

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        in_pauses,  in_audio_dur  = _get_pauses(in_path)
        out_pauses, out_audio_dur = _get_pauses(out_path)

        # Dynamic count threshold (scaled by duration)
        count_threshold = min(
            config.PAUSE_COUNT_THRESHOLD,
            max(2, round(in_audio_dur * config.PAUSE_COUNT_RATE_THRESHOLD))
        )

        # Input density check
        in_rate = len(in_pauses) / in_audio_dur if in_audio_dur > 0 else 0
        is_degraded = in_rate > config.REF_PAUSES_PER_SECOND_LIMIT or is_short

        if is_degraded:
            flag = "SHORT" if is_short else f"DENSE_INPUT ({in_rate:.2f}/s)"
            results.append({
                "Character": character, "Sample": sample,
                "In_Pause_Count": len(in_pauses), "Out_Pause_Count": len(out_pauses),
                "Count_Threshold": count_threshold, "Count_Delta": abs(len(in_pauses) - len(out_pauses)),
                "Matched_Pairs": 0, "Median_Pos_Offset_s": None, "Median_Dur_Ratio": None,
                "Final_Pass": "DEGRADED_INPUT", "Flag": flag,
            })
            continue

        cmp = _compare_pauses(in_pauses, out_pauses)

        # Verdict components
        count_ok  = cmp["count_delta"] <= count_threshold
        n_matched = len(cmp["matched_pairs"])

        pos_offsets = cmp["position_offsets"]
        dur_ratios  = cmp["duration_ratios"]

        med_pos  = round(float(np.median(pos_offsets)), 3)  if pos_offsets else None
        med_dur  = round(float(np.median(dur_ratios)),  3)  if dur_ratios  else None

        warns = []
        if not count_ok:
            warns.append(f"COUNT_DELTA={cmp['count_delta']}(thresh={count_threshold})")

        if med_pos is not None:
            pos_thr = config.POSITION_OFFSET_THRESHOLD
            if med_pos > pos_thr * (1 + NM):
                warns.append(f"POSITION_DRIFT={med_pos}s")
            elif med_pos > pos_thr:
                warns.append(f"POSITION_BORDERLINE={med_pos}s")

        if med_dur is not None:
            dur_min = config.VAD_DURATION_RATIO_MIN
            dur_max = config.VAD_DURATION_RATIO_MAX
            nm_band = NM
            if med_dur < dur_min * (1 - nm_band) or med_dur > dur_max * (1 + nm_band):
                warns.append(f"DUR_RATIO={med_dur}")
            elif med_dur < dur_min or med_dur > dur_max:
                warns.append(f"DUR_RATIO_BORDERLINE={med_dur}")

        if not warns:
            final = "PASS"
        elif any("BORDERLINE" in w or "position_borderline" in w.lower() for w in warns) and len(warns) == 1:
            final = "NEAR_MISS"
        else:
            final = f"WARN ({'; '.join(warns)})"  # soft — not FAIL

        results.append({
            "Character"           : character,
            "Sample"              : sample,
            "In_Pause_Count"      : cmp["ref_count"],
            "Out_Pause_Count"     : cmp["out_count"],
            "Count_Threshold"     : count_threshold,
            "Count_Delta"         : cmp["count_delta"],
            "Matched_Pairs"       : n_matched,
            "Median_Pos_Offset_s" : med_pos,
            "Median_Dur_Ratio"    : med_dur,
            "Final_Pass"          : final,
            "Flag"                : "—",
        })
        print(f"  [vad] {sample}: count_delta={cmp['count_delta']} | pos={med_pos}s | dur_ratio={med_dur} → {final}")

    df = pd.DataFrame(results)

    total = len(df)
    passes = (df["Final_Pass"] == "PASS").sum()
    nm     = (df["Final_Pass"] == "NEAR_MISS").sum()
    warns  = df["Final_Pass"].str.startswith("WARN").fillna(False).sum()

    summary = pd.DataFrame([{
        "Character"            : character,
        "Total"                : total,
        "PASS"                 : passes,
        "NEAR_MISS"            : nm,
        "WARN"                 : warns,
        "DEGRADED_INPUT"       : (df["Final_Pass"] == "DEGRADED_INPUT").sum(),
        "Pass_Rate"            : f"{passes}/{total}",
        "Median_Pos_Offset_s"  : round(df["Median_Pos_Offset_s"].dropna().median(), 3) if "Median_Pos_Offset_s" in df else None,
        "Median_Dur_Ratio"     : round(df["Median_Dur_Ratio"].dropna().median(), 3)    if "Median_Dur_Ratio" in df else None,
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [vad] saved → {output_dir}")
