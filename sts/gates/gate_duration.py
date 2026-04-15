"""
STS Gate: Duration
Compares output segment duration vs input segment duration.
Same language, same words — STS should preserve timing closely.
Threshold: ±10% (DURATION_TOLERANCE); calibrate down to ±5% after first run.
"""

from __future__ import annotations

import os
import sys

import soundfile as sf
import pandas as pd

_STS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
import config_sts as config


def run_gate(input_dir: str, output_dir: str, train_file: str | None, character: str) -> tuple:
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    if not wav_files:
        raise ValueError(f"No .wav files in input_dir: {input_dir}")

    TOL       = config.DURATION_TOLERANCE
    NM_MARGIN = config.DURATION_NEAR_MISS_MARGIN
    results   = []

    for wav_file in wav_files:
        sample = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,  wav_file)
        out_path = os.path.join(output_dir, wav_file)

        if not os.path.exists(out_path):
            results.append({
                "Character": character, "Sample": sample,
                "Input_Dur_s": None, "Output_Dur_s": None,
                "Dur_Ratio": None, "Dur_Delta_s": None,
                "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT",
            })
            continue

        in_dur  = round(sf.info(in_path).duration,  3)
        out_dur = round(sf.info(out_path).duration, 3)
        is_short = in_dur < config.MIN_SEGMENT_DURATION

        ratio = round(out_dur / in_dur, 4) if in_dur > 0 else None
        delta = round(out_dur - in_dur, 3) if in_dur > 0 else None

        if ratio is None:
            final = "ERROR"
        else:
            deviation = abs(ratio - 1.0)
            if deviation <= TOL:
                final = "PASS"
            elif deviation <= TOL * (1.0 + NM_MARGIN):
                final = "NEAR_MISS"
            else:
                final = "FAIL"

        results.append({
            "Character"     : character,
            "Sample"        : sample,
            "Input_Dur_s"   : in_dur,
            "Output_Dur_s"  : out_dur,
            "Dur_Ratio (out/in; PASS ±10%)" : ratio,
            "Dur_Delta_s"   : delta,
            "Final_Pass"    : final,
            "Flag"          : "SHORT" if is_short else "—",
        })

    df = pd.DataFrame(results)

    total      = len(df)
    passes     = (df["Final_Pass"] == "PASS").sum()
    near_miss  = (df["Final_Pass"] == "NEAR_MISS").sum()
    fails      = (df["Final_Pass"] == "FAIL").sum()
    short_cnt  = (df["Flag"] == "SHORT").sum()
    ratio_col  = "Dur_Ratio (out/in; PASS ±10%)"
    ratios     = df[ratio_col].dropna()
    mean_ratio = round(ratios.mean(), 4) if len(ratios) > 0 else None
    max_dev    = round((ratios - 1.0).abs().max(), 4) if len(ratios) > 0 else None

    summary = pd.DataFrame([{
        "Character"       : character,
        "Total"           : total,
        "PASS"            : passes,
        "NEAR_MISS"       : near_miss,
        "FAIL"            : fails,
        "Short_Segments"  : short_cnt,
        "Mean_Ratio"      : mean_ratio,
        "Max_Deviation"   : max_dev,
        "Pass_Rate"       : f"{passes}/{total}",
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [duration] saved → {output_dir}")
