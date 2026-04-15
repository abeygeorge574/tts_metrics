"""
STS Gate: NISQA (MOS, Noisiness, Discontinuity, Coloration, Loudness)
Runs NISQA on BOTH input (dubbing artist) and output (STS), shows delta.
Input quality flag: if input fails NISQA_REF_THRESHOLDS → DEGRADED_INPUT.
Verdict on output: same absolute + delta logic as TTS gate.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_STS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

import config_sts as config

# NISQA package lives inside the weights dir — add to path so nisqa.NISQA_model is importable
if config.NISQA_REPO not in sys.path:
    sys.path.insert(0, config.NISQA_REPO)

# Import NISQA scoring from TTS gate (pure computation, no config dependency)
from _tts_imports import get_score_single_file
score_single_file = get_score_single_file()


def _absolute_pass(scores: dict) -> bool:
    return all(scores[k] >= config.NISQA_THRESHOLDS[k] for k in config.NISQA_THRESHOLDS)


def _delta_pass(deltas: dict) -> bool:
    return all(deltas[k] >= config.NISQA_DELTA_THRESHOLDS[k] for k in config.NISQA_DELTA_THRESHOLDS)


def _near_miss(scores: dict) -> bool:
    margin = 0.20
    for k, threshold in config.NISQA_THRESHOLDS.items():
        if scores[k] < threshold:
            if scores[k] < threshold * (1 - margin):
                return False
    return True


def _primary_failure(scores: dict | None, deltas: dict | None) -> str:
    if deltas:
        return min(deltas, key=deltas.get)
    if scores:
        dists = {k: scores[k] - config.NISQA_THRESHOLDS[k] for k in config.NISQA_THRESHOLDS}
        return min(dists, key=dists.get)
    return "—"


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str,
             nisqa_weight: str | None = None) -> tuple:

    if nisqa_weight is None:
        nisqa_weight = config.NISQA_WEIGHT
    if not os.path.exists(nisqa_weight):
        raise FileNotFoundError(f"NISQA weights not found: {nisqa_weight}")

    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    results   = []

    import soundfile as sf

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION
        print(f"  [nisqa] {sample}{' [SHORT]' if is_short else ''}")

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        # Score input
        try:
            inp_scores = score_single_file(in_path,  nisqa_weight)
        except Exception as e:
            inp_scores = None
            print(f"    input NISQA error: {e}")

        # Score output
        try:
            out_scores = score_single_file(out_path, nisqa_weight)
        except Exception as e:
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "ERROR", "Flag": str(e)})
            continue

        # Input quality check
        in_flag = "—"
        if is_short:
            in_flag = "SHORT_SEGMENT"
        elif inp_scores is not None:
            if any(inp_scores[k] < config.NISQA_REF_THRESHOLDS[k] for k in config.NISQA_REF_THRESHOLDS):
                failed = [k for k in config.NISQA_REF_THRESHOLDS if inp_scores[k] < config.NISQA_REF_THRESHOLDS[k]]
                in_flag = f"DEGRADED_INPUT ({', '.join(failed)})"
        elif inp_scores is None:
            in_flag = "INPUT_ERROR"

        is_degraded = in_flag not in ("—",)

        # Deltas (output - input)
        deltas = None
        if inp_scores is not None and not is_degraded:
            deltas = {k: round(out_scores[k] - inp_scores[k], 3) for k in out_scores}

        abs_pass = _absolute_pass(out_scores)

        if deltas is not None:
            dlt_pass = _delta_pass(deltas)
            if abs_pass:
                final = "PASS"
                prim  = "—"
            elif dlt_pass:
                final = "REVIEW"
                prim  = _primary_failure(None, deltas)
            elif _near_miss(out_scores):
                final = "NEAR_MISS"
                prim  = _primary_failure(out_scores, deltas)
            else:
                final = "FAIL"
                prim  = _primary_failure(out_scores, deltas)
        else:
            if abs_pass:
                final = "PASS"
                prim  = "—"
            elif _near_miss(out_scores):
                final = "NEAR_MISS"
                prim  = _primary_failure(out_scores, None)
            else:
                final = "FAIL"
                prim  = _primary_failure(out_scores, None)

        row = {
            "Character"       : character,
            "Sample"          : sample,
            "Out_MOS"         : out_scores["MOS"],
            "Out_Noisiness"   : out_scores["Noisiness"],
            "Out_Discontinuity": out_scores["Discontinuity"],
            "Out_Coloration"  : out_scores["Coloration"],
            "Out_Loudness"    : out_scores["Loudness"],
            "In_MOS"          : inp_scores["MOS"]          if inp_scores else None,
            "In_Noisiness"    : inp_scores["Noisiness"]    if inp_scores else None,
            "In_Discontinuity": inp_scores["Discontinuity"] if inp_scores else None,
            "In_Coloration"   : inp_scores["Coloration"]   if inp_scores else None,
            "In_Loudness"     : inp_scores["Loudness"]     if inp_scores else None,
            "ΔMOS"            : deltas["MOS"]          if deltas else None,
            "ΔNoisiness"      : deltas["Noisiness"]    if deltas else None,
            "ΔDiscontinuity"  : deltas["Discontinuity"] if deltas else None,
            "ΔColoration"     : deltas["Coloration"]   if deltas else None,
            "ΔLoudness"       : deltas["Loudness"]     if deltas else None,
            "Abs_Pass"        : "PASS" if abs_pass else "FAIL",
            "Final_Pass"      : final,
            "Primary_Failure" : prim,
            "Input_Flag"      : in_flag,
        }
        results.append(row)
        print(f"    Out MOS={out_scores['MOS']} | In MOS={inp_scores['MOS'] if inp_scores else 'N/A'} → {final}")

    df = pd.DataFrame(results)

    total  = len(df)
    passes = df["Final_Pass"].str.startswith("PASS").fillna(False).sum()
    nm     = df["Final_Pass"].str.startswith("NEAR_MISS").fillna(False).sum()
    fails  = df["Final_Pass"].str.startswith("FAIL").fillna(False).sum()
    review = (df["Final_Pass"] == "REVIEW").sum()

    summary = pd.DataFrame([{
        "Character"        : character,
        "Total"            : total,
        "PASS"             : passes,
        "NEAR_MISS"        : nm,
        "REVIEW"           : review,
        "FAIL"             : fails,
        "Pass_Rate"        : f"{passes}/{total}",
        "Median_Out_MOS"   : round(df["Out_MOS"].dropna().median(), 3) if "Out_MOS" in df else None,
        "Median_In_MOS"    : round(df["In_MOS"].dropna().median(), 3)  if "In_MOS" in df else None,
        "Mean_ΔMOS"        : round(df["ΔMOS"].dropna().mean(), 3)      if "ΔMOS" in df else None,
        "Degraded_Input_Count": df["Input_Flag"].str.startswith("DEGRADED_INPUT").fillna(False).sum() if "Input_Flag" in df else 0,
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [nisqa] saved → {output_dir}")
