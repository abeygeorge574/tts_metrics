"""
STS Gate: Speech Emotion Recognition (SER)
Measures emotion TRANSFER: does the STS output preserve the dubbing artist's emotion?
Comparison axis: output vs input (same language — more reliable than TTS cross-lingual SER).

MERaLiON-SER-v1 (Whisper-Medium + LoRA + ECAPA-TDNN)
Hindi is cross-lingual for MERaLiON (trained on en/zh/ms/ta/id/th/vi), but since
both sides have the same cross-lingual bias, the delta comparison is valid.

Pass/fail: same top-2 overlap logic as TTS gate.
  PASS      — output top-1 == input top-1
  NEAR_MISS — top-2 label sets intersect
  REVIEW    — input confidence < 0.5 but output confidence >= 0.5
  FAIL      — no label overlap
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_STS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

import config_sts as config
from _tts_imports import get_ser_functions
_load_ser_model, get_emotion = get_ser_functions()


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str,
             model_state: dict | None = None) -> tuple:

    import soundfile as sf

    if model_state is None:
        model_state = _load_ser_model()

    model     = model_state["model"]
    processor = model_state["processor"]

    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    results   = []
    CONF_THR  = config.SER_CONFIDENCE_THRESHOLD

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION
        print(f"  [ser] {sample}{' [SHORT]' if is_short else ''}")

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        # Get emotion for input and output
        in_t1, in_c1, in_t2, in_c2, in_val, in_aro, in_dom = get_emotion(in_path,  model, processor)
        out_t1, out_c1, out_t2, out_c2, out_val, out_aro, out_dom = get_emotion(out_path, model, processor)

        if in_t1 is None or out_t1 is None:
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "ERROR", "Flag": "MERALION_ERROR"})
            continue

        # Arousal delta
        aro_delta = round(abs(out_aro - in_aro), 4) if (out_aro is not None and in_aro is not None) else None

        # Pass/fail logic (top-2 overlap, same as TTS)
        in_top2  = {in_t1, in_t2}
        out_top2 = {out_t1, out_t2}
        in_degraded  = in_c1 < CONF_THR
        out_degraded = out_c1 < CONF_THR

        if in_degraded:
            if out_degraded:
                final = "FAIL (both_low_conf)"
            else:
                final = "REVIEW"
        elif in_t1 == out_t1:
            final = "PASS"
        elif in_top2 & out_top2:
            final = "NEAR_MISS"
        else:
            final = "FAIL"

        results.append({
            "Character"    : character,
            "Sample"       : sample,
            "In_Emotion"   : in_t1,
            "In_Conf"      : in_c1,
            "In_Emotion2"  : in_t2,
            "In_Conf2"     : in_c2,
            "Out_Emotion"  : out_t1,
            "Out_Conf"     : out_c1,
            "Out_Emotion2" : out_t2,
            "Out_Conf2"    : out_c2,
            "In_Arousal"   : in_aro,
            "Out_Arousal"  : out_aro,
            "ΔArousal"     : aro_delta,
            "In_Valence (diag)" : in_val,
            "Out_Valence (diag)": out_val,
            "Final_Pass"   : final,
            "Flag"         : "SHORT" if is_short else "—",
        })
        print(f"    In: {in_t1}({in_c1:.2f}) | Out: {out_t1}({out_c1:.2f}) → {final} | ΔAro={aro_delta}")

    df = pd.DataFrame(results)

    total  = len(df)
    passes = (df["Final_Pass"] == "PASS").sum()
    nm     = (df["Final_Pass"] == "NEAR_MISS").sum()
    review = (df["Final_Pass"] == "REVIEW").sum()
    fails  = df["Final_Pass"].str.startswith("FAIL").fillna(False).sum()

    summary = pd.DataFrame([{
        "Character"      : character,
        "Total"          : total,
        "PASS"           : passes,
        "NEAR_MISS"      : nm,
        "REVIEW"         : review,
        "FAIL"           : fails,
        "Pass_Rate"      : f"{passes}/{total}",
        "Mean_ΔArousal"  : round(df["ΔArousal"].dropna().mean(), 4) if "ΔArousal" in df else None,
        "Top_In_Emotion" : df["In_Emotion"].value_counts().index[0] if "In_Emotion" in df and len(df) > 0 else None,
        "Top_Out_Emotion": df["Out_Emotion"].value_counts().index[0] if "Out_Emotion" in df and len(df) > 0 else None,
    }])

    return df, summary, model_state


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [ser] saved → {output_dir}")
