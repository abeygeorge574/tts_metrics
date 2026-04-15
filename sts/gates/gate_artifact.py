"""
STS Gate: Artifact Detection
Runs artifact detection on BOTH output (verdict) and input (baseline).
Comparing input vs output artifacts shows what the STS model introduced.

Output verdict: PASS / NEAR_MISS / FAIL (same logic as TTS gate).
Input baseline: diagnostic only — shows pre-existing artifacts in dubbing recording.
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

# Import pure computation functions from TTS artifact gate
from _tts_imports import get_compute_hnr, get_compute_pause_median_db, get_compute_spectral_artifact_score
compute_hnr                   = get_compute_hnr()
compute_pause_median_db       = get_compute_pause_median_db()
compute_spectral_artifact_score = get_compute_spectral_artifact_score()


def _artifact_verdict(hnr: dict, pause: dict, sa: dict, cfg) -> tuple[str, list]:
    """Returns (verdict, error_flags) using STS config thresholds."""
    errors  = []
    warns   = []

    # HNR check
    if hnr.get("hnr_mean") is not None:
        hnr_val = hnr["hnr_mean"]
        if hnr_val < cfg.ARTIFACT_HNR_ABS_THRESHOLD:
            nm_floor = cfg.ARTIFACT_HNR_ABS_THRESHOLD * (1 - cfg.ARTIFACT_NEAR_MISS_MARGIN)
            if hnr_val >= nm_floor:
                warns.append(f"WARN_HNR_LOW ({hnr_val:.1f}dB)")
            else:
                errors.append(f"ERR_VOICE_BUZZ ({hnr_val:.1f}dB)")

    # Pause floor check
    if not pause.get("silence_na") and pause.get("median_db") is not None:
        db = pause["median_db"]
        if db > cfg.ARTIFACT_SILENCE_FAIL_DB:
            errors.append(f"ERR_BACKGROUND_STATIC ({db:.1f}dBFS)")
        elif db > cfg.ARTIFACT_SILENCE_WARN_DB:
            warns.append(f"WARN_SILENCE_FLOOR ({db:.1f}dBFS)")

    # Spectral artifact check
    combined = sa.get("combined")
    if combined is not None:
        nm_ceil = cfg.ARTIFACT_COMBINED_THRESHOLD * (1 + cfg.ARTIFACT_NEAR_MISS_MARGIN)
        if combined > cfg.ARTIFACT_COMBINED_THRESHOLD:
            if combined <= nm_ceil:
                warns.append(f"WARN_SA_BORDERLINE ({combined:.3f})")
            else:
                errors.append(f"ERR_SPECTRAL_ARTIFACT ({combined:.3f})")

    all_flags = errors + warns
    if errors:
        verdict = "FAIL"
    elif warns:
        verdict = "NEAR_MISS"
    else:
        verdict = "PASS"

    return verdict, all_flags


def _run_one(audio_path: str) -> tuple[dict, dict, dict]:
    """Run all three artifact metrics on a single file."""
    hnr   = compute_hnr(audio_path)
    pause = compute_pause_median_db(audio_path)
    sa    = compute_spectral_artifact_score(audio_path)
    return hnr, pause, sa


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str) -> tuple:
    import soundfile as sf
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    results   = []

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION
        print(f"  [artifact] {sample}{' [SHORT]' if is_short else ''}")

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        # Run on output (verdict)
        out_hnr, out_pause, out_sa = _run_one(out_path)
        out_verdict, out_flags     = _artifact_verdict(out_hnr, out_pause, out_sa, config)

        # Run on input (baseline diagnostic)
        in_hnr, in_pause, in_sa = _run_one(in_path)
        in_verdict, in_flags    = _artifact_verdict(in_hnr, in_pause, in_sa, config)

        # Delta HNR (positive = output cleaner than input)
        out_hnr_val = out_hnr.get("hnr_mean")
        in_hnr_val  = in_hnr.get("hnr_mean")
        delta_hnr   = round(out_hnr_val - in_hnr_val, 2) if (out_hnr_val is not None and in_hnr_val is not None) else None

        final = "SHORT_SKIP" if is_short else out_verdict

        results.append({
            "Character"           : character,
            "Sample"              : sample,
            "Out_HNR_dB"          : out_hnr_val,
            "In_HNR_dB"           : in_hnr_val,
            "ΔHNR_dB (out-in)"    : delta_hnr,
            "Out_Pause_dBFS"      : out_pause.get("median_db"),
            "In_Pause_dBFS"       : in_pause.get("median_db"),
            "Out_SA_H1H2"         : out_sa.get("h1h2"),
            "Out_SA_CepMidQ"      : out_sa.get("cep_midq"),
            "Out_SA_SFM_HF"       : out_sa.get("sfm_hf"),
            "Out_SA_Combined"     : out_sa.get("combined"),
            "In_SA_Combined"      : in_sa.get("combined"),
            "Out_Flags"           : "; ".join(out_flags) if out_flags else "—",
            "In_Flags (baseline)" : "; ".join(in_flags)  if in_flags  else "—",
            "Final_Pass"          : final,
            "Flag"                : "SHORT" if is_short else "—",
        })
        print(f"    Out: {out_verdict} {out_flags} | In baseline: {in_verdict} {in_flags}")

    df = pd.DataFrame(results)

    total  = len(df)
    passes = (df["Final_Pass"] == "PASS").sum()
    nm     = (df["Final_Pass"] == "NEAR_MISS").sum()
    fails  = (df["Final_Pass"] == "FAIL").sum()

    summary = pd.DataFrame([{
        "Character"           : character,
        "Total"               : total,
        "PASS"                : passes,
        "NEAR_MISS"           : nm,
        "FAIL"                : fails,
        "Pass_Rate"           : f"{passes}/{total}",
        "Mean_Out_HNR_dB"     : round(df["Out_HNR_dB"].dropna().mean(), 2) if "Out_HNR_dB" in df else None,
        "Mean_In_HNR_dB"      : round(df["In_HNR_dB"].dropna().mean(), 2)  if "In_HNR_dB" in df else None,
        "Mean_Out_SA_Combined": round(df["Out_SA_Combined"].dropna().mean(), 4) if "Out_SA_Combined" in df else None,
        "Mean_In_SA_Combined" : round(df["In_SA_Combined"].dropna().mean(), 4)  if "In_SA_Combined" in df else None,
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [artifact] saved → {output_dir}")
