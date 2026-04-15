"""
STS Gate: Amplitude / Loudness
Compares output vs input: LUFS, LRA, spectral centroid, true peak.
Both sides are Hindi studio recordings — LUFS delta should be small.
Also flags input recordings with degraded loudness (very quiet or clipped).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
import pandas as pd

warnings.filterwarnings("ignore")

_STS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
import config_sts as config


def analyze_audio(file_path: str) -> dict:
    """Returns LUFS, LRA, spectral_centroid_Hz, peak_dBFS, clip_rate."""
    try:
        data, sr = librosa.load(file_path, sr=None, mono=False)
        if data.ndim == 1:
            data_pln = data.reshape(-1, 1)
        else:
            data_pln = data.T  # (ch, samples) → (samples, ch)

        meter = pyln.Meter(sr)
        lufs  = meter.integrated_loudness(data_pln)
        lra   = meter.loudness_range(data_pln)

        mono     = data if data.ndim == 1 else librosa.to_mono(data)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

        peak_amp  = float(np.max(np.abs(data_pln)))
        peak_db   = 20 * np.log10(peak_amp) if peak_amp > 0 else -100.0

        clip_amp  = 10 ** (config.PEAK_LIMIT / 20)
        clip_rate = float(np.mean(np.abs(data_pln) >= clip_amp))

        return {
            "lufs"    : round(lufs, 3),
            "lra"     : round(lra, 3),
            "centroid": round(centroid, 2),
            "peak_db" : round(peak_db, 3),
            "clip_rate": round(clip_rate, 6),
            "error"   : None,
        }
    except Exception as e:
        return {"lufs": None, "lra": None, "centroid": None,
                "peak_db": None, "clip_rate": None, "error": str(e)}


def run_gate(input_dir: str, output_dir: str, train_file: str | None, character: str) -> tuple:
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])

    LUFS_TOL  = config.LUFS_TOLERANCE
    LRA_TOL   = config.LRA_TOLERANCE
    CENT_TOL  = config.CENTROID_TOLERANCE
    NM_MARGIN = config.AMPLITUDE_NEAR_MISS_MARGIN
    results   = []

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,  wav_file)
        out_path = os.path.join(output_dir, wav_file)

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION

        inp = analyze_audio(in_path)
        out = analyze_audio(out_path)

        if inp["error"] or out["error"]:
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "ERROR", "Flag": inp["error"] or out["error"]})
            continue

        # Input quality check (degraded input → can't trust delta)
        in_lufs_degraded = (
            inp["lufs"] < config.REF_LUFS_MIN or inp["lufs"] > config.REF_LUFS_MAX
        )
        in_clipping = inp["clip_rate"] >= config.CLIP_RATE_WARN

        delta_lufs = round(out["lufs"]     - inp["lufs"],     3)
        delta_lra  = round(out["lra"]      - inp["lra"],      3)
        delta_cent = round(out["centroid"] - inp["centroid"], 2)

        flag = "—"
        if is_short:
            flag = "SHORT_SEGMENT"
        elif in_lufs_degraded:
            flag = "INPUT_LUFS_DEGRADED"
        elif in_clipping:
            flag = "INPUT_CLIPPED"

        is_degraded = flag not in ("—",)

        if is_degraded:
            # Absolute output sanity only
            out_lufs_ok = config.TTS_LUFS_ABS_MIN <= out["lufs"] <= config.TTS_LUFS_ABS_MAX
            out_lra_ok  = config.TTS_LRA_ABS_MIN  <= out["lra"]  <= config.TTS_LRA_ABS_MAX
            out_clip_ok = out["clip_rate"] < config.CLIP_RATE_WARN
            if out_lufs_ok and out_lra_ok and out_clip_ok:
                final = "REVIEW"
            else:
                fails = []
                if not out_lufs_ok: fails.append("LUFS_Abs")
                if not out_lra_ok:  fails.append("LRA_Abs")
                if not out_clip_ok: fails.append("Clipping")
                final = f"FAIL ({', '.join(fails)})"
        else:
            fail_reasons = []
            near_miss    = []

            # LUFS delta
            lufs_dev = abs(delta_lufs)
            if lufs_dev > LUFS_TOL * (1 + NM_MARGIN):
                fail_reasons.append("LUFS")
            elif lufs_dev > LUFS_TOL:
                near_miss.append("LUFS")

            # LRA delta
            lra_dev = abs(delta_lra)
            if lra_dev > LRA_TOL * (1 + NM_MARGIN):
                fail_reasons.append("LRA")
            elif lra_dev > LRA_TOL:
                near_miss.append("LRA")

            # Centroid delta
            cent_dev = abs(delta_cent)
            if cent_dev > CENT_TOL * (1 + NM_MARGIN):
                fail_reasons.append("Centroid")
            elif cent_dev > CENT_TOL:
                near_miss.append("Centroid")

            # Output clipping (absolute, always)
            if out["clip_rate"] >= config.CLIP_RATE_WARN:
                fail_reasons.append("Clipping")

            if fail_reasons:
                final = f"FAIL ({', '.join(fail_reasons)})"
            elif near_miss:
                final = f"NEAR_MISS ({', '.join(near_miss)})"
            else:
                final = "PASS"

        results.append({
            "Character"               : character,
            "Sample"                  : sample,
            "Input_LUFS"              : inp["lufs"],
            "Output_LUFS"             : out["lufs"],
            "ΔLUFS (out-in)"          : delta_lufs,
            "Input_LRA"               : inp["lra"],
            "Output_LRA"              : out["lra"],
            "ΔLRA (out-in)"           : delta_lra,
            "Input_Centroid_Hz"       : inp["centroid"],
            "Output_Centroid_Hz"      : out["centroid"],
            "ΔCentroid_Hz (out-in)"   : delta_cent,
            "Input_Peak_dBFS"         : inp["peak_db"],
            "Output_Peak_dBFS"        : out["peak_db"],
            "Output_ClipRate"         : out["clip_rate"],
            "Final_Pass"              : final,
            "Flag"                    : flag,
            "_is_degraded"            : is_degraded,
        })

    df = pd.DataFrame(results)

    total    = len(df)
    passes   = (df["Final_Pass"] == "PASS").sum()
    nm       = df["Final_Pass"].str.startswith("NEAR_MISS").fillna(False).sum()
    fails    = df["Final_Pass"].str.startswith("FAIL").fillna(False).sum()

    summary = pd.DataFrame([{
        "Character"  : character,
        "Total"      : total,
        "PASS"       : passes,
        "NEAR_MISS"  : nm,
        "FAIL"       : fails,
        "REVIEW"     : (df["Final_Pass"] == "REVIEW").sum(),
        "Pass_Rate"  : f"{passes}/{total}",
        "Mean_ΔLUFS" : round(df["ΔLUFS (out-in)"].dropna().mean(), 3) if "ΔLUFS (out-in)" in df else None,
        "Mean_ΔCentroid_Hz": round(df["ΔCentroid_Hz (out-in)"].dropna().mean(), 2) if "ΔCentroid_Hz (out-in)" in df else None,
    }])

    return df.drop(columns=["_is_degraded"], errors="ignore"), summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [amplitude] saved → {output_dir}")
