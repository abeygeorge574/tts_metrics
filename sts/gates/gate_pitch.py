"""
STS Gate: Pitch (two-axis)

Axis 1 — Preservation (output vs input):
  Contour correlation: Pearson r of F0 frame sequences (voiced frames only)
    PASS >= 0.70 | NEAR_MISS >= 0.55
  Std ratio (output/input): >= 0.70
  Range ratio (output/input): >= 0.60
  Absolute std floor (output): >= 20 Hz

Axis 2 — Register (output vs train enrollment):
  Median F0 delta |output_median - train_median| <= 30 Hz
  Train F0 computed from the long Train_Data WAV (chunked, averaged).

Final verdict:
  PASS      — both axes pass
  NEAR_MISS — one axis near-miss, neither hard fails
  REVIEW    — input unvoiced/degraded; only absolute floor checked
  FAIL      — any hard fail
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.interpolate import interp1d

_STS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

import config_sts as config
from _tts_imports import get_compute_pitch
compute_pitch = get_compute_pitch()


def _compute_f0_contour(audio_path: str, time_step: float = 0.01) -> np.ndarray | None:
    """
    Returns the full F0 array (0 = unvoiced frame) via PRAAT.
    Returns None on error.
    """
    try:
        import parselmouth
        snd   = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=60.0, pitch_ceiling=600.0)
        return pitch.selected_array["frequency"]
    except Exception:
        return None


def _contour_correlation(f0_input: np.ndarray, f0_output: np.ndarray) -> float | None:
    """
    Pearson r between F0 contours of input and output.
    Handles length mismatch via linear interpolation.
    Only uses frames where both are voiced (> 0).
    Returns None if < 10 shared voiced frames.
    """
    if f0_input is None or f0_output is None:
        return None

    n_in, n_out = len(f0_input), len(f0_output)

    if n_in != n_out:
        # Resample both to common length (min of the two)
        n_common = min(n_in, n_out)
        t_in     = np.linspace(0, 1, n_in)
        t_out    = np.linspace(0, 1, n_out)
        t_comm   = np.linspace(0, 1, n_common)
        f0_input  = interp1d(t_in,  f0_input,  kind="linear")(t_comm)
        f0_output = interp1d(t_out, f0_output, kind="linear")(t_comm)

    voiced = (f0_input > 0) & (f0_output > 0)
    if voiced.sum() < 10:
        return None

    f_in  = f0_input[voiced]
    f_out = f0_output[voiced]
    r     = float(np.corrcoef(f_in, f_out)[0, 1])
    return round(r, 3) if not np.isnan(r) else None


def _compute_train_median_f0(train_file: str, chunk_s: float = 15.0) -> float | None:
    """
    Compute the characteristic median F0 of the target voice from the train file.
    Long train file is chunked; F0 medians across chunks are averaged.
    """
    if not train_file or not os.path.exists(train_file):
        return None

    import tempfile, math
    data, sr_raw = sf.read(train_file, always_2d=False)
    dur = len(data) / sr_raw

    chunk_frames = int(chunk_s * sr_raw)
    n_chunks     = max(1, math.ceil(len(data) / chunk_frames))
    medians      = []

    with tempfile.TemporaryDirectory(prefix="pitch_train_") as tmpdir:
        for i in range(n_chunks):
            start  = i * chunk_frames
            end    = min((i + 1) * chunk_frames, len(data))
            chunk  = data[start:end]
            if (end - start) / sr_raw < 1.0:
                continue
            tmp_path = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
            sf.write(tmp_path, chunk, sr_raw)
            mean_hz, std_hz, range_hz, voiced_ratio = compute_pitch(tmp_path)
            if mean_hz is not None and voiced_ratio is not None and voiced_ratio >= 0.2:
                medians.append(mean_hz)

    if not medians:
        return None
    return round(float(np.median(medians)), 2)


def _check_high(value, threshold, margin):
    if value is None:
        return None
    if value >= threshold:
        return "PASS"
    if value >= threshold * (1 - margin):
        return "NEAR_MISS"
    return "FAIL"


def _check_low(value, threshold, margin):
    if value is None:
        return None
    if value <= threshold:
        return "PASS"
    if value <= threshold * (1 + margin):
        return "NEAR_MISS"
    return "FAIL"


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str) -> tuple:
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])

    NM = config.PITCH_NEAR_MISS_MARGIN

    # Pre-compute train F0 (one-time, shared across all segments)
    print(f"  [pitch] Computing train F0 from {os.path.basename(train_file) if train_file else 'N/A'} ...")
    train_f0_median = _compute_train_median_f0(train_file) if train_file else None
    print(f"  [pitch] Train median F0: {train_f0_median} Hz")

    results = []

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION
        print(f"  [pitch] {sample}{' [SHORT]' if is_short else ''}")

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        # Compute pitch stats for input and output
        in_mean,  in_std,  in_range,  in_voiced  = compute_pitch(in_path)
        out_mean, out_std, out_range, out_voiced  = compute_pitch(out_path)

        # Compute F0 contour correlation
        f0_in  = _compute_f0_contour(in_path)
        f0_out = _compute_f0_contour(out_path)
        corr_r = _contour_correlation(f0_in, f0_out)

        # Ratios (output / input)
        std_ratio   = round(out_std   / in_std,   3) if (in_std   and in_std   > 0 and out_std   is not None) else None
        range_ratio = round(out_range / in_range, 3) if (in_range and in_range > 0 and out_range is not None) else None

        # Register delta (output vs train)
        reg_delta = round(abs(out_mean - train_f0_median), 2) if (out_mean and train_f0_median) else None

        # ── Axis 1 verdicts (preservation) ────────────────────────────────────
        in_degraded = (in_voiced is None or in_voiced < 0.2)

        if not in_degraded:
            corr_r_v    = _check_high(corr_r,    config.PITCH_CONTOUR_CORR_THRESHOLD,  NM)
            std_ratio_v = _check_high(std_ratio,  config.PITCH_STD_RATIO_THRESHOLD,    NM)
            range_r_v   = _check_high(range_ratio, config.PITCH_RANGE_RATIO_MIN,        NM)
        else:
            corr_r_v    = None
            std_ratio_v = None
            range_r_v   = None

        # Absolute std floor (always)
        std_abs_v = _check_high(out_std, config.PITCH_STD_ABS_THRESHOLD, NM)

        # ── Axis 2 verdict (register) ─────────────────────────────────────────
        reg_v = _check_low(reg_delta, config.PITCH_REGISTER_DELTA_THRESHOLD,
                           config.PITCH_REGISTER_NEAR_MISS_MARGIN) if reg_delta is not None else None

        # ── Final verdict ──────────────────────────────────────────────────────
        if out_mean is None:
            final = "ERROR"
        elif in_degraded:
            # Can't do preservation axis — absolute sanity only
            abs_ok = (
                out_voiced is not None and out_voiced >= config.TTS_VOICED_ABS_MIN and
                out_std    is not None and out_std    >= config.TTS_PITCH_STD_ABS_MIN
            )
            final = "REVIEW" if abs_ok else "FAIL (Abs_Sanity)"
        else:
            all_verdicts = [v for v in [corr_r_v, std_ratio_v, range_r_v, std_abs_v, reg_v] if v is not None]
            if "FAIL" in all_verdicts:
                fail_labels = []
                if corr_r_v    == "FAIL": fail_labels.append("Contour_r")
                if std_ratio_v == "FAIL": fail_labels.append("Std_Ratio")
                if range_r_v   == "FAIL": fail_labels.append("Range_Ratio")
                if std_abs_v   == "FAIL": fail_labels.append("Std_Abs")
                if reg_v       == "FAIL": fail_labels.append("Register")
                final = f"FAIL ({', '.join(fail_labels)})"
            elif "NEAR_MISS" in all_verdicts:
                nm_labels = []
                if corr_r_v    == "NEAR_MISS": nm_labels.append("Contour_r")
                if std_ratio_v == "NEAR_MISS": nm_labels.append("Std_Ratio")
                if range_r_v   == "NEAR_MISS": nm_labels.append("Range_Ratio")
                if std_abs_v   == "NEAR_MISS": nm_labels.append("Std_Abs")
                if reg_v       == "NEAR_MISS": nm_labels.append("Register")
                final = f"NEAR_MISS ({', '.join(nm_labels)})"
            else:
                final = "PASS"

        results.append({
            "Character"                  : character,
            "Sample"                     : sample,
            "Out_Mean_Hz"                : out_mean,
            "Out_Std_Hz"                 : out_std,
            "Out_Range_Hz"               : out_range,
            "Out_Voiced_Ratio"           : out_voiced,
            "In_Mean_Hz"                 : in_mean,
            "In_Std_Hz"                  : in_std,
            "In_Range_Hz"                : in_range,
            "In_Voiced_Ratio"            : in_voiced,
            "Train_Median_Hz"            : train_f0_median,
            "Contour_Corr_r (out vs in)" : corr_r,
            "Std_Ratio (out/in)"         : std_ratio,
            "Range_Ratio (out/in)"       : range_ratio,
            "Register_Delta_Hz (out vs train)": reg_delta,
            "Contour_r_v"                : corr_r_v or "—",
            "Std_Ratio_v"                : std_ratio_v or "—",
            "Range_Ratio_v"              : range_r_v or "—",
            "Std_Abs_v"                  : std_abs_v or "—",
            "Register_v"                 : reg_v or "—",
            "Final_Pass"                 : final,
            "Flag"                       : "SHORT" if is_short else ("INPUT_DEGRADED" if in_degraded else "—"),
        })
        print(f"    r={corr_r} | std_ratio={std_ratio} | range_ratio={range_ratio} | reg_delta={reg_delta}Hz → {final}")

    df = pd.DataFrame(results)

    total  = len(df)
    passes = (df["Final_Pass"] == "PASS").sum()
    nm     = df["Final_Pass"].str.startswith("NEAR_MISS").fillna(False).sum()
    fails  = df["Final_Pass"].str.startswith("FAIL").fillna(False).sum()
    review = (df["Final_Pass"] == "REVIEW").sum()

    summary = pd.DataFrame([{
        "Character"          : character,
        "Total"              : total,
        "PASS"               : passes,
        "NEAR_MISS"          : nm,
        "REVIEW"             : review,
        "FAIL"               : fails,
        "Pass_Rate"          : f"{passes}/{total}",
        "Train_F0_Median_Hz" : train_f0_median,
        "Median_Contour_r"   : round(df["Contour_Corr_r (out vs in)"].dropna().median(), 3) if "Contour_Corr_r (out vs in)" in df else None,
        "Median_Std_Ratio"   : round(df["Std_Ratio (out/in)"].dropna().median(), 3)         if "Std_Ratio (out/in)" in df else None,
        "Median_Reg_Delta_Hz": round(df["Register_Delta_Hz (out vs train)"].dropna().median(), 2) if "Register_Delta_Hz (out vs train)" in df else None,
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [pitch] saved → {output_dir}")
