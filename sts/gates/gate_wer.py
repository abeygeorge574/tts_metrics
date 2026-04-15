"""
STS Gate: WER — Content Preservation
Transcribes both input (dubbing artist) and output (STS) with Whisper (Hindi).
Content preservation WER = WER(input_transcript, output_transcript).

In a perfect voice conversion, the words are identical → WER ≈ 0%.
High WER indicates the STS model mangled, dropped, or hallucinated words.

Runs as a single subprocess per character in utmos conda env (Python 3.9,
MLX Whisper on Apple Silicon). Model is loaded once per character, not per segment.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess

import soundfile as sf
import pandas as pd

_STS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)

import config_sts as config

_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wer_worker_hi.py")


def _find_utmos_python() -> str | None:
    """Find the utmos env Python interpreter."""
    candidates = [
        os.path.expanduser("~/miniconda3/envs/utmos/bin/python"),
        os.path.expanduser("~/anaconda3/envs/utmos/bin/python"),
        os.path.expanduser("~/opt/miniconda3/envs/utmos/bin/python"),
        os.path.expanduser("~/opt/anaconda3/envs/utmos/bin/python"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _run_wer_subprocess(input_dir: str, output_dir: str) -> list[dict]:
    """
    Call _wer_worker_hi.py once for the entire character directory.
    Returns list of result dicts (one per WAV file).
    """
    utmos_python = _find_utmos_python()

    if utmos_python:
        cmd = [utmos_python, _WORKER_SCRIPT,
               "--input-dir", input_dir, "--output-dir", output_dir]
    else:
        cmd = ["conda", "run", "-n", config.UTMOS_CONDA_ENV, "--no-capture-output",
               "python", _WORKER_SCRIPT,
               "--input-dir", input_dir, "--output-dir", output_dir]

    # Force CPU: MLX crashes with an uncatchable Objective-C NSRangeException when
    # the Metal GPU is unavailable (e.g. sandbox, GPU busy). CPU Whisper is reliable.
    env = os.environ.copy()
    env["WHISPER_FORCE_CPU"] = "1"

    try:
        # Per-character timeout: ~23 segs × 20s CPU ≈ 8 min + headroom
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600, env=env
        )
        if result.returncode != 0:
            return [{"sample": None, "wer": None, "wer_error": result.stderr[-500:]}]
        # Last non-empty line is the JSON array
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return [{"sample": None, "wer": None, "wer_error": "no output"}]
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        return [{"sample": None, "wer": None, "wer_error": "TIMEOUT"}]
    except json.JSONDecodeError as e:
        return [{"sample": None, "wer": None, "wer_error": f"JSON_PARSE: {e}"}]
    except Exception as e:
        return [{"sample": None, "wer": None, "wer_error": str(e)}]


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str) -> tuple:
    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])

    WER_THR = config.WER_THRESHOLD
    NM_THR  = WER_THR * (1 + config.WER_NEAR_MISS_MARGIN)

    print(f"  [wer] Running Whisper batch on {len(wav_files)} segments ...")
    batch = _run_wer_subprocess(input_dir, output_dir_data)

    # Index batch results by sample name
    batch_by_sample = {r["sample"]: r for r in batch if r.get("sample")}

    # Subprocess-level error (e.g. TIMEOUT, model fail)
    if len(batch) == 1 and batch[0].get("sample") is None:
        error_msg = batch[0].get("wer_error", "UNKNOWN")
        print(f"  [wer] Subprocess error: {error_msg}")
        results = []
        for wav_file in wav_files:
            sample = os.path.splitext(wav_file)[0]
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "ERROR", "Flag": error_msg})
        df = pd.DataFrame(results)
        summary = pd.DataFrame([{"Character": character, "Total": len(df),
                                  "PASS": 0, "NEAR_MISS": 0, "FAIL": 0,
                                  "ERROR": len(df), "Pass_Rate": f"0/{len(df)}",
                                  "Median_WER": None, "Max_WER": None}])
        return df, summary

    results = []
    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION

        res      = batch_by_sample.get(sample, {"wer": None, "wer_error": "NOT_IN_BATCH"})
        wer_val  = res.get("wer")
        wer_err  = res.get("wer_error")

        if wer_val is None:
            final = "ERROR"
            flag  = wer_err or "UNKNOWN"
        elif wer_val <= WER_THR:
            final = "PASS"
            flag  = "SHORT" if is_short else "—"
        elif wer_val <= NM_THR:
            final = "NEAR_MISS"
            flag  = "SHORT" if is_short else "—"
        else:
            final = "FAIL"
            flag  = "SHORT" if is_short else "—"

        results.append({
            "Character"               : character,
            "Sample"                  : sample,
            "Content_WER"             : wer_val,
            "Intelligibility (output)": res.get("intelligibility"),
            "Input_Transcript"        : res.get("in_text", ""),
            "Output_Transcript"       : res.get("out_text", ""),
            "Whisper_Engine"          : res.get("engine", ""),
            "Final_Pass"              : final,
            "Flag"                    : flag,
        })
        print(f"    {sample}: WER={wer_val} → {final}")

    df = pd.DataFrame(results)

    total    = len(df)
    passes   = (df["Final_Pass"] == "PASS").sum()
    nm       = (df["Final_Pass"] == "NEAR_MISS").sum()
    fails    = (df["Final_Pass"] == "FAIL").sum()
    wer_vals = df["Content_WER"].dropna()

    summary = pd.DataFrame([{
        "Character"  : character,
        "Total"      : total,
        "PASS"       : passes,
        "NEAR_MISS"  : nm,
        "FAIL"       : fails,
        "ERROR"      : (df["Final_Pass"] == "ERROR").sum(),
        "Pass_Rate"  : f"{passes}/{total}",
        "Median_WER" : round(wer_vals.median(), 4) if len(wer_vals) > 0 else None,
        "Max_WER"    : round(wer_vals.max(), 4)    if len(wer_vals) > 0 else None,
    }])

    return df, summary


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [wer] saved → {output_dir}")
