"""
STS Gate: Speaker Similarity (two-axis)

Primary (output vs train):
  Does the output sound like the trained target voice?
  PASS >= 0.50 | NEAR_MISS >= 0.40

Conversion check (output vs input):
  Did the conversion actually happen?
  Should be LOW (< 0.60). Above 0.70 → CONVERSION_WEAK warning.

The train file is long (~4 min). It is chunked into 10s segments and the
embedding is averaged (mean pooling of L2-normalised embeddings).
"""

from __future__ import annotations

import os
import sys
import math

import torch
import torchaudio
import soundfile as sf
import pandas as pd

_STS_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTS_ROOT = os.path.dirname(_STS_DIR)
if _STS_DIR not in sys.path:
    sys.path.insert(0, _STS_DIR)
if _TTS_ROOT not in sys.path:
    sys.path.insert(0, _TTS_ROOT)

import config_sts as config
from _tts_imports import get_speaker_sim_functions
load_model, get_embedding, cosine_similarity = get_speaker_sim_functions()


def _chunk_train_embedding(train_file: str, classifier, chunk_s: float = 10.0):
    """
    Compute a speaker embedding from the long train file by:
    1. Splitting into chunk_s second chunks
    2. Embedding each chunk
    3. Averaging the L2-normalised embeddings
    Returns a normalised 1D tensor.
    """
    data, sr = sf.read(train_file, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    chunk_frames = int(chunk_s * sr)
    n_chunks     = max(1, math.ceil(len(data) / chunk_frames))
    embeddings   = []

    import tempfile
    with tempfile.TemporaryDirectory(prefix="spkr_train_") as tmpdir:
        for i in range(n_chunks):
            start = i * chunk_frames
            end   = min((i + 1) * chunk_frames, len(data))
            chunk = data[start:end]
            if (end - start) / sr < 1.0:   # skip chunks < 1s
                continue
            tmp_path = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
            sf.write(tmp_path, chunk, sr)
            try:
                emb = get_embedding(tmp_path, classifier)
                # L2-normalise before pooling
                emb_norm = emb / torch.norm(emb)
                embeddings.append(emb_norm)
            except Exception:
                pass

    if not embeddings:
        return None

    # Mean pool of normalised embeddings, then re-normalise
    stacked = torch.stack(embeddings, dim=0)
    mean_emb = stacked.mean(dim=0)
    mean_emb = mean_emb / torch.norm(mean_emb)
    return mean_emb


def run_gate(input_dir: str, output_dir_data: str, train_file: str | None, character: str,
             model_state: dict | None = None) -> tuple:

    if model_state is None:
        model_state = load_model()

    classifier = model_state["classifier"]

    SIM_THR    = config.SPEAKER_SIM_THRESHOLD
    NM_MARGIN  = config.SPEAKER_SIM_NEAR_MISS_MARGIN
    NM_LOWER   = SIM_THR * (1 - NM_MARGIN)

    # Pre-compute train embedding
    print(f"  [speaker_sim] Computing train embedding from {os.path.basename(train_file) if train_file else 'N/A'} ...")
    train_emb = _chunk_train_embedding(train_file, classifier) if train_file else None
    if train_emb is not None:
        # Warn if train data was very short (< 30s → fewer than 3 chunks)
        try:
            train_dur = sf.info(train_file).duration if train_file else 0
            n_chunks_approx = int(train_dur // 10)
            if n_chunks_approx < 3:
                print(f"  [speaker_sim] WARNING: train file only ~{train_dur:.0f}s "
                      f"(~{n_chunks_approx} chunks) — embedding may be unreliable")
        except Exception:
            pass
        print(f"  [speaker_sim] Train embedding ready.")
    else:
        print(f"  [speaker_sim] WARNING: no train embedding — primary sim will be SKIP")

    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    results   = []

    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(input_dir,       wav_file)
        out_path = os.path.join(output_dir_data, wav_file)

        in_dur   = sf.info(in_path).duration
        is_short = in_dur < config.MIN_SEGMENT_DURATION

        if is_short:
            results.append({"Character": character, "Sample": sample,
                            "Sim_vs_Train (out vs train)": None,
                            "Sim_vs_Input (out vs in)": None,
                            "Train_Pass (PASS/NM/FAIL)": "SKIP",
                            "Conversion_Check": "SKIP",
                            "Final_Pass": "SKIP", "Flag": "SHORT"})
            continue

        if not os.path.exists(out_path):
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "NO_OUTPUT", "Flag": "MISSING_OUTPUT"})
            continue

        # Compute output embedding
        try:
            out_emb = get_embedding(out_path, classifier)
        except Exception as e:
            results.append({"Character": character, "Sample": sample,
                            "Final_Pass": "ERROR", "Flag": str(e)})
            continue

        # Primary: output vs train
        if train_emb is not None:
            sim_train = cosine_similarity(out_emb, train_emb)
            if sim_train >= SIM_THR:
                train_pass = "PASS"
            elif sim_train >= NM_LOWER:
                train_pass = "NEAR_MISS"
            else:
                train_pass = "FAIL"
        else:
            sim_train  = None
            train_pass = "SKIP (no train)"

        # Conversion check: output vs input (should be LOW)
        try:
            in_emb    = get_embedding(in_path, classifier)
            sim_input = cosine_similarity(out_emb, in_emb)
        except Exception:
            in_emb    = None
            sim_input = None

        if sim_input is not None:
            if sim_input >= config.CONVERSION_SIM_MAX:
                conversion_check = f"CONVERSION_WEAK ({sim_input:.3f})"
            elif sim_input >= config.CONVERSION_SIM_WARN:
                conversion_check = f"CONVERSION_BORDERLINE ({sim_input:.3f})"
            else:
                conversion_check = f"PASS ({sim_input:.3f})"
        else:
            conversion_check = "SKIP"

        # Final verdict: primary (output vs train) drives the gate.
        # CONVERSION_WEAK (sim_vs_input >= 0.70) means the STS model likely
        # didn't convert — output still sounds like the dubbing artist.
        # If conversion failed AND primary sim is already FAIL, keep FAIL.
        # If primary passes but conversion clearly failed, mark as NEAR_MISS minimum.
        final = train_pass
        if "CONVERSION_WEAK" in conversion_check:
            if final == "PASS":
                final = "NEAR_MISS+CONVERSION_WEAK"
            elif final == "NEAR_MISS":
                final = "NEAR_MISS+CONVERSION_WEAK"
            elif final == "FAIL":
                final = "FAIL+CONVERSION_WEAK"

        flag = "SHORT" if is_short else "—"

        results.append({
            "Character"                    : character,
            "Sample"                       : sample,
            "Sim_vs_Train (out vs train)"  : sim_train,
            "Sim_vs_Input (out vs in)"     : sim_input,
            "Train_Pass (PASS/NM/FAIL)"    : train_pass,
            "Conversion_Check"             : conversion_check,
            "Final_Pass"                   : final,
            "Flag"                         : flag,
        })
        print(f"    Sim_train={sim_train} ({train_pass}) | Sim_input={sim_input} | Conv: {conversion_check}")

    df = pd.DataFrame(results)

    total     = len(df)
    sim_col   = "Sim_vs_Train (out vs train)"
    iconv_col = "Sim_vs_Input (out vs in)"
    sims_t    = df[sim_col].dropna()
    sims_i    = df[iconv_col].dropna()

    passes = df["Train_Pass (PASS/NM/FAIL)"].str.startswith("PASS").fillna(False).sum()
    nm     = (df["Train_Pass (PASS/NM/FAIL)"] == "NEAR_MISS").sum()
    fails  = (df["Train_Pass (PASS/NM/FAIL)"] == "FAIL").sum()
    conv_w = df["Conversion_Check"].str.startswith("CONVERSION_WEAK").fillna(False).sum()

    summary = pd.DataFrame([{
        "Character"           : character,
        "Total"               : total,
        "PASS (vs_train)"     : passes,
        "NEAR_MISS (vs_train)": nm,
        "FAIL (vs_train)"     : fails,
        "Pass_Rate"           : f"{passes}/{total}",
        "Mean_Sim_vs_Train"   : round(sims_t.mean(), 4)   if len(sims_t) > 0 else None,
        "Median_Sim_vs_Train" : round(sims_t.median(), 4) if len(sims_t) > 0 else None,
        "Min_Sim_vs_Train"    : round(sims_t.min(), 4)    if len(sims_t) > 0 else None,
        "Mean_Sim_vs_Input"   : round(sims_i.mean(), 4)   if len(sims_i) > 0 else None,
        "Conversion_Weak_Count": conv_w,
    }])

    return df, summary, model_state


def save_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"  [speaker_sim] saved → {output_dir}")
