"""
NISQA Hindi Delta Experiment
============================
Scores all Hindi reference recordings and all English TTS model outputs
with NISQA. Computes cross-lingual deltas (English TTS − Hindi reference mean)
to understand how much of the NISQA score change is due to language mismatch
vs actual quality differences.

Run from the tts_metrics root (base env, Python 3.13):
    python notebooks/nisqa_hindi_delta_experiment.py

Output:
    output/nisqa_hindi_delta/hindi_scores.csv
    output/nisqa_hindi_delta/tts_scores.csv
    output/nisqa_hindi_delta/delta_summary.csv
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from gates.gate_nisqa import load_model, score_single_file

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "nisqa_hindi_delta")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HINDI_DIR  = config.REFERENCE_DIR   # data/reference/ — Hindi recordings
MODELS_DIR = config.MODELS_DIR      # data/models/    — English TTS outputs

METRICS = ["MOS", "Noisiness", "Discontinuity", "Coloration", "Loudness"]


def score_all_hindi(nisqa_weight):
    """Score every WAV in the Hindi reference directory."""
    import pandas as pd

    hindi_files = sorted([f for f in os.listdir(HINDI_DIR) if f.endswith(".wav")])
    if not hindi_files:
        raise FileNotFoundError(f"No WAV files in {HINDI_DIR}")

    rows = []
    print(f"\n{'='*60}")
    print("Scoring Hindi reference files")
    print(f"{'='*60}")
    for fname in hindi_files:
        fpath = os.path.join(HINDI_DIR, fname)
        print(f"\n  {fname}")
        try:
            scores = score_single_file(fpath, nisqa_weight)
            print(f"  MOS={scores['MOS']} Noi={scores['Noisiness']} "
                  f"Dis={scores['Discontinuity']} Col={scores['Coloration']} "
                  f"Lou={scores['Loudness']}")
            rows.append({"File": fname, **scores})
        except Exception as e:
            print(f"  SKIP: {e}")
            rows.append({"File": fname, **{m: None for m in METRICS}})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "hindi_scores.csv"), index=False)
    print(f"\nHindi scores saved → {OUTPUT_DIR}/hindi_scores.csv")
    return df


def score_all_tts(nisqa_weight):
    """Score all sample WAVs for each TTS model in MODELS_DIR."""
    import pandas as pd

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])

    rows = []
    print(f"\n{'='*60}")
    print("Scoring English TTS model outputs")
    print(f"{'='*60}")
    for model in model_folders:
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])
        print(f"\n{'─'*40}")
        print(f"Model: {model}")
        print(f"{'─'*40}")
        for fname in wav_files:
            fpath = os.path.join(model_path, fname)
            sample = os.path.splitext(fname)[0]
            print(f"  {sample}")
            try:
                scores = score_single_file(fpath, nisqa_weight)
                print(f"  MOS={scores['MOS']} Noi={scores['Noisiness']} "
                      f"Dis={scores['Discontinuity']} Col={scores['Coloration']} "
                      f"Lou={scores['Loudness']}")
                rows.append({"Model": model, "Sample": sample, **scores})
            except Exception as e:
                print(f"  SKIP: {e}")
                rows.append({"Model": model, "Sample": sample, **{m: None for m in METRICS}})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "tts_scores.csv"), index=False)
    print(f"\nTTS scores saved → {OUTPUT_DIR}/tts_scores.csv")
    return df


def build_delta_summary(hindi_df, tts_df):
    """
    Compute per-model mean TTS scores and delta vs mean Hindi reference.
    Also shows per-metric delta so you can see which dimensions change cross-lingually.
    """
    import pandas as pd

    # Hindi baseline: mean of all valid Hindi scores
    hindi_valid = hindi_df.dropna(subset=METRICS)
    if len(hindi_valid) == 0:
        raise ValueError("No valid Hindi scores to use as baseline.")

    hindi_mean = {m: round(hindi_valid[m].mean(), 3) for m in METRICS}
    print(f"\nHindi reference mean: {hindi_mean}")

    # Per-model mean TTS scores and delta
    rows = []
    for model in tts_df["Model"].unique():
        model_df    = tts_df[tts_df["Model"] == model].dropna(subset=METRICS)
        model_mean  = {m: round(model_df[m].mean(), 3) for m in METRICS}
        model_delta = {m: round(model_mean[m] - hindi_mean[m], 3) for m in METRICS}

        row = {"Model": model}
        for m in METRICS:
            row[f"TTS_{m}"]   = model_mean[m]
            row[f"Hindi_{m}"] = hindi_mean[m]
            row[f"Δ{m}"]      = model_delta[m]
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sort by ΔMOS descending (best quality delta first)
    summary_df = summary_df.sort_values("ΔMOS", ascending=False)

    out_path = os.path.join(OUTPUT_DIR, "delta_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nDelta summary saved → {out_path}")
    return summary_df, hindi_mean


def print_summary(summary_df, hindi_mean):
    print(f"\n{'='*80}")
    print("NISQA CROSS-LINGUAL DELTA: English TTS − Hindi Reference")
    print(f"{'='*80}")
    print(f"Hindi reference mean (n={len(hindi_mean)} metrics):")
    for m, v in hindi_mean.items():
        print(f"  {m:14s}: {v:.3f}")

    print(f"\n{'Model':<22} {'ΔMOS':>7} {'ΔNoi':>7} {'ΔDis':>7} {'ΔCol':>7} {'ΔLou':>7}")
    print("─" * 60)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<22} "
              f"{row['ΔMOS']:>+7.3f} "
              f"{row['ΔNoisiness']:>+7.3f} "
              f"{row['ΔDiscontinuity']:>+7.3f} "
              f"{row['ΔColoration']:>+7.3f} "
              f"{row['ΔLoudness']:>+7.3f}")

    print(f"\n{'='*80}")
    print("INTERPRETATION NOTE")
    print(f"{'='*80}")
    print("ΔNoisiness, ΔDiscontinuity, ΔLoudness: valid cross-lingual quality signals.")
    print("  Positive = TTS is cleaner / less distorted than the Hindi reference.")
    print("  Negative = TTS is noisier / more distorted.")
    print()
    print("ΔMOS, ΔColoration: carry a baseline offset from the language difference itself.")
    print("  Even a perfect English TTS will score differently from Hindi on MOS/Coloration")
    print("  because NISQA was trained on English TTS/VoIP. Treat as directional only.")
    print()
    print("Use this data to check:")
    print("  1. Do speecht5 models show strongly negative ΔNoisiness/ΔDiscontinuity?  (expect yes)")
    print("  2. Does f5tts/kokoro show near-zero or positive delta?                   (expect yes)")
    print("  3. Is the ΔMOS baseline offset consistent across all models?             (language mismatch artifact)")


if __name__ == "__main__":
    print("Loading NISQA model...")
    model_state  = load_model()
    nisqa_weight = model_state["nisqa_weight"]

    hindi_df = score_all_hindi(nisqa_weight)
    tts_df   = score_all_tts(nisqa_weight)

    summary_df, hindi_mean = build_delta_summary(hindi_df, tts_df)
    print_summary(summary_df, hindi_mean)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
