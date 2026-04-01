"""
Gate: UTMOS (naturalness MOS)
Env : utmos (python 3.9)
Scores TTS audio with the UTMOS model (checkpoint-based, CPU).
Returns scores in the 1–5 MOS range.
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import torch
    import torchaudio

    UTMOS_MODEL_DIR = config.UTMOS_MODEL_DIR
    UTMOS_CKPT      = config.UTMOS_CKPT

    if not os.path.exists(UTMOS_CKPT):
        raise FileNotFoundError(f"UTMOS checkpoint not found: {UTMOS_CKPT}")

    os.chdir(UTMOS_MODEL_DIR)
    if UTMOS_MODEL_DIR not in sys.path:
        sys.path.insert(0, UTMOS_MODEL_DIR)

    from score import Score

    scorer = Score(
        ckpt_path=UTMOS_CKPT,
        input_sample_rate=16000,
        device="cpu"
    )

    print(f"UTMOS model loaded from {UTMOS_CKPT}")
    print(f"Threshold: {config.UTMOS_THRESHOLD}")

    return {"scorer": scorer}


# ── Score single file ──────────────────────────────────────────────────────────
def utmos_score(audio_path, scorer):
    import torch
    import torchaudio

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    scorer.in_sr = sr
    scorer.resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=16000,
        resampling_method="sinc_interpolation",
        lowpass_filter_width=6,
        dtype=torch.float32,
    )

    score = scorer.score(wav)
    return round(float(score[0]), 3)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    scorer = model_state["scorer"]

    BASE_DIR   = config.UTMOS_BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models found: {model_folders}")

    model_samples = {}
    for model in model_folders:
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])
        model_samples[model] = wav_files
        print(f"   {model}: {len(wav_files)} samples")

    reference_filenames = set(model_samples[model_folders[0]])
    for model in model_folders[1:]:
        current_filenames = set(model_samples[model])
        if current_filenames != reference_filenames:
            missing = reference_filenames - current_filenames
            extra   = current_filenames - reference_filenames
            raise ValueError(
                f"Model '{model}' has mismatched filenames.\n"
                f"  Missing : {missing}\n"
                f"  Extra   : {extra}"
            )
    print("All models have identical filenames.")

    sample_names = model_samples[model_folders[0]]
    total        = len(model_folders) * len(sample_names)
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = {total} evaluations")

    UTMOS_THRESHOLD = config.UTMOS_THRESHOLD
    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            audio_path  = os.path.join(MODELS_DIR, model, wav_file)

            print(f"\n  Sample : {sample_name}")

            score  = utmos_score(audio_path, scorer)
            passed = score >= UTMOS_THRESHOLD

            print(f"  UTMOS  : {score} → {'PASS' if passed else 'FAIL'}")

            results.append({
                "Model" : model,
                "Sample": sample_name,
                "UTMOS" : score,
                "Pass"  : "PASS" if passed else "FAIL",
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df   = df[df["Model"] == model]
        utmos_vals = model_df["UTMOS"]
        pass_count = (model_df["Pass"] == "PASS").sum()
        total      = len(model_df)

        summary_rows.append({
            "Model"       : model,
            "Segments"    : total,
            "Mean UTMOS"  : round(utmos_vals.mean(), 3),
            "Median UTMOS": round(utmos_vals.median(), 3),
            "Min UTMOS"   : round(utmos_vals.min(), 3),
            "Max UTMOS"   : round(utmos_vals.max(), 3),
            "Pass Rate"   : f"{pass_count}/{total}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"] = summary_df["Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df = summary_df.sort_values(
        by=["_pass_num", "Median UTMOS", "Min UTMOS"],
        ascending=[False, False, False]
    ).drop(columns=["_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[["Model", "Sample", "UTMOS", "Pass"]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Pass Rate", "Median UTMOS", "Min UTMOS", "Mean UTMOS"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Pass Rate    → % of segments production ready — primary decision metric")
    print("Median UTMOS → typical naturalness score — tiebreaker")
    print("Min UTMOS    → worst segment — how bad do failures get")
    print(f"\nThreshold: >= {config.UTMOS_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTMOS gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "utmos"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
