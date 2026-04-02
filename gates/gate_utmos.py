"""
Gate: UTMOS (naturalness MOS)
Env : utmos (python 3.9)
Scores TTS audio with the UTMOS model (checkpoint-based).
Returns scores in the 1–5 MOS range.
Device priority: CUDA → MPS (Apple Silicon) → CPU.
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Device detection ───────────────────────────────────────────────────────────
def _get_device():
    """Return the best available torch device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── lightning_fabric compatibility patch ──────────────────────────────────────
def _patch_lightning_fabric():
    """
    PyTorch >= 2.x sets weights_only=True by default in torch.load, which
    breaks lightning_fabric checkpoint loading.  Monkey-patch torch.load to
    always pass weights_only=False so the UTMOS checkpoint loads correctly
    without any manual sed edits.
    """
    try:
        import torch
        if getattr(torch, "_lightning_fabric_patched", False):
            return
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load
        torch._lightning_fabric_patched = True
    except Exception:
        pass


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import torch
    import torchaudio

    _patch_lightning_fabric()

    UTMOS_MODEL_DIR = config.UTMOS_MODEL_DIR
    UTMOS_CKPT      = config.UTMOS_CKPT

    if not os.path.exists(UTMOS_CKPT):
        raise FileNotFoundError(f"UTMOS checkpoint not found: {UTMOS_CKPT}")

    os.chdir(UTMOS_MODEL_DIR)
    if UTMOS_MODEL_DIR not in sys.path:
        sys.path.insert(0, UTMOS_MODEL_DIR)

    from score import Score

    device = _get_device()
    print(f"UTMOS device: {device}")

    scorer = Score(
        ckpt_path=UTMOS_CKPT,
        input_sample_rate=16000,
        device=device,
    )

    print(f"UTMOS model loaded from {UTMOS_CKPT}")
    print(f"Threshold: {config.UTMOS_THRESHOLD}")

    return {"scorer": scorer, "device": device}


# ── Score single file ──────────────────────────────────────────────────────────
_UTMOS_MAX_CHUNK_S = 10   # training distribution ~3-10s; also prevents O(n²) OOM
_UTMOS_MIN_S       = 1    # discard chunks shorter than this

def utmos_score(audio_path, scorer, device="cpu"):
    import math
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
    ).to(device)

    max_chunk_frames = int(_UTMOS_MAX_CHUNK_S * sr)
    min_frames       = int(_UTMOS_MIN_S       * sr)
    total_frames     = wav.shape[-1]

    if total_frames <= max_chunk_frames:
        # Short file — score in one shot
        return round(float(scorer.score(wav.to(device))[0]), 3)

    # Long file — split into N equal-length chunks (each ≤ max_chunk_frames)
    n_chunks    = math.ceil(total_frames / max_chunk_frames)
    chunk_size  = total_frames / n_chunks   # float → round at boundaries

    scores = []
    for i in range(n_chunks):
        start = round(i       * chunk_size)
        end   = round((i + 1) * chunk_size)
        chunk = wav[:, start:end]
        if chunk.shape[-1] < min_frames:
            continue
        scores.append(float(scorer.score(chunk.to(device))[0]))

    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    scorer = model_state["scorer"]
    device = model_state.get("device", "cpu")

    MODELS_DIR = config.MODELS_DIR

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
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            audio_path  = os.path.join(MODELS_DIR, model, wav_file)

            duration = sf.info(audio_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            flag     = "SHORT_SEGMENT" if is_short else "—"
            if is_short:
                print(f"\n  Sample : {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample : {sample_name}")

            score  = utmos_score(audio_path, scorer, device)
            passed = score >= UTMOS_THRESHOLD

            print(f"  UTMOS  : {score} → {'PASS' if passed else 'FAIL'}")

            results.append({
                "Model" : model,
                "Sample": sample_name,
                "UTMOS" : score,
                "Pass"  : "PASS" if passed else "FAIL",
                "Flag"  : flag,
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
