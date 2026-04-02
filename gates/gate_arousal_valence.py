"""
Gate: Arousal / Valence (dimensional emotion intensity)
Env : base (python 3.13)

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to measure
the *intensity* and *polarity* of emotion for both reference and output.

  Arousal  — energy level: high = excited/angry/tense, low = calm/flat/sleepy
  Valence  — tone polarity: high = positive/happy, low = negative/sad/angry
  Dominance — assertiveness: high = commanding, low = submissive

All three are [0, 1] continuous scores.

Pass = |ref_arousal - out_arousal| ≤ threshold
       AND
       |ref_valence - out_valence| ≤ threshold

This catches cases where SER says the emotion *category* is correct
but the *intensity* is wrong — e.g. reference is high-energy angry,
output is flat/low-energy angry.

Device priority: CUDA → MPS (Apple Silicon) → CPU.
"""

import os
import sys
import argparse
import logging

import json
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)


# ── Device detection ───────────────────────────────────────────────────────────
def _get_device():
    """Return best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Regression head (plain nn.Module — no PreTrainedModel dependency) ─────────
class _RegressionHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.dense    = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _EmotionModel(nn.Module):
    """wav2vec2 backbone + regression head for arousal/dominance/valence.

    Built as a plain nn.Module so it is not affected by PreTrainedModel
    API changes across transformers versions.
    """

    def __init__(self, backbone, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.wav2vec2   = backbone
        self.classifier = _RegressionHead(hidden_size, num_labels, dropout_prob)

    def forward(self, input_values):
        outputs       = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits        = self.classifier(hidden_states)
        return hidden_states, logits


# ── Config patch ───────────────────────────────────────────────────────────────
def _download_and_patch(model_name):
    """
    Download model to HF cache and patch vocab_size=null → 32 in config.json.

    Newer huggingface_hub strictly rejects None for typed fields.  This model's
    config.json ships with "vocab_size": null (it's a regression model with no
    vocabulary).  We fix the cached file in-place so standard from_pretrained
    works.  The patch is idempotent — only applied once on first download.
    """
    local_dir   = snapshot_download(model_name)
    config_path = os.path.join(local_dir, "config.json")

    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("vocab_size") is None:
        cfg["vocab_size"] = 32          # wav2vec2 standard; unused by regression head
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log.info("Patched vocab_size=null → 32 in cached config.json")

    return local_dir


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    device = _get_device()
    log.info("Arousal/Valence device: %s", device)
    log.info("Loading %s ...", MODEL_NAME)

    local_dir = _download_and_patch(MODEL_NAME)

    # Feature extractor for audio preprocessing
    processor = Wav2Vec2FeatureExtractor.from_pretrained(local_dir)

    # Load backbone using standard transformers (fully supported)
    backbone  = Wav2Vec2Model.from_pretrained(local_dir)

    # Read head dimensions from config
    hf_cfg    = Wav2Vec2Config.from_pretrained(local_dir)
    num_labels    = getattr(hf_cfg, "num_labels",    3)
    final_dropout = getattr(hf_cfg, "final_dropout", 0.1)
    hidden_size   = hf_cfg.hidden_size

    # Build composite model
    model = _EmotionModel(backbone, hidden_size, num_labels, final_dropout)

    # Load regression head weights from the full checkpoint
    ckpt_path = os.path.join(local_dir, "pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_state = {
        k.replace("classifier.", ""): v
        for k, v in state_dict.items()
        if k.startswith("classifier.")
    }
    model.classifier.load_state_dict(head_state, strict=True)

    model.to(device)
    model.eval()

    log.info("Dimensional emotion model loaded.")
    log.info("Arousal threshold  : ±%.2f", config.AROUSAL_DELTA_THRESHOLD)
    log.info("Valence threshold  : ±%.2f", config.VALENCE_DELTA_THRESHOLD)

    return {"processor": processor, "model": model, "device": device}


# wav2vec2 attention is O(n²) — cap chunk length to avoid OOM on long audio.
# Chunks are equal length (each ≤ max) so averaging isn't biased by a short tail.
_MAX_CHUNK_SAMPLES = 15 * 16000   # training distribution ~5-15s; also prevents OOM
_MIN_CHUNK_SAMPLES = 1  * 16000   # discard chunks shorter than 1 second


# ── Score a single audio file ──────────────────────────────────────────────────
def _score(audio_path, processor, model, device):
    """Return (arousal, dominance, valence) floats in [0, 1].

    Long audio is split into equal-length chunks (each ≤ 30 s) and scores
    are averaged, preventing the O(n²) attention OOM on MPS/GPU.
    """
    import math

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    wav_np       = wav.squeeze().numpy()
    total_samples = len(wav_np)

    if total_samples > _MAX_CHUNK_SAMPLES:
        n_chunks   = math.ceil(total_samples / _MAX_CHUNK_SAMPLES)
        chunk_size = total_samples / n_chunks   # float → round at boundaries
        chunks = [
            wav_np[round(i * chunk_size) : round((i + 1) * chunk_size)]
            for i in range(n_chunks)
        ]
        chunks = [c for c in chunks if len(c) >= _MIN_CHUNK_SAMPLES]
        if not chunks:
            chunks = [wav_np[:_MAX_CHUNK_SAMPLES]]
    else:
        chunks = [wav_np]

    all_scores = []
    for chunk in chunks:
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            _, logits = model(inputs["input_values"])

        all_scores.append(logits.squeeze().cpu().numpy())

    scores = np.mean(all_scores, axis=0)
    # model output order: [arousal, dominance, valence]
    return round(float(scores[0]), 4), round(float(scores[1]), 4), round(float(scores[2]), 4)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    processor = model_state["processor"]
    model     = model_state["model"]
    device    = model_state.get("device", "cpu")

    MODELS_DIR = config.MODELS_DIR
    REF_DIR    = config.REFERENCE_DIR

    AR_THRESH  = config.AROUSAL_DELTA_THRESHOLD
    VAL_THRESH = config.VALENCE_DELTA_THRESHOLD

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")
    if not os.path.exists(REF_DIR):
        raise FileNotFoundError(f"Reference folder not found: {REF_DIR}")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    log.info("Models found: %s", model_folders)

    model_samples = {}
    for m in model_folders:
        wav_files = sorted([
            f for f in os.listdir(os.path.join(MODELS_DIR, m))
            if f.endswith(".wav")
        ])
        model_samples[m] = wav_files
        log.info("   %s: %d samples", m, len(wav_files))

    results = []

    for m in model_folders:
        log.info("")
        log.info("=" * 50)
        log.info("Model: %s", m)
        log.info("=" * 50)

        for wav_file in model_samples[m]:
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            out_path    = os.path.join(MODELS_DIR, m, wav_file)
            ref_path    = os.path.join(REF_DIR, wav_file)

            duration = sf.info(out_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            if is_short:
                log.info("  Sample: %s [SHORT: %.2fs]", sample_name, duration)
            else:
                log.info("  Sample: %s", sample_name)

            if not os.path.exists(ref_path):
                log.warning("  Reference missing — skipping")
                results.append({
                    "Model"        : m,
                    "Sample"       : sample_name,
                    "Ref_Arousal"  : None, "Ref_Valence"  : None, "Ref_Dominance"  : None,
                    "Out_Arousal"  : None, "Out_Valence"  : None, "Out_Dominance"  : None,
                    "Delta_Arousal": None, "Delta_Valence": None,
                    "Pass"         : "SKIP",
                })
                continue

            # Score reference
            try:
                ref_ar, ref_dom, ref_val = _score(ref_path, processor, model, device)
            except Exception as e:
                log.error("  Reference scoring failed: %s", e)
                ref_ar = ref_dom = ref_val = None

            # Score output
            try:
                out_ar, out_dom, out_val = _score(out_path, processor, model, device)
            except Exception as e:
                log.error("  Output scoring failed: %s", e)
                out_ar = out_dom = out_val = None

            # Pass / fail
            if ref_ar is None or out_ar is None:
                passed    = "ERROR"
                delta_ar  = None
                delta_val = None
            else:
                delta_ar  = round(abs(ref_ar  - out_ar),  4)
                delta_val = round(abs(ref_val - out_val), 4)
                passed    = "PASS" if (delta_ar <= AR_THRESH and delta_val <= VAL_THRESH) else "FAIL"

            log.info(
                "  Arousal : ref=%.4f  out=%.4f  Δ=%s",
                ref_ar or 0, out_ar or 0, delta_ar,
            )
            log.info(
                "  Valence : ref=%.4f  out=%.4f  Δ=%s",
                ref_val or 0, out_val or 0, delta_val,
            )
            log.info("  → %s", passed)

            results.append({
                "Model"         : m,
                "Sample"        : sample_name,
                "Ref_Arousal"   : ref_ar,
                "Ref_Valence"   : ref_val,
                "Ref_Dominance" : ref_dom,
                "Out_Arousal"   : out_ar,
                "Out_Valence"   : out_val,
                "Out_Dominance" : out_dom,
                "Delta_Arousal" : delta_ar,
                "Delta_Valence" : delta_val,
                "Pass"          : passed,
                "Flag"          : "SHORT_SEGMENT" if is_short else "—",
                "_is_degraded"  : is_short,
            })

    log.info("")
    log.info("All evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for m in model_folders:
        model_df   = df[df["Model"] == m]
        clean_df   = model_df[model_df["Pass"].isin(["PASS", "FAIL"])]
        pass_count = (clean_df["Pass"] == "PASS").sum()
        total      = len(clean_df)

        summary_rows.append({
            "Model"              : m,
            "Segments"           : len(model_df),
            "Pass Rate"          : f"{pass_count}/{total}",
            "Mean_Delta_Arousal" : round(clean_df["Delta_Arousal"].dropna().mean(), 4) if total > 0 else None,
            "Mean_Delta_Valence" : round(clean_df["Delta_Valence"].dropna().mean(), 4) if total > 0 else None,
            "Mean_Ref_Arousal"   : round(clean_df["Ref_Arousal"].dropna().mean(), 4) if total > 0 else None,
            "Mean_Out_Arousal"   : round(clean_df["Out_Arousal"].dropna().mean(), 4) if total > 0 else None,
            "Mean_Ref_Valence"   : round(clean_df["Ref_Valence"].dropna().mean(), 4) if total > 0 else None,
            "Mean_Out_Valence"   : round(clean_df["Out_Valence"].dropna().mean(), 4) if total > 0 else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"] = summary_df["Pass Rate"].apply(
        lambda x: int(x.split("/")[0]) if "/" in str(x) else -1
    )
    summary_df = summary_df.sort_values(
        by=["_pass_num", "Mean_Delta_Arousal"],
        ascending=[False, True],
    ).drop(columns=["_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "Ref_Arousal", "Out_Arousal", "Delta_Arousal",
        "Ref_Valence", "Out_Valence", "Delta_Valence",
        "Pass",
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Delta_Arousal → energy/intensity gap  (high = TTS sounds flat vs expressive reference)")
    print("Delta_Valence → tone polarity gap      (high = TTS sounds wrong positive/negative feel)")
    print(f"\nThresholds: Arousal Δ ≤ {config.AROUSAL_DELTA_THRESHOLD},  Valence Δ ≤ {config.VALENCE_DELTA_THRESHOLD}")
    print("Both must pass. Either exceeding threshold → FAIL.")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    log.info("Results saved to %s", output_dir)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Arousal/Valence dimensional emotion gate")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.OUTPUT_DIR, "arousal_valence"),
    )
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
