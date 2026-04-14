"""
Gate: Arousal / Valence (dimensional emotion intensity)
Env : base (python 3.13)

Uses MERaLiON-SER-v1 dimensional output (same model as gate_ser.py) to measure
the energy level (arousal) and tone polarity (valence) of reference vs TTS audio.

  Arousal   — energy level: high = excited/angry/tense, low = calm/flat/sleepy  [0, 1]
  Valence   — tone polarity: high = positive/happy, low = negative/sad/angry    [0, 1]
  Dominance — assertiveness: high = commanding, low = submissive                [0, 1]

Pass/fail:
  PASS if |ref_arousal - out_arousal| ≤ AROUSAL_DELTA_THRESHOLD

Valence is recorded but NOT used in pass/fail:
  English-trained model reads Hindi prosody as systematically lower-valence,
  creating a ~0.10–0.25 gap regardless of TTS quality.

Device priority: MPS (Apple Silicon) → CUDA → CPU.
"""

import os
import sys
import argparse
import logging

# Proxy clearing — before any HF / httpx imports
for _pv in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_pv, None)

import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    os.environ.setdefault("HF_HOME", "/tmp/claude/hf_cache")

    from transformers import AutoModel, AutoProcessor

    local_path = config.MERALION_LOCAL_PATH
    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"MERaLiON model not found at {local_path}\n"
            f"Download with:\n"
            f"  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('MERaLiON/MERaLiON-SER-v1', local_dir='{local_path}')"
        )

    log.info("Loading MERaLiON-SER-v1 from %s ...", local_path)
    model = AutoModel.from_pretrained(
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    processor = AutoProcessor.from_pretrained(
        local_path,
        trust_remote_code=True,
    )
    model.eval()
    log.info("MERaLiON-SER-v1 loaded.")
    return {"model": model, "processor": processor}


# ── Score a single audio file ──────────────────────────────────────────────────
def _score(audio_path, model, processor):
    """Return (valence, arousal, dominance) floats in [0, 1]."""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    dims = outputs['dims'].squeeze().tolist()   # [valence, arousal, dominance]
    return round(dims[0], 4), round(dims[1], 4), round(dims[2], 4)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    model     = model_state["model"]
    processor = model_state["processor"]

    MODELS_DIR = model_state.get("models_dir") or config.MODELS_DIR
    REF_DIR    = model_state.get("ref_dir")    or config.REFERENCE_DIR

    AR_THRESH  = config.AROUSAL_DELTA_THRESHOLD

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
    log.info("Models: %s", model_folders)

    model_samples = {}
    for m in model_folders:
        wav_files = sorted([
            f for f in os.listdir(os.path.join(MODELS_DIR, m))
            if f.endswith(".wav")
        ])
        model_samples[m] = wav_files

    results = []

    for m in model_folders:
        log.info("=" * 50)
        log.info("Model: %s", m)

        for wav_file in model_samples[m]:
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            out_path    = os.path.join(MODELS_DIR, m, wav_file)
            ref_path    = os.path.join(REF_DIR, wav_file)

            duration = sf.info(out_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            log.info("  Sample: %s%s", sample_name, " [SHORT]" if is_short else "")

            if not os.path.exists(ref_path):
                log.warning("  Reference missing — skipping")
                results.append({
                    "Model"         : m, "Sample": sample_name,
                    "Ref_Arousal"   : None, "Ref_Valence"  : None, "Ref_Dominance"  : None,
                    "Out_Arousal"   : None, "Out_Valence"  : None, "Out_Dominance"  : None,
                    "Delta_Arousal" : None, "Delta_Valence": None,
                    "Pass"          : "SKIP", "Flag": "NO_REF",
                    "_is_degraded"  : False,
                })
                continue

            try:
                ref_val, ref_ar, ref_dom = _score(ref_path, model, processor)
            except Exception as e:
                log.error("  Reference scoring failed: %s", e)
                ref_val = ref_ar = ref_dom = None

            try:
                out_val, out_ar, out_dom = _score(out_path, model, processor)
            except Exception as e:
                log.error("  Output scoring failed: %s", e)
                out_val = out_ar = out_dom = None

            # Pass / fail — arousal only.
            # Valence excluded: cross-lingual Hindi→English bias produces a systematic
            # gap (~0.10–0.25) unrelated to TTS quality.
            if ref_ar is None or out_ar is None:
                passed    = "ERROR"
                delta_ar  = None
                delta_val = None
            else:
                delta_ar  = round(abs(ref_ar  - out_ar),  4)
                delta_val = round(abs(ref_val - out_val), 4)
                passed    = "PASS" if delta_ar <= AR_THRESH else "FAIL"

            log.info("  Arousal: ref=%.4f  out=%.4f  Δ=%s  → %s",
                     ref_ar or 0, out_ar or 0, delta_ar, passed)
            log.info("  Valence: ref=%.4f  out=%.4f  Δ=%s  (diagnostic)",
                     ref_val or 0, out_val or 0, delta_val)

            results.append({
                "Model"         : m, "Sample": sample_name,
                "Ref_Arousal"   : ref_ar,  "Ref_Valence"  : ref_val,  "Ref_Dominance"  : ref_dom,
                "Out_Arousal"   : out_ar,  "Out_Valence"  : out_val,  "Out_Dominance"  : out_dom,
                "Delta_Arousal" : delta_ar, "Delta_Valence": delta_val,
                "Pass"          : passed,
                "Flag"          : "SHORT_SEGMENT" if is_short else "—",
                "_is_degraded"  : is_short,
            })

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
            "Mean_Ref_Arousal"   : round(clean_df["Ref_Arousal"].dropna().mean(),   4) if total > 0 else None,
            "Mean_Out_Arousal"   : round(clean_df["Out_Arousal"].dropna().mean(),   4) if total > 0 else None,
            "Mean_Ref_Valence"   : round(clean_df["Ref_Valence"].dropna().mean(),   4) if total > 0 else None,
            "Mean_Out_Valence"   : round(clean_df["Out_Valence"].dropna().mean(),   4) if total > 0 else None,
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
    print("Delta_Arousal → energy/intensity gap  (PASS if ≤ threshold)")
    print("Delta_Valence → tone polarity gap      (diagnostic only — not in pass/fail)")
    print(f"\nPass threshold: Arousal Δ ≤ {config.AROUSAL_DELTA_THRESHOLD}")
    print("Valence excluded from pass/fail: Hindi reference reads ~0.10–0.25 lower-valence")
    print("to an English-trained model regardless of TTS quality (cross-lingual bias).")
    print("\nModel: MERaLiON-SER-v1 dims output [valence, arousal, dominance]")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.drop(columns=["_is_degraded"], errors="ignore").to_csv(
        os.path.join(output_dir, "per_segment_results.csv"), index=False
    )
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    log.info("Results saved to %s", output_dir)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Arousal/Valence gate — MERaLiON-SER-v1 dims")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "arousal_valence"))
    parser.add_argument("--models-dir", default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",    default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = load_model()
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)

    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
