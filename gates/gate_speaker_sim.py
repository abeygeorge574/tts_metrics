"""
Gate: Speaker Similarity
Env : base (python 3.13)
Uses SpeechBrain ECAPA-TDNN to compute cosine similarity between
a reference (utterance or enrollment) and TTS audio.
Device priority: CUDA → MPS (Apple Silicon) → CPU.
"""

import os
import sys
import argparse

import torch
import torchaudio
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Device detection ───────────────────────────────────────────────────────────
def _get_device():
    """Return the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Compatibility patches ──────────────────────────────────────────────────────
# Two issues with newer huggingface_hub + SpeechBrain:
#
# 1. SpeechBrain calls hf_hub_download(use_auth_token=...) — removed in
#    huggingface_hub >= 0.24. Remap to the current `token` kwarg.
#
# 2. SpeechBrain tries to fetch custom.py from the HF repo. The repo no longer
#    has this file so it gets a 404. Patch speechbrain.utils.fetching.fetch to
#    return an empty temp file for any 404'd filename so loading continues.

def _patch_huggingface_hub():
    import tempfile
    import huggingface_hub

    if getattr(huggingface_hub, "_speechbrain_patched", False):
        return

    _orig = huggingface_hub.hf_hub_download

    def _patched(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig(*args, **kwargs)

    huggingface_hub.hf_hub_download = _patched
    huggingface_hub._speechbrain_patched = True


def _patch_speechbrain_fetch():
    import tempfile
    import speechbrain.utils.fetching as sb_fetching

    if getattr(sb_fetching, "_custom_py_patched", False):
        return

    _orig_fetch = sb_fetching.fetch

    def _patched_fetch(filename, *args, **kwargs):
        try:
            return _orig_fetch(filename, *args, **kwargs)
        except Exception as e:
            # If fetch fails with a 404 for custom.py, return a blank temp file.
            # SpeechBrain imports this file; an empty module is safe.
            # Return a Path object — SpeechBrain calls .parent on the result.
            if "custom.py" in str(filename) and ("404" in str(e) or "EntryNotFound" in type(e).__name__ or "RemoteEntryNotFound" in type(e).__name__):
                from pathlib import Path
                tmp = tempfile.NamedTemporaryFile(
                    suffix="_custom.py", delete=False, mode="w"
                )
                tmp.write("# auto-generated placeholder — custom.py not found in repo\n")
                tmp.close()
                return Path(tmp.name)
            raise

    sb_fetching.fetch = _patched_fetch
    sb_fetching._custom_py_patched = True


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import logging
    # Suppress SpeechBrain checkpoint hook registrations and internal path DEBUG lines
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    # Suppress httpx HTTP Request lines printed during HuggingFace Hub cache checks
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _patch_huggingface_hub()
    _patch_speechbrain_fetch()

    from speechbrain.inference.speaker import EncoderClassifier

    device = _get_device()
    print(f"Speaker similarity device: {device}")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    print("ECAPA-TDNN loaded.")
    return {"classifier": classifier, "device": device}


# ── Embedding and similarity functions ────────────────────────────────────────
def get_embedding(audio_path, classifier):
    signal, sr = torchaudio.load(audio_path)

    if signal.shape[-1] < 160:
        raise ValueError(f"Audio too short: {audio_path}")

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    return embedding.squeeze()


def cosine_similarity(emb1, emb2):
    emb1_norm = emb1 / torch.norm(emb1)
    emb2_norm = emb2 / torch.norm(emb2)
    return round(float(torch.dot(emb1_norm, emb2_norm)), 4)


def compute_speaker_sim(reference_path, tts_path, classifier):
    ref_emb = get_embedding(reference_path, classifier)
    tts_emb = get_embedding(tts_path, classifier)
    return cosine_similarity(ref_emb, tts_emb)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    classifier = model_state["classifier"]

    MODELS_DIR      = config.MODELS_DIR
    REFERENCE_DIR   = config.REFERENCE_DIR
    ENROLLMENT_FILE = os.path.join(config.ENROLLMENT_DIR, "speaker.wav")

    SPEAKER_SIM_THRESHOLD = config.SPEAKER_SIM_THRESHOLD

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    enrollment_available = os.path.exists(ENROLLMENT_FILE)
    if enrollment_available:
        print(f"Enrollment file found.")
    else:
        print(f"No enrollment file — segments without utterance ref will be skipped.")

    reference_available = os.path.exists(REFERENCE_DIR)
    if reference_available:
        ref_files = sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".wav")])
        print(f"Reference folder found: {len(ref_files)} utterance files")
    else:
        print(f"No reference folder — using enrollment for all segments.")

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

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)

            duration = sf.info(tts_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            if is_short:
                print(f"\n  Sample : {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample : {sample_name}")

            utterance_path = os.path.join(REFERENCE_DIR, wav_file) if reference_available else None

            if utterance_path and os.path.exists(utterance_path):
                reference_path = utterance_path
                ref_type       = "UTTERANCE_REF"
            elif enrollment_available:
                reference_path = ENROLLMENT_FILE
                ref_type       = "ENROLLMENT_REF"
            else:
                print(f"  No reference available — skipping")
                results.append({
                    "Model"   : model,
                    "Sample"  : sample_name,
                    "Score"   : None,
                    "Pass"    : "SKIP",
                    "Ref Type": "NO_REF",
                })
                continue

            score  = compute_speaker_sim(reference_path, tts_path, classifier)
            passed = score >= SPEAKER_SIM_THRESHOLD

            print(f"  Score  : {score} | {'PASS' if passed else 'FAIL'} | {ref_type}")

            results.append({
                "Model"   : model,
                "Sample"  : sample_name,
                "Score"   : score,
                "Pass"    : "PASS" if passed else "FAIL",
                "Ref Type": ref_type,
                "Flag"    : "SHORT_SEGMENT" if is_short else "—",
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df         = df[df["Model"] == model]
        valid_df         = model_df[model_df["Score"].notna()]
        scores           = valid_df["Score"]
        pass_count       = (model_df["Pass"] == "PASS").sum()
        total            = len(model_df)
        enrollment_count = (model_df["Ref Type"] == "ENROLLMENT_REF").sum()

        summary_rows.append({
            "Model"          : model,
            "Segments"       : total,
            "Mean Score"     : round(scores.mean(), 4)   if len(scores) > 0 else None,
            "Median Score"   : round(scores.median(), 4) if len(scores) > 0 else None,
            "Min Score"      : round(scores.min(), 4)    if len(scores) > 0 else None,
            "Pass Rate"      : f"{pass_count}/{total}",
            "Enrollment Rate": f"{enrollment_count}/{total}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"]       = summary_df["Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df["_enrollment_num"] = summary_df["Enrollment Rate"].apply(lambda x: int(x.split("/")[0]))

    summary_df = summary_df.sort_values(
        by=["_pass_num", "Median Score", "Min Score", "_enrollment_num"],
        ascending=[False, False, False, True]
    ).drop(columns=["_pass_num", "_enrollment_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[["Model", "Sample", "Score", "Pass", "Ref Type"]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Pass Rate", "Median Score", "Min Score", "Enrollment Rate"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Pass Rate      → primary ranking")
    print("Median Score   → typical speaker similarity")
    print("Min Score      → worst segment")
    print("Enrollment Rate→ segments using enrollment (less reliable than utterance ref)")
    print(f"\nThreshold: >= {config.SPEAKER_SIM_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker similarity gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "speaker_sim"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
