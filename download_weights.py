"""
Download all model weights required by the TTS evaluation pipeline.
Run once after cloning the repo — weights are gitignored (large binaries).

Usage:
    python download_weights.py

Downloads:
    weights/speaker_sim/   — SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb, ~87 MB)
    weights/chatterbox/    — Chatterbox TTS (ResembleAI/chatterbox, ~3 GB)
    weights/nisqa/         — NISQA MOS predictor (~40 MB, cloned from GitHub)

NISQA and UTMOS weights must be placed manually (see README).
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def download(url, dst, label):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"  [cached] {label}  ({os.path.getsize(dst) // 1024} KB)")
        return
    print(f"  downloading {label} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"done  ({os.path.getsize(dst) // 1024} KB)")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


def download_speaker_sim():
    """SpeechBrain spkrec-ecapa-voxceleb — speaker identity embeddings."""
    savedir = os.path.join(ROOT, "weights", "speaker_sim")
    os.makedirs(savedir, exist_ok=True)
    base = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main"
    files = {
        "embedding_model.ckpt" : f"{base}/embedding_model.ckpt",
        "mean_var_norm_emb.ckpt": f"{base}/mean_var_norm_emb.ckpt",
        "classifier.ckpt"      : f"{base}/classifier.ckpt",
    }
    print("Speaker-sim weights (SpeechBrain ECAPA-TDNN, ~87 MB):")
    for fname, url in files.items():
        download(url, os.path.join(savedir, fname), fname)


def download_chatterbox():
    """Chatterbox TTS weights — voice cloning model (~3 GB)."""
    savedir = os.path.join(ROOT, "weights", "chatterbox")
    os.makedirs(savedir, exist_ok=True)
    base = "https://huggingface.co/ResembleAI/chatterbox/resolve/main"
    files = {
        "ve.safetensors"      : f"{base}/ve.safetensors",
        "t3_cfg.safetensors"  : f"{base}/t3_cfg.safetensors",
        "s3gen.safetensors"   : f"{base}/s3gen.safetensors",
        "conds.pt"            : f"{base}/conds.pt",
    }
    print("Chatterbox TTS weights (~3 GB — this will take a while on slow connections):")
    for fname, url in files.items():
        download(url, os.path.join(savedir, fname), fname)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download pipeline model weights")
    parser.add_argument("--skip-chatterbox", action="store_true",
                        help="Skip Chatterbox download (3 GB — only needed for voice cloning generation)")
    args = parser.parse_args()

    download_speaker_sim()
    if not args.skip_chatterbox:
        download_chatterbox()
    else:
        print("Chatterbox skipped (--skip-chatterbox).")

    print("\nAll weights ready. You can now run: python run_pipeline.py")
