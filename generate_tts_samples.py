"""
generate_tts_samples.py — Generate TTS outputs from multiple SOTA models.

Writes to data/models/<model_name>/sample_{1..5}.wav

Models:
  - f5tts          : F5-TTS (zero-shot voice cloning from enrollment/speaker.wav)
  - kokoro         : Kokoro ONNX (high-quality English TTS, no cloning)
  - parler         : Parler-TTS mini (instruction-conditioned)
  - edge_tts_ava   : Microsoft edge-tts en-US-AvaNeural (neural, no cloning)
  - edge_tts_andrew: Microsoft edge-tts en-US-AndrewNeural (neural, no cloning)

Usage:
  conda activate base
  python generate_tts_samples.py
  python generate_tts_samples.py --models f5tts kokoro edge_tts_ava
"""

import asyncio
import os
import sys
import argparse
import subprocess
import numpy as np
import soundfile as sf

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "data", "models")
ENROLL     = os.path.join(ROOT, "data", "enrollment", "speaker.wav")

SENTENCES = [
    "She walked quietly through the empty hallway, pausing briefly at the door before stepping outside.",
    "The weather forecast predicted heavy rainfall throughout the weekend with possible thunderstorms by Sunday.",
    "He carefully placed the fragile glass on the shelf, making sure it would not fall during the night.",
    "The doctors announced that the patient had made a remarkable recovery after months of treatment.",
    "Despite the challenges they faced, the team managed to deliver the project ahead of schedule.",
]


def save_wav(audio: np.ndarray, sr: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if audio.ndim > 1:
        audio = audio.squeeze()
    sf.write(path, audio, sr)
    import soundfile as sf2
    info = sf2.info(path)
    print(f"    saved: {path}  ({info.duration:.2f}s)")


# ── F5-TTS ────────────────────────────────────────────────────────────────────

def generate_f5tts():
    print("\n=== F5-TTS ===")
    from f5_tts.api import F5TTS
    model = F5TTS()
    out_dir = os.path.join(MODELS_DIR, "f5tts")
    for i, text in enumerate(SENTENCES, 1):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        print(f"  [{i}/5] {text[:60]}...")
        wav, sr, _ = model.infer(
            ref_file=ENROLL,
            ref_text="",        # auto-transcribe
            gen_text=text,
            remove_silence=True,
        )
        save_wav(np.array(wav), sr, path)
    print("F5-TTS done.")


# ── Kokoro ONNX ───────────────────────────────────────────────────────────────

def generate_kokoro():
    print("\n=== Kokoro ONNX ===")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(
        os.path.join(ROOT, "kokoro-v1.0.onnx"),
        os.path.join(ROOT, "voices-v1.0.bin"),
    )
    out_dir = os.path.join(MODELS_DIR, "kokoro_v1")
    for i, text in enumerate(SENTENCES, 1):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        print(f"  [{i}/5] {text[:60]}...")
        samples, sr = kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
        save_wav(samples, sr, path)
    print("Kokoro done.")


# ── Parler-TTS ────────────────────────────────────────────────────────────────

def generate_parler():
    print("\n=== Parler-TTS mini ===")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    description = (
        "A female speaker with a slightly low-pitched voice delivers her words "
        "at a moderate speed with a very close recording that almost has no noise."
    )

    out_dir = os.path.join(MODELS_DIR, "parler_mini")
    for i, text in enumerate(SENTENCES, 1):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        print(f"  [{i}/5] {text[:60]}...")

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

        audio = gen.cpu().numpy().squeeze()
        sr = model.config.sampling_rate
        save_wav(audio, sr, path)
    print("Parler-TTS done.")


# ── Edge-TTS ──────────────────────────────────────────────────────────────────

def _generate_edge_tts(voice: str, out_dir: str):
    """Generate 5 samples with edge-tts (requires ffmpeg on PATH)."""
    try:
        import edge_tts
    except ImportError:
        print("edge-tts not installed: pip install edge-tts")
        return

    os.makedirs(out_dir, exist_ok=True)

    async def _run():
        for i, text in enumerate(SENTENCES, 1):
            path   = os.path.join(out_dir, f"sample_{i}.wav")
            tmp_mp3 = path.replace(".wav", "_tmp.mp3")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(tmp_mp3)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "22050", path],
                capture_output=True, check=True,
            )
            os.remove(tmp_mp3)
            info = sf.info(path)
            print(f"    saved: {path}  ({info.duration:.2f}s)")

    asyncio.run(_run())


def generate_edge_tts_ava():
    print("\n=== edge-tts AvaNeural ===")
    _generate_edge_tts("en-US-AvaNeural",
                       os.path.join(MODELS_DIR, "edge_tts_ava"))
    print("edge-tts AvaNeural done.")


def generate_edge_tts_andrew():
    print("\n=== edge-tts AndrewNeural ===")
    _generate_edge_tts("en-US-AndrewNeural",
                       os.path.join(MODELS_DIR, "edge_tts_andrew"))
    print("edge-tts AndrewNeural done.")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_MODELS = {
    "f5tts"           : generate_f5tts,
    "kokoro"          : generate_kokoro,
    "parler"          : generate_parler,
    "edge_tts_ava"    : generate_edge_tts_ava,
    "edge_tts_andrew" : generate_edge_tts_andrew,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*",
        choices=list(ALL_MODELS.keys()),
        default=list(ALL_MODELS.keys()),
        help="Models to generate (default: all)",
    )
    args = parser.parse_args()

    for name in args.models:
        try:
            ALL_MODELS[name]()
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\nAll done. Outputs in data/models/")
