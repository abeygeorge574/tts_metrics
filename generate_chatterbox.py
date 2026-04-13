"""
Generate voice-cloned audio using Chatterbox TTS.
Run from the tts_metrics project root.

Install first:
    pip install chatterbox-tts

Usage:
    python generate_chatterbox.py
"""

import os
import sys

SENTENCES = [
    "She walked quietly through the empty hallway, pausing briefly at the door before stepping outside.",
    "The weather forecast predicted heavy rainfall throughout the weekend with possible thunderstorms by Sunday.",
    "He carefully placed the fragile glass on the shelf, making sure it would not fall during the night.",
    "The doctors announced that the patient had made a remarkable recovery after months of treatment.",
    "Despite the challenges they faced, the team managed to deliver the project ahead of schedule.",
]

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEAKER_WAV = os.path.join(ROOT, "data", "enrollment", "speaker.wav")
OUTPUT_DIR  = os.path.join(ROOT, "data", "models", "chatterbox")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print("ERROR: chatterbox-tts not installed.")
        print("Run: pip install chatterbox-tts")
        sys.exit(1)

    if not os.path.exists(SPEAKER_WAV):
        print(f"ERROR: enrollment speaker not found: {SPEAKER_WAV}")
        sys.exit(1)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    LOCAL_WEIGHTS = os.path.join(ROOT, "weights", "chatterbox")
    print(f"Loading Chatterbox TTS from {LOCAL_WEIGHTS} on {device}...")
    model = ChatterboxTTS.from_local(LOCAL_WEIGHTS, device=device)

    for i, sentence in enumerate(SENTENCES, 1):
        out_path = os.path.join(OUTPUT_DIR, f"sample_{i}.wav")
        print(f"Generating sample_{i}.wav ...")
        wav = model.generate(
            text=sentence,
            audio_prompt_path=SPEAKER_WAV,
        )
        import torchaudio
        torchaudio.save(out_path, wav, model.sr)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Generated {len(SENTENCES)} files in {OUTPUT_DIR}")
    print("Now run: bash run_speaker_sim_cloning_test.sh")

if __name__ == "__main__":
    main()
