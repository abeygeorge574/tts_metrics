"""
Generate voice-cloned episode audio using Chatterbox TTS.
Uses each Hindi reference segment as the voice prompt so the output
sounds like the original Hindi speaker saying the English text.

Run from the tts_metrics project root:
    python generate_episode_cloned.py

Requires chatterbox-tts:
    pip install chatterbox-tts

Output goes to data/hindi_eval/models/chatterbox_cloned/
Then run:  bash run_speaker_sim_episode_cloned.sh
"""

import os
import sys

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "data", "hindi_eval", "models")
REF_DIR    = os.path.join(ROOT, "data", "hindi_eval", "reference")
TEXT_DIR   = os.path.join(ROOT, "data", "hindi_eval", "text_refs")
OUTPUT_DIR = os.path.join(MODELS_DIR, "chatterbox_cloned")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_text(segment_name):
    txt_path = os.path.join(TEXT_DIR, segment_name + ".txt")
    if not os.path.exists(txt_path):
        print(f"  WARNING: no text ref for {segment_name}")
        return None
    with open(txt_path) as f:
        return f.read().strip()


def main():
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print("ERROR: chatterbox-tts not installed.")
        print("Run: pip install chatterbox-tts")
        sys.exit(1)

    import torch, torchaudio
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    LOCAL_WEIGHTS = os.path.join(ROOT, "weights", "chatterbox")
    print(f"Loading Chatterbox TTS from {LOCAL_WEIGHTS} on {device}...")
    model = ChatterboxTTS.from_local(LOCAL_WEIGHTS, device=device)

    ref_wavs = sorted([f for f in os.listdir(REF_DIR) if f.endswith(".wav")])
    if not ref_wavs:
        print(f"ERROR: no reference wavs in {REF_DIR}")
        sys.exit(1)

    print(f"Found {len(ref_wavs)} reference segments.\n")

    for wav_file in ref_wavs:
        segment_name = os.path.splitext(wav_file)[0]
        ref_path     = os.path.join(REF_DIR, wav_file)
        out_path     = os.path.join(OUTPUT_DIR, wav_file)

        text = load_text(segment_name)
        if text is None:
            continue

        # Truncate very long text — chatterbox works best under ~200 chars per call
        if len(text) > 300:
            text = text[:300].rsplit(" ", 1)[0]

        print(f"[{segment_name}]")
        print(f"  text  : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  voice : {wav_file}")

        wav = model.generate(text=text, audio_prompt_path=ref_path)
        torchaudio.save(out_path, wav, model.sr)
        print(f"  saved : {out_path}\n")

    print(f"Done. {len(ref_wavs)} files in {OUTPUT_DIR}")
    print("Now run: bash run_speaker_sim_episode_cloned.sh")


if __name__ == "__main__":
    main()
