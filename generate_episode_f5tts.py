"""
Generate voice-cloned episode audio using F5-TTS.
Uses each Hindi reference segment as the voice prompt so the output
sounds like the original Hindi speaker saying the English text.

Run:  python generate_episode_f5tts.py
Output: data/hindi_eval/models/f5tts_cloned/
"""

import os, sys

# Clear SOCKS proxy before any HuggingFace imports
for _v in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_v, None)

ROOT       = os.path.dirname(os.path.abspath(__file__))
REF_DIR    = os.path.join(ROOT, "data", "hindi_eval", "reference")
TEXT_DIR   = os.path.join(ROOT, "data", "hindi_eval", "text_refs")
OUTPUT_DIR = os.path.join(ROOT, "data", "hindi_eval", "models", "f5tts_cloned")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_text(segment_name):
    txt_path = os.path.join(TEXT_DIR, segment_name + ".txt")
    if not os.path.exists(txt_path):
        print(f"  WARNING: no text ref for {segment_name}")
        return None
    with open(txt_path) as f:
        return f.read().strip()


def main():
    from f5_tts.api import F5TTS
    import soundfile as sf

    print("Loading F5-TTS...")
    tts = F5TTS()

    ref_wavs = sorted([f for f in os.listdir(REF_DIR) if f.endswith(".wav")])
    print(f"Found {len(ref_wavs)} reference segments.\n")

    for wav_file in ref_wavs:
        segment_name = os.path.splitext(wav_file)[0]
        ref_path     = os.path.join(REF_DIR, wav_file)
        out_path     = os.path.join(OUTPUT_DIR, wav_file)

        text = load_text(segment_name)
        if text is None:
            continue

        # F5TTS tensor mismatch above ~120 chars — truncate to first sentence
        if len(text) > 120:
            # take first sentence ending in punctuation
            for punct in ['.', '!', '?', '—']:
                idx = text.find(punct)
                if 20 < idx < 120:
                    text = text[:idx + 1].strip()
                    break
            else:
                text = text[:120].rsplit(" ", 1)[0]

        print(f"[{segment_name}]")
        print(f"  text  : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  voice : {wav_file}")

        wav, sr, _ = tts.infer(
            ref_file=ref_path,
            ref_text="सुनो, यह बहुत जरूरी है।",  # short Hindi placeholder — skips Whisper download
            gen_text=text,
        )
        sf.write(out_path, wav, sr)
        print(f"  saved : {out_path}\n")

    print(f"Done. {len(ref_wavs)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
