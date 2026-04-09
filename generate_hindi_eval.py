"""
generate_hindi_eval.py — Generate English TTS for Hindi episode segments,
then run the full pipeline against the Hindi source as reference.

Reads:  data/listen_test/transcriptions.json
        data/listen_test/hindi_files/<segment_id>.wav  (Hindi source = reference)

Writes: data/hindi_eval/models/<model>/<segment_id>.wav   (English TTS output)
        data/hindi_eval/reference/<segment_id>.wav         (copy of Hindi source)
        output/hindi_eval/<gate>/...                        (pipeline results)

Run from repo root in base (Python 3.13) env:
    python generate_hindi_eval.py
    python generate_hindi_eval.py --models gtts edge_tts_ava f5tts
    python generate_hindi_eval.py --skip-generate    # just run pipeline on existing files
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import argparse
import numpy as np
import soundfile as sf

ROOT         = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS  = os.path.join(ROOT, "data", "listen_test", "transcriptions.json")
HINDI_DIR    = os.path.join(ROOT, "data", "listen_test", "hindi_files")
MODELS_DIR   = os.path.join(ROOT, "data", "hindi_eval", "models")
REF_DIR      = os.path.join(ROOT, "data", "hindi_eval", "reference")
OUTPUT_DIR   = os.path.join(ROOT, "output", "hindi_eval")
ENROLL       = os.path.join(ROOT, "data", "enrollment", "speaker.wav")


# ── Load segments ─────────────────────────────────────────────────────────────

def load_segments():
    """
    Returns list of dicts:
        { "segment_id": str, "hindi_path": str, "english": str }
    segment_id is the wav filename without extension.
    """
    with open(TRANSCRIPTS) as f:
        data = json.load(f)

    segments = []
    for path, entry in data.items():
        fname      = os.path.basename(path)
        segment_id = os.path.splitext(fname)[0]
        hindi_path = os.path.join(HINDI_DIR, fname)
        if not os.path.exists(hindi_path):
            print(f"  WARNING: Hindi file not found: {hindi_path}")
            continue
        segments.append({
            "segment_id": segment_id,
            "hindi_path": hindi_path,
            "english"   : entry["english"],
        })
    return segments


# ── Reference setup ───────────────────────────────────────────────────────────

def setup_reference(segments):
    """Copy Hindi source files to data/hindi_eval/reference/ with matching names."""
    os.makedirs(REF_DIR, exist_ok=True)
    for seg in segments:
        dst = os.path.join(REF_DIR, seg["segment_id"] + ".wav")
        shutil.copy2(seg["hindi_path"], dst)
        print(f"  ref: {seg['segment_id']}.wav")
    print(f"Reference files ready in {REF_DIR}")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def save_wav(audio: np.ndarray, sr: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if audio.ndim > 1:
        audio = audio.squeeze()
    sf.write(path, audio, sr)
    info = sf.info(path)
    print(f"    saved: {os.path.basename(path)}  ({info.duration:.2f}s)")


def out_path(model_name, segment_id):
    return os.path.join(MODELS_DIR, model_name, segment_id + ".wav")


# ── TTS generators ────────────────────────────────────────────────────────────

def generate_gtts(segments):
    print("\n=== gTTS ===")
    from gtts import gTTS
    import tempfile, subprocess as sp
    for seg in segments:
        path = out_path("gtts", seg["segment_id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  {seg['segment_id']}")
        tts = gTTS(text=seg["english"], lang="en", slow=False)
        tmp = path.replace(".wav", "_tmp.mp3")
        tts.save(tmp)
        sp.run(["ffmpeg", "-y", "-i", tmp, "-ar", "22050", path],
               capture_output=True, check=True)
        os.remove(tmp)
        info = sf.info(path)
        print(f"    saved: {os.path.basename(path)}  ({info.duration:.2f}s)")
    print("gTTS done.")


def generate_edge_tts_ava(segments):
    print("\n=== edge-tts AvaNeural ===")
    _generate_edge_tts(segments, "en-US-AvaNeural", "edge_tts_ava")
    print("edge-tts AvaNeural done.")


def generate_edge_tts_andrew(segments):
    print("\n=== edge-tts AndrewNeural ===")
    _generate_edge_tts(segments, "en-US-AndrewNeural", "edge_tts_andrew")
    print("edge-tts AndrewNeural done.")


def _generate_edge_tts(segments, voice, model_name):
    try:
        import edge_tts
    except ImportError:
        print("edge-tts not installed: pip install edge-tts")
        return

    async def _run():
        for seg in segments:
            path    = out_path(model_name, seg["segment_id"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_mp3 = path.replace(".wav", "_tmp.mp3")
            print(f"  {seg['segment_id']}")
            communicate = edge_tts.Communicate(seg["english"], voice)
            await communicate.save(tmp_mp3)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "22050", path],
                capture_output=True, check=True,
            )
            os.remove(tmp_mp3)
            info = sf.info(path)
            print(f"    saved: {os.path.basename(path)}  ({info.duration:.2f}s)")

    asyncio.run(_run())


def generate_kokoro(segments):
    print("\n=== Kokoro ONNX ===")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(
        os.path.join(ROOT, "kokoro-v1.0.onnx"),
        os.path.join(ROOT, "voices-v1.0.bin"),
    )
    for seg in segments:
        path = out_path("kokoro", seg["segment_id"])
        print(f"  {seg['segment_id']}")
        samples, sr = kokoro.create(seg["english"], voice="af_heart", speed=1.0, lang="en-us")
        save_wav(samples, sr, path)
    print("Kokoro done.")


def generate_parler(segments):
    print("\n=== Parler-TTS mini ===")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    model     = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    description = (
        "A female speaker with a slightly low-pitched voice delivers her words "
        "at a moderate speed with a very close recording that almost has no noise."
    )

    for seg in segments:
        path = out_path("parler_mini", seg["segment_id"])
        print(f"  {seg['segment_id']}")
        input_ids  = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(seg["english"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = gen.cpu().numpy().squeeze()
        sr    = model.config.sampling_rate
        save_wav(audio, sr, path)
    print("Parler-TTS done.")


def generate_f5tts(segments):
    print("\n=== F5-TTS ===")
    from f5_tts.api import F5TTS
    model = F5TTS()
    for seg in segments:
        path = out_path("f5tts", seg["segment_id"])
        print(f"  {seg['segment_id']}")
        wav, sr, _ = model.infer(
            ref_file=ENROLL,
            ref_text="",
            gen_text=seg["english"],
            remove_silence=True,
        )
        save_wav(np.array(wav), sr, path)
    print("F5-TTS done.")


ALL_MODELS = {
    "gtts"           : generate_gtts,
    "edge_tts_ava"   : generate_edge_tts_ava,
    "edge_tts_andrew": generate_edge_tts_andrew,
    "kokoro"         : generate_kokoro,
    "parler_mini"    : generate_parler,
    "f5tts"          : generate_f5tts,
}


# ── Pipeline run ──────────────────────────────────────────────────────────────

def run_nisqa(models_dir, ref_dir, output_dir):
    """Run the NISQA gate against the Hindi eval directories."""
    print(f"\n{'='*60}")
    print("Running NISQA gate")
    print(f"  models : {models_dir}")
    print(f"  ref    : {ref_dir}")
    print(f"  output : {output_dir}")
    print(f"{'='*60}")

    sys.path.insert(0, ROOT)
    import config

    # Temporarily redirect config paths for this run
    orig_models = config.MODELS_DIR
    orig_ref    = config.REFERENCE_DIR

    config.MODELS_DIR    = models_dir
    config.REFERENCE_DIR = ref_dir

    try:
        from gates.gate_nisqa import load_model, run_gate, print_results, save_results
        model_state    = load_model()
        df, summary_df = run_gate(model_state)
        print_results(df, summary_df)
        nisqa_out = os.path.join(output_dir, "nisqa")
        save_results(df, summary_df, nisqa_out)
        print(f"\nNISQA results saved to {nisqa_out}")
    finally:
        config.MODELS_DIR    = orig_models
        config.REFERENCE_DIR = orig_ref

    return df, summary_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*",
        choices=list(ALL_MODELS.keys()),
        default=list(ALL_MODELS.keys()),
        help="Models to generate (default: all)",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip TTS generation, just run the pipeline on existing files",
    )
    args = parser.parse_args()

    print("Loading segments from transcriptions.json...")
    segments = load_segments()
    print(f"Found {len(segments)} segments:")
    for seg in segments:
        print(f"  {seg['segment_id']}")
        print(f"    EN: {seg['english'][:80]}{'...' if len(seg['english']) > 80 else ''}")

    print("\nSetting up Hindi reference files...")
    setup_reference(segments)

    if not args.skip_generate:
        for name in args.models:
            try:
                ALL_MODELS[name](segments)
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
                import traceback; traceback.print_exc()

    print("\n\nTTS generation complete. Running NISQA gate...")
    run_nisqa(MODELS_DIR, REF_DIR, OUTPUT_DIR)

    print(f"\nAll done. Results in {OUTPUT_DIR}")
