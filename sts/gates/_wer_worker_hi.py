"""
WER worker for STS pipeline — runs in utmos conda env (Python 3.9).
Transcribes both input and output with Whisper (language=hi).
Computes content preservation WER: WER(input_transcript, output_transcript).

Processes an entire character directory at once (model loaded once).
Prints a JSON list of result dicts to stdout, one per WAV file.

Usage:
  python _wer_worker_hi.py --input-dir /path/Recorded_Segments \
                            --output-dir /path/Converted_Segments
"""

import os
import sys
import json
import argparse
import re
import math

import jiwer


# ── Hindi text normalisation ─────────────────────────────────────────────────
_HINDI_PUNCT_RE = re.compile(r"[।॥,.!?;:\"'()\[\]{}\-—–\u0964\u0965]+")

def normalise_hi(text):
    text = _HINDI_PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ── Whisper engine detection ──────────────────────────────────────────────────
def load_engine():
    import torch
    if torch.cuda.is_available():
        import whisper as openai_whisper
        model = openai_whisper.load_model("medium", device="cuda")
        return "cuda", model
    try:
        if not os.environ.get("WHISPER_FORCE_CPU"):
            import mlx_whisper
            return "mlx", None
        raise ImportError
    except ImportError:
        import whisper as openai_whisper
        model = openai_whisper.load_model("medium", device="cpu")
        return "cpu", model


def transcribe(audio_path, engine, model):
    """Returns {'text': str, 'segments': list} or {'text': '', 'segments': [], 'error': str}."""
    try:
        # no_speech_threshold: return empty rather than hallucinating on silence/noise.
        # condition_on_previous_text=False: prevents repetition loops on garbled audio.
        _kwargs = dict(
            language="hi",
            word_timestamps=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        if engine == "cuda":
            result = model.transcribe(audio_path, fp16=True, **_kwargs)
        elif engine == "mlx":
            import mlx_whisper
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo="mlx-community/whisper-medium-mlx",
                **_kwargs,
            )
        else:  # cpu
            result = model.transcribe(audio_path, fp16=False, **_kwargs)
        return {"text": result.get("text", "").strip(), "segments": result.get("segments", [])}
    except Exception as e:
        return {"text": "", "segments": [], "error": str(e)}


def word_intelligibility(segments):
    """Fraction of output words above log-prob threshold."""
    log_probs = []
    for seg in segments:
        for word in seg.get("words", []):
            if isinstance(word, dict) and "probability" in word:
                p = word["probability"]
                if p > 0:
                    log_probs.append(math.log(p))
    if not log_probs:
        return None
    mumble_thr = -1.0
    above = sum(1 for lp in log_probs if lp >= mumble_thr)
    return round(above / len(log_probs), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True, help="Recorded_Segments directory")
    parser.add_argument("--output-dir", required=True, help="Converted_Segments directory")
    args = parser.parse_args()

    wav_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".wav"))

    print(f"[wer_worker] Loading Whisper ...", file=sys.stderr)
    engine, model = load_engine()
    print(f"[wer_worker] Engine: {engine}. Processing {len(wav_files)} files ...", file=sys.stderr)

    results = []
    for wav_file in wav_files:
        sample   = os.path.splitext(wav_file)[0]
        in_path  = os.path.join(args.input_dir,  wav_file)
        out_path = os.path.join(args.output_dir, wav_file)

        if not os.path.exists(out_path):
            results.append({"sample": sample, "wer": None, "wer_error": "MISSING_OUTPUT"})
            continue

        in_res  = transcribe(in_path,  engine, model)
        out_res = transcribe(out_path, engine, model)

        in_text  = normalise_hi(in_res.get("text", ""))
        out_text = normalise_hi(out_res.get("text", ""))

        wer_value = None
        wer_error = None
        if in_text and out_text:
            try:
                wer_value = round(jiwer.wer(in_text, out_text), 4)
            except Exception as e:
                wer_error = str(e)
        elif not in_text:
            wer_error = "INPUT_EMPTY_TRANSCRIPT"
        elif not out_text:
            wer_error = "OUTPUT_EMPTY_TRANSCRIPT"

        intell = word_intelligibility(out_res.get("segments", []))

        results.append({
            "sample"         : sample,
            "in_text"        : in_text,
            "out_text"       : out_text,
            "wer"            : wer_value,
            "wer_error"      : wer_error,
            "intelligibility": intell,
            "engine"         : engine,
            "in_error"       : in_res.get("error"),
            "out_error"      : out_res.get("error"),
        })
        status = f"WER={wer_value:.3f}" if wer_value is not None else f"ERR={wer_error}"
        print(f"  {sample}: {status}", file=sys.stderr)

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
