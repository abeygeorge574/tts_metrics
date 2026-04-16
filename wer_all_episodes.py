"""
WER — Multi-Episode Runner (Hindi → Hindi)

Strategy:
  Recorded_Segments (Hindi original) → Whisper transcribe (hi) → reference text
  Converted_Segments (Hindi STS out) → Whisper transcribe (hi) → hypothesis text
  Compare → WER

Measures: did the STS output say the same words as the original?

Usage (Mac, utmos env):
  EP_START=21 EP_END=21 ~/miniconda3/envs/utmos/bin/python3 -u wer_all_episodes.py

Usage (GCP, CUDA):
  TTS_ROOT=/home/jupyter/tts_metrics EPISODES_DIR=/path/to/episodes python3 -u wer_all_episodes.py

Env vars:
  TTS_ROOT     - path to tts_metrics repo root
  EPISODES_DIR - where episodes live  (default: /Users/abey/Downloads)
  OUT_DIR      - where to save CSVs   (default: /tmp/wer_results)
  EP_START     - first episode number (default: 21)
  EP_END       - last episode number  (default: 35)
"""

import os, sys, re, math
import numpy as np
import pandas as pd
import jiwer
from jiwer import Compose, ToLowerCase, RemovePunctuation, Strip, SubstituteRegexes

# ── Resolve repo root ─────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_ROOT    = os.environ.get("TTS_ROOT", _SCRIPT_DIR)
sys.path.insert(0, TTS_ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
EPISODES_DIR   = os.environ.get("EPISODES_DIR", "/Users/abey/Downloads")
OUT_DIR        = os.environ.get("OUT_DIR", "/tmp/wer_results")
EP_START       = int(os.environ.get("EP_START", 21))
EP_END         = int(os.environ.get("EP_END",   35))
INTEL_LOG_THR    = -1.0    # log-prob threshold per word
INTEL_PASS_THR   = 0.85    # fraction of words above threshold to pass
WER_PASS_THR     = 0.10    # WER ≤ 10% = PASS

os.makedirs(OUT_DIR, exist_ok=True)

# ── Text normalisation ────────────────────────────────────────────────────────
_TRANSFORM = Compose([
    ToLowerCase(),
    SubstituteRegexes({r"[-\u2013\u2014]": " "}),
    RemovePunctuation(),
    Strip(),
])

def normalize(text):
    return _TRANSFORM(text)


# ── Load Whisper ──────────────────────────────────────────────────────────────
def load_whisper():
    import torch
    if torch.cuda.is_available():
        import whisper as openai_whisper
        engine     = "cuda"
        model_tier = "medium"
        print(f"CUDA — loading Whisper '{model_tier}'...")
        model = openai_whisper.load_model(model_tier, device="cuda")
    else:
        try:
            import mlx_whisper
            engine     = "mlx"
            model_tier = "mlx-community/whisper-medium-mlx"
            model      = None
            print("Apple Silicon — MLX Whisper medium.")
        except ImportError:
            import whisper as openai_whisper
            engine     = "cpu"
            model_tier = "base"
            print("CPU fallback — Whisper 'base'.")
            model = openai_whisper.load_model(model_tier, device="cpu")
    return engine, model_tier, model


def transcribe(audio_path, engine, model_tier, model, task="transcribe", language="en"):
    """task: 'transcribe' or 'translate' (translate always outputs English)."""
    if engine == "cuda":
        return model.transcribe(audio_path, task=task, language=language,
                                word_timestamps=True, fp16=True)
    elif engine == "mlx":
        import mlx_whisper
        return mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_tier,
                                      task=task, language=language,
                                      word_timestamps=True)
    else:
        return model.transcribe(audio_path, task=task, language=language,
                                word_timestamps=True, fp16=False)


def intel_score(result):
    """Mean log-prob and intelligibility rate from Whisper word timestamps."""
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            lp = w.get("probability", None)
            if lp is not None and lp > 0:
                words.append(math.log(lp))
    if not words:
        seg_lps = [s.get("avg_logprob", -9.0) for s in result.get("segments", [])]
        return (float(np.mean(seg_lps)) if seg_lps else -9.0), None
    mean_lp    = float(np.mean(words))
    intel_rate = sum(1 for lp in words if lp >= INTEL_LOG_THR) / len(words)
    return mean_lp, round(intel_rate, 4)



def compute_wer(ref_text, hyp_text):
    ref_n = normalize(ref_text)
    hyp_n = normalize(hyp_text)
    if not ref_n or not hyp_n:
        return None, None, None, None, None
    try:
        out = jiwer.process_words(ref_n, hyp_n)
        ref_words  = len(ref_n.split())
        hits       = ref_words - out.substitutions - out.deletions
        hit_rate   = round(hits / ref_words, 4) if ref_words > 0 else None
        return round(out.wer, 4), out.substitutions, out.deletions, out.insertions, hit_rate
    except Exception:
        return None, None, None, None, None


# ── Episode discovery ─────────────────────────────────────────────────────────
def is_valid_episode(name):
    m = re.match(r'^EP(\d+)$', name)
    if not m: return False
    return EP_START <= int(m.group(1)) <= EP_END

episodes = sorted(
    [d for d in os.listdir(EPISODES_DIR)
     if is_valid_episode(d) and os.path.isdir(os.path.join(EPISODES_DIR, d))],
    key=lambda d: int(re.search(r'\d+', d).group())
)
print(f"Found {len(episodes)} episodes (EP{EP_START}–EP{EP_END}): {episodes}\n")

# ── Load model ────────────────────────────────────────────────────────────────
engine, model_tier, model = load_whisper()
print(f"Engine: {engine}  Model: {model_tier}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
ep_summaries = []

for ep_name in episodes:
    ep_path = os.path.join(EPISODES_DIR, ep_name)
    print(f"\n{'='*60}")
    print(f"  EPISODE: {ep_name}")
    print(f"{'='*60}")

    chars = sorted(d for d in os.listdir(ep_path)
                   if os.path.isdir(os.path.join(ep_path, d)))
    ep_rows = []

    for char in chars:
        conv_dir = os.path.join(ep_path, char, "Converted_Segments")
        rec_dir  = os.path.join(ep_path, char, "Recorded_Segments")
        if not os.path.isdir(conv_dir):
            continue

        wav_files = sorted(f for f in os.listdir(conv_dir) if f.endswith(".wav"))
        char_rows = []

        for wav_file in wav_files:
            conv_path = os.path.join(conv_dir, wav_file)
            rec_path  = os.path.join(rec_dir,  wav_file) if os.path.isdir(rec_dir) else None

            import soundfile as sf
            try:
                dur = sf.info(conv_path).duration
            except Exception:
                dur = None

            # ── Transcribe original Hindi (reference) ─────────────────────────
            ref_text = None
            if rec_path and os.path.exists(rec_path):
                try:
                    ref_result = transcribe(rec_path, engine, model_tier, model,
                                            task="transcribe", language="hi")
                    ref_text   = ref_result.get("text", "").strip()
                except Exception:
                    ref_text = None

            # ── Transcribe STS Hindi output (hypothesis) ──────────────────────
            try:
                hyp_result = transcribe(conv_path, engine, model_tier, model,
                                        task="transcribe", language="hi")
                hyp_text   = hyp_result.get("text", "").strip()
                mean_lp, intel_rate = intel_score(hyp_result)
                intel_pass = (intel_rate >= INTEL_PASS_THR) if intel_rate is not None else None
            except Exception:
                hyp_text   = None
                mean_lp    = None
                intel_rate = None
                intel_pass = None

            # ── WER ───────────────────────────────────────────────────────────
            wer = subs = dels = ins = hit_rate = wer_pass = None
            if ref_text and hyp_text:
                wer, subs, dels, ins, hit_rate = compute_wer(ref_text, hyp_text)
                if wer is not None:
                    wer_pass = (wer <= WER_PASS_THR)

            final_pass = wer_pass if wer_pass is not None else intel_pass

            char_rows.append({
                "Episode"      : ep_name,
                "Character"    : char,
                "Sample"       : os.path.splitext(wav_file)[0],
                "duration_s"   : round(dur, 2) if dur else None,
                "ref_text"     : ref_text,
                "hyp_text"     : hyp_text,
                "WER"          : wer,
                "substitutions": subs,
                "deletions"    : dels,
                "insertions"   : ins,
                "hit_rate"     : hit_rate,
                "wer_pass"     : wer_pass,
                "mean_log_prob": round(mean_lp, 4) if mean_lp is not None else None,
                "intel_rate"   : intel_rate,
                "intel_pass"   : intel_pass,
                "final_pass"   : final_pass,
            })

        # Per-character summary
        cdf     = pd.DataFrame(char_rows)
        scored  = cdf[cdf["final_pass"].notna()]
        passing = scored["final_pass"].sum() if len(scored) > 0 else 0
        rate    = passing / len(scored) * 100 if len(scored) > 0 else 0
        wer_scored = cdf[cdf["WER"].notna()]
        avg_wer = wer_scored["WER"].mean() if len(wer_scored) > 0 else None
        avg_wer_str = f"{avg_wer:.3f}" if avg_wer is not None else "no_ref"
        print(f"    [{char}]  pass={int(passing)}/{len(scored)} ({rate:.0f}%)  "
              f"avg_wer={avg_wer_str}  segs={len(cdf)}")

        ep_rows.extend(char_rows)

    # Save per-episode CSV
    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(os.path.join(OUT_DIR, f"wer_{ep_name}.csv"), index=False)

    # Episode summary
    scored_ep = ep_df[ep_df["final_pass"].notna()]
    pass_ep   = int(scored_ep["final_pass"].sum()) if len(scored_ep) > 0 else 0
    rate_ep   = pass_ep / len(scored_ep) * 100 if len(scored_ep) > 0 else 0
    wer_ep    = ep_df[ep_df["WER"].notna()]
    avg_wer_ep = wer_ep["WER"].mean() if len(wer_ep) > 0 else None
    avg_wer_ep_str = f"{avg_wer_ep:.3f}" if avg_wer_ep is not None else "N/A"

    ep_summaries.append({
        "Episode"     : ep_name,
        "Total_segs"  : len(ep_df),
        "Scored_segs" : len(scored_ep),
        "Pass"        : pass_ep,
        "Pass_rate"   : round(rate_ep, 1),
        "Avg_WER"     : round(avg_wer_ep, 4) if avg_wer_ep is not None else None,
        "WER_scored"  : len(wer_ep),
    })

    print(f"\n  ── {ep_name} SUMMARY ──")
    print(f"  pass={pass_ep}/{len(scored_ep)} ({rate_ep:.1f}%)  avg_wer={avg_wer_ep_str}")

# ── Save combined outputs ─────────────────────────────────────────────────────
all_df = pd.concat(
    [pd.read_csv(os.path.join(OUT_DIR, f"wer_{ep}.csv")) for ep in episodes
     if os.path.exists(os.path.join(OUT_DIR, f"wer_{ep}.csv"))],
    ignore_index=True
)
sum_df = pd.DataFrame(ep_summaries)

all_df.to_csv(os.path.join(OUT_DIR, "wer_all_segments.csv"), index=False)
sum_df.to_csv(os.path.join(OUT_DIR, "wer_summary.csv"),      index=False)

# ── Final cross-episode summary ───────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"CROSS-EPISODE SUMMARY  ({len(episodes)} episodes)")
print(f"{'='*60}")
print(sum_df[["Episode", "Total_segs", "Pass_rate", "Avg_WER", "WER_scored"]].to_string(index=False))

total_scored = sum_df["Scored_segs"].sum()
total_pass   = sum_df["Pass"].sum()
if total_scored > 0:
    print(f"\nOverall: {total_pass}/{total_scored} = {total_pass/total_scored*100:.1f}%")
print(f"\nSaved to: {OUT_DIR}/")
