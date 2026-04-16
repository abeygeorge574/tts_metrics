"""
Speaker Similarity — Multi-Episode Runner

Runs B5 + D across all episodes found in EPISODES_DIR.

Episode folder structure expected:
  EPISODES_DIR/
    EP21/
      Karthikeya/
        Converted_Segments/*.wav
        Train_Data/Karthikeya.wav
      Rambo/
        ...
    EP22/
      ...

Usage (local Mac):
  /Users/abey/miniconda3/bin/python3 -u spk_all_episodes.py

Usage (GCP):
  TTS_ROOT=/home/user/tts_metrics EPISODES_DIR=/path/to/episodes OUT_DIR=/home/user/results python3 -u spk_all_episodes.py

Env vars:
  TTS_ROOT     — path to tts_metrics repo root (default: auto-detect from script location)
  EPISODES_DIR — where episodes live (default: /Users/abey/Downloads)
  OUT_DIR      — where to save CSVs (default: /tmp/spk_results)
  EP_START     — first episode number (default: 21)
  EP_END       — last episode number  (default: 35)
"""

import os, sys, math, tempfile, json, re
import numpy as np
import soundfile as sf
import torch
import pandas as pd
from datetime import datetime

# ── Resolve repo root (works on Mac and GCP) ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_ROOT = os.environ.get("TTS_ROOT", None)
if TTS_ROOT is None:
    # Try to auto-detect: script may live in tts_metrics or /tmp/claude
    _candidate = "/Users/abey/Documents/tts_metrics"
    TTS_ROOT = _candidate if os.path.isdir(_candidate) else _SCRIPT_DIR

sys.path.insert(0, os.path.join(TTS_ROOT, "sts"))
sys.path.insert(0, os.path.join(TTS_ROOT, "sts", "gates"))
sys.path.insert(0, TTS_ROOT)

from _tts_imports import get_speaker_sim_functions
load_model, get_embedding, cosine_similarity = get_speaker_sim_functions()

# ── Config ────────────────────────────────────────────────────────────────────
EPISODES_DIR = os.environ.get("EPISODES_DIR", "/Users/abey/Downloads")
OUT_DIR      = os.environ.get("OUT_DIR", "/tmp/spk_results")
TRIM_S       = 1.0
SILENCE_DB   = -40
WINDOW_S     = 0.1
MIN_VOICED   = 5.0      # B5
SIM_THR      = 0.50     # placeholder — calibrated after seeing all episodes
EP_START     = int(os.environ.get("EP_START", 21))
EP_END       = int(os.environ.get("EP_END",   35))

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def emb_from_array(data, sr, clf):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, data, sr)
    try:
        return get_embedding(tmp, clf)
    finally:
        os.unlink(tmp)


def chunk_train_emb(train_file, clf, chunk_s=10.0):
    data, sr = sf.read(train_file, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    chunk_frames = int(chunk_s * sr)
    embeddings = []
    for i in range(max(1, math.ceil(len(data) / chunk_frames))):
        s, e = i * chunk_frames, min((i + 1) * chunk_frames, len(data))
        if (e - s) / sr < 1.0:
            continue
        try:
            emb = emb_from_array(data[s:e], sr, clf)
            embeddings.append(emb / torch.norm(emb))
        except Exception:
            pass
    if not embeddings:
        return None
    m = torch.stack(embeddings).mean(dim=0)
    return m / torch.norm(m)


def measure_voiced(data, sr):
    wf = int(WINDOW_S * sr)
    count = 0
    for i in range(0, len(data), wf):
        chunk = data[i:i + wf]
        if len(chunk) < wf // 2:
            continue
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        if rms > 1e-10 and 20 * np.log10(rms) > SILENCE_DB:
            count += 1
    return count * WINDOW_S


def flush_group(buf_audio, buf_segs, buf_metric, clf, train_emb, results, ep):
    if not buf_segs:
        return
    combined = np.concatenate(buf_audio)
    sr = buf_segs[0]["sr"]
    try:
        emb = emb_from_array(combined, sr, clf)
        sim = float(cosine_similarity(emb, train_emb))
    except Exception:
        sim = None
    for seg in buf_segs:
        results.append({**seg, "Episode": ep,
                        "scored_as": "group",
                        "group_voiced_s": round(buf_metric, 2),
                        "sim": round(sim, 4) if sim is not None else None,
                        "pass": (sim is not None and sim >= SIM_THR)})


def run_episode(ep_name, ep_path, clf):
    """Run B5 + D for one episode. Returns (seg_rows, char_d_rows)."""
    seg_rows  = []
    char_d_scores = []

    chars = sorted(d for d in os.listdir(ep_path)
                   if os.path.isdir(os.path.join(ep_path, d)))

    for char in chars:
        out_dir   = os.path.join(ep_path, char, "Converted_Segments")
        train_dir = os.path.join(ep_path, char, "Train_Data")
        # Train file may be named after base character (Karthikeya_1 → Karthikeya.wav)
        train_wavs = sorted(f for f in os.listdir(train_dir)
                            if f.endswith(".wav")) if os.path.isdir(train_dir) else []
        train_f = os.path.join(train_dir, train_wavs[0]) if train_wavs else None
        if not os.path.isdir(out_dir) or train_f is None:
            continue

        train_dur = sf.info(train_f).duration
        train_emb = chunk_train_emb(train_f, clf)
        if train_emb is None:
            print(f"    [{char}] WARNING: no train embedding")
            continue

        wav_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".wav"))

        # ── Pre-compute per segment ───────────────────────────────────────────
        segs = []
        for wav_file in wav_files:
            path = os.path.join(out_dir, wav_file)
            data, sr = sf.read(path, always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            dur_s = len(data) / sr
            tf = int(TRIM_S * sr)
            trimmed = data[tf:-tf] if len(data) > 2 * tf else np.array([])
            dur_trim = len(trimmed) / sr if len(trimmed) > 0 else 0.0
            voiced   = measure_voiced(trimmed, sr) if len(trimmed) > 0 else 0.0
            segs.append({
                "Character": char, "Sample": os.path.splitext(wav_file)[0],
                "dur_s": round(dur_s, 2), "dur_trim_s": round(dur_trim, 2),
                "voiced_s": round(voiced, 2), "trimmed": trimmed, "sr": sr,
            })

        # ── B5 merge logic ────────────────────────────────────────────────────
        char_rows = []
        buf_audio, buf_segs, buf_metric = [], [], 0.0
        pool_embs = []   # for D

        for seg in segs:
            is_silent = (seg["voiced_s"] < 0.2 or len(seg["trimmed"]) < 10)

            if is_silent:
                if buf_segs:
                    flush_group(buf_audio, buf_segs, buf_metric, clf,
                                train_emb, char_rows, ep_name)
                    buf_audio, buf_segs, buf_metric = [], [], 0.0
                char_rows.append({
                    "Episode": ep_name, "Character": char, "Sample": seg["Sample"],
                    "dur_s": seg["dur_s"], "dur_trim_s": seg["dur_trim_s"],
                    "voiced_s": seg["voiced_s"], "scored_as": "skip_silent",
                    "group_voiced_s": None, "sim": None, "pass": None,
                })
                continue

            if seg["voiced_s"] >= MIN_VOICED:
                if buf_segs:
                    flush_group(buf_audio, buf_segs, buf_metric, clf,
                                train_emb, char_rows, ep_name)
                    buf_audio, buf_segs, buf_metric = [], [], 0.0
                try:
                    emb = emb_from_array(seg["trimmed"], seg["sr"], clf)
                    sim = float(cosine_similarity(emb, train_emb))
                    pool_embs.append(emb / torch.norm(emb))
                except Exception:
                    emb = None
                    sim = None
                char_rows.append({
                    "Episode": ep_name, "Character": char, "Sample": seg["Sample"],
                    "dur_s": seg["dur_s"], "dur_trim_s": seg["dur_trim_s"],
                    "voiced_s": seg["voiced_s"], "scored_as": "individual",
                    "group_voiced_s": round(seg["voiced_s"], 2),
                    "sim": round(sim, 4) if sim is not None else None,
                    "pass": (sim is not None and sim >= SIM_THR),
                })
            else:
                buf_audio.append(seg["trimmed"])
                buf_segs.append({**seg, "sr": seg["sr"]})
                buf_metric += seg["voiced_s"]
                if buf_metric >= MIN_VOICED:
                    flush_group(buf_audio, buf_segs, buf_metric, clf,
                                train_emb, char_rows, ep_name)
                    buf_audio, buf_segs, buf_metric = [], [], 0.0

        if buf_segs:
            flush_group(buf_audio, buf_segs, buf_metric, clf,
                        train_emb, char_rows, ep_name)

        # ── D: character-level pooled ─────────────────────────────────────────
        # Also pool group embeddings (embed all group audios for D)
        for seg in segs:
            if 0.2 <= seg["voiced_s"] < MIN_VOICED and len(seg["trimmed"]) > 0:
                try:
                    emb = emb_from_array(seg["trimmed"], seg["sr"], clf)
                    pool_embs.append(emb / torch.norm(emb))
                except Exception:
                    pass

        d_sim = None
        if pool_embs:
            pooled = torch.stack(pool_embs).mean(dim=0)
            pooled = pooled / torch.norm(pooled)
            d_sim  = float(cosine_similarity(pooled, train_emb))

        # Per-char stats
        cdf    = pd.DataFrame(char_rows)
        scored = cdf[cdf["sim"].notna()]
        passes = scored["pass"].sum() if len(scored) > 0 else 0
        skip   = (cdf["scored_as"] == "skip_silent").sum()
        rate   = passes / len(scored) * 100 if len(scored) > 0 else 0

        char_d_scores.append({
            "Episode": ep_name, "Character": char,
            "Train_dur_s": round(train_dur, 0),
            "Total_segs": len(segs),
            "Scored_segs": len(scored), "Skipped_segs": skip,
            "B5_pass": passes, "B5_rate": round(rate, 1),
            "D_sim": round(d_sim, 4) if d_sim is not None else None,
            "D_pass": (d_sim is not None and d_sim >= SIM_THR),
        })

        d_sim_str = f"{d_sim:.4f}" if d_sim is not None else "N/A"
        print(f"    [{char}]  B5={passes}/{len(scored)} ({rate:.0f}%)  "
              f"D_sim={d_sim_str}  train={train_dur:.0f}s  segs={len(segs)}  skip={skip}")

        seg_rows.extend(char_rows)

    return seg_rows, char_d_scores


# ── Discover episodes EP21–EP35 only (exact digits, excludes EP21-2 etc.) ─────
def is_valid_episode(name):
    m = re.match(r'^EP(\d+)$', name)   # digits only after EP, no dashes or suffixes
    if not m:
        return False
    num = int(m.group(1))
    return EP_START <= num <= EP_END

episodes = sorted(
    [d for d in os.listdir(EPISODES_DIR)
     if is_valid_episode(d) and os.path.isdir(os.path.join(EPISODES_DIR, d))],
    key=lambda d: int(re.search(r'\d+', d).group())
)
print(f"Found {len(episodes)} episodes (EP{EP_START}–EP{EP_END}): {episodes}")

# ── Load model once ───────────────────────────────────────────────────────────
print("\nLoading ECAPA model...")
state = load_model()
clf   = state["classifier"]
print(f"Model loaded on {state['device']}\n")

# ── Run all episodes ──────────────────────────────────────────────────────────
all_seg_rows  = []
all_char_rows = []
ep_summaries  = []

for ep_name in episodes:
    ep_path = os.path.join(EPISODES_DIR, ep_name)
    print(f"\n{'='*60}")
    print(f"  EPISODE: {ep_name}  ({ep_path})")
    print(f"{'='*60}")

    seg_rows, char_rows = run_episode(ep_name, ep_path, clf)

    if not seg_rows:
        print(f"  [!] No results for {ep_name} — check folder structure")
        continue

    # Save per-episode CSV
    ep_csv = os.path.join(OUT_DIR, f"spk_{ep_name}.csv")
    pd.DataFrame(seg_rows).to_csv(ep_csv, index=False)

    # Episode summary
    ep_df     = pd.DataFrame(seg_rows)
    ep_scored = ep_df[ep_df["sim"].notna()]
    ep_pass   = ep_scored["pass"].sum() if len(ep_scored) > 0 else 0
    ep_skip   = (ep_df["scored_as"] == "skip_silent").sum()
    ep_rate   = ep_pass / len(ep_scored) * 100 if len(ep_scored) > 0 else 0

    d_sims = [r["D_sim"] for r in char_rows if r["D_sim"] is not None]
    d_pass = sum(1 for r in char_rows if r.get("D_pass"))
    d_total= sum(1 for r in char_rows if r["D_sim"] is not None)

    ep_summaries.append({
        "Episode": ep_name,
        "Total_segs": len(ep_df), "Scored_segs": len(ep_scored), "Skipped_segs": int(ep_skip),
        "B5_pass": int(ep_pass), "B5_rate": round(ep_rate, 1),
        "B5_avg_sim": round(ep_scored["sim"].mean(), 4) if len(ep_scored) > 0 else None,
        "B5_median_sim": round(ep_scored["sim"].median(), 4) if len(ep_scored) > 0 else None,
        "D_chars_scored": d_total, "D_chars_pass": d_pass,
        "D_avg_sim": round(np.mean(d_sims), 4) if d_sims else None,
        "D_median_sim": round(np.median(d_sims), 4) if d_sims else None,
    })

    all_seg_rows.extend(seg_rows)
    all_char_rows.extend(char_rows)

    print(f"\n  ── {ep_name} SUMMARY ──")
    print(f"  B5: {ep_pass}/{len(ep_scored)} PASS = {ep_rate:.1f}%  "
          f"avg_sim={ep_summaries[-1]['B5_avg_sim']}  skip={ep_skip}")
    print(f"  D:  {d_pass}/{d_total} chars PASS  avg_sim={ep_summaries[-1]['D_avg_sim']}")

# ── Save all outputs ──────────────────────────────────────────────────────────
all_seg_df  = pd.DataFrame(all_seg_rows)
all_char_df = pd.DataFrame(all_char_rows)
sum_df      = pd.DataFrame(ep_summaries)

all_seg_df.to_csv(os.path.join(OUT_DIR, "spk_all_segments.csv"), index=False)
all_char_df.to_csv(os.path.join(OUT_DIR, "spk_all_characters.csv"), index=False)
sum_df.to_csv(os.path.join(OUT_DIR, "spk_summary.csv"), index=False)

# ── Final cross-episode summary ───────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"CROSS-EPISODE SUMMARY  ({len(episodes)} episodes)")
print(f"{'='*60}")
print(sum_df[["Episode","Total_segs","B5_rate","B5_avg_sim","D_chars_pass","D_avg_sim"]].to_string(index=False))

print(f"\nOverall B5:  {all_seg_df['pass'].sum()}/{all_seg_df['sim'].notna().sum()} "
      f"= {all_seg_df['pass'].sum()/all_seg_df['sim'].notna().sum()*100:.1f}%")
d_all = all_char_df["D_sim"].dropna()
print(f"Overall D:   mean={d_all.mean():.4f}  median={d_all.median():.4f}")
print(f"\nSaved to: {OUT_DIR}/")
