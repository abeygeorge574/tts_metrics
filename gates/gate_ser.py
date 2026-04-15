"""
Gate: Speech Emotion Recognition (SER)
Env : base (python 3.13)

Uses MERaLiON-SER-v1 (Whisper-Medium + LoRA + ECAPA-TDNN) to classify emotion
in reference vs TTS audio.

7 emotion classes: neutral, happy, sad, angry, fearful, disgusted, surprised

Pass/fail logic (top-2 overlap):
  PASS      — ref top-1 matches TTS top-1
  NEAR_MISS — top-2 label sets intersect but top-1s differ
              (e.g. ref=angry/fearful, TTS=fearful/surprised → fearful matches)
  REVIEW    — ref top1 conf < 0.5 (degraded) but TTS top1 conf >= 0.5 (abs OK)
  FAIL      — no label in common between ref top-2 and TTS top-2
            — or: ref degraded AND TTS also low confidence

Arousal from MERaLiON dims is recorded as a diagnostic column (not pass/fail).
Valence is also recorded but NOT used in pass/fail due to cross-lingual bias
(English-trained model reads Hindi prosody as systematically lower-valence).

Model: MERaLiON/MERaLiON-SER-v1
Languages trained: en, zh, ms, ta, id, th, vi
Hindi reference: cross-lingual — categorical errors expected on reference audio.
Gate is still useful because relative comparison (ref pattern vs TTS pattern)
catches TTS that radically shifts away from the reference delivery.
"""

import os
import sys
import argparse

# Proxy clearing — before any HF / httpx imports
for _pv in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_pv, None)

import torch
import torch.nn.functional as F
import torchaudio
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    # Redirect HF module cache away from ~/.cache (may be write-protected in some envs)
    os.environ.setdefault("HF_HOME", "/tmp/claude/hf_cache")

    from transformers import AutoModel, AutoProcessor

    local_path = config.MERALION_LOCAL_PATH
    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"MERaLiON model not found at {local_path}\n"
            f"Download with:\n"
            f"  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('MERaLiON/MERaLiON-SER-v1', local_dir='{local_path}')"
        )

    print(f"Loading MERaLiON-SER-v1 from {local_path} ...")
    model = AutoModel.from_pretrained(
        local_path,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    processor = AutoProcessor.from_pretrained(
        local_path,
        trust_remote_code=True,
    )
    model.eval()
    print("MERaLiON-SER-v1 loaded.")
    return {"model": model, "processor": processor}


# ── Emotion extraction ─────────────────────────────────────────────────────────
def get_emotion(audio_path, model, processor):
    """
    Returns (top1_label, top1_conf, top2_label, top2_conf, valence, arousal, dominance).
    dims order: [valence, arousal, dominance] — all in [0, 1].
    Returns all None on error.
    """
    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs['logits']  # (1, 7)
        dims   = outputs['dims']    # (1, 3) — [valence, arousal, dominance]

        probs      = F.softmax(logits, dim=-1).squeeze()  # (7,)
        sorted_idx = probs.argsort(descending=True)

        top1_label = EMOTION_LABELS[sorted_idx[0].item()]
        top1_conf  = round(probs[sorted_idx[0]].item(), 4)
        top2_label = EMOTION_LABELS[sorted_idx[1].item()]
        top2_conf  = round(probs[sorted_idx[1]].item(), 4)

        d         = dims.squeeze().tolist()
        valence   = round(d[0], 4)
        arousal   = round(d[1], 4)
        dominance = round(d[2], 4)

        return top1_label, top1_conf, top2_label, top2_conf, valence, arousal, dominance

    except Exception as e:
        print(f"  MERaLiON error on {audio_path}: {e}")
        return None, None, None, None, None, None, None


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    model     = model_state["model"]
    processor = model_state["processor"]

    MODELS_DIR    = model_state.get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = model_state.get("ref_dir")    or config.REFERENCE_DIR

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")
    if not os.path.exists(REFERENCE_DIR):
        raise FileNotFoundError(
            f"Reference folder not found: {REFERENCE_DIR}\n"
            f"SER gate requires reference audio to compare against."
        )

    ref_files = sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".wav")])
    print(f"Reference folder: {len(ref_files)} files")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models: {model_folders}")

    model_samples = {}
    for m in model_folders:
        wav_files = sorted([f for f in os.listdir(os.path.join(MODELS_DIR, m)) if f.endswith(".wav")])
        model_samples[m] = wav_files
        print(f"   {m}: {len(wav_files)} samples")

    results = []

    for m in model_folders:
        print(f"\n{'='*50}\nModel: {m}\n{'='*50}")

        for wav_file in model_samples[m]:
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, m, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            print(f"\n  Sample: {sample_name}")

            if not os.path.exists(ref_path):
                print(f"  No reference — skipping")
                results.append({
                    "Model"                                          : m,
                    "Sample"                                         : sample_name,
                    "Ref Top1"                                       : None,
                    "Ref Top1 Conf (degraded if<0.50)"               : None,
                    "Ref Top2"                                       : None,
                    "Ref Top2 Conf"                                  : None,
                    "Ref Arousal"                                    : None,
                    "Ref Valence"                                    : None,
                    "TTS Top1"                                       : None,
                    "TTS Top1 Conf"                                  : None,
                    "TTS Top2"                                       : None,
                    "TTS Top2 Conf"                                  : None,
                    "TTS Arousal"                                    : None,
                    "TTS Valence"                                    : None,
                    "Arousal Delta (threshold≤0.15)"                 : None,
                    "Emotion Pass (PASS/NEAR_MISS/REVIEW/FAIL)"      : "SKIP",
                    "Arousal Pass (PASS/FAIL, threshold≤0.15)"       : "SKIP",
                    "Ref Flag (—=clean|LOW_CONF_REF=ref_conf<0.50)"  : "NO_REF",
                    "_is_degraded"                                   : True,
                })
                continue

            r_t1, r_t1c, r_t2, r_t2c, r_val, r_ar, r_dom = get_emotion(ref_path,  model, processor)
            t_t1, t_t1c, t_t2, t_t2c, t_val, t_ar, t_dom = get_emotion(tts_path,  model, processor)

            print(f"  Ref: {r_t1}({r_t1c}) / {r_t2}({r_t2c})  ar={r_ar}")
            print(f"  TTS: {t_t1}({t_t1c}) / {t_t2}({t_t2c})  ar={t_ar}")

            if r_t1 is None or t_t1 is None:
                emotion_pass = "ERROR"
                arousal_pass = "ERROR"
                ref_flag     = "ERROR"
                is_degraded  = False
            else:
                # Degraded when ref top1 confidence is below threshold
                is_degraded = r_t1c is not None and r_t1c < config.SER_CONFIDENCE_THRESHOLD
                ref_flag    = "LOW_CONF_REF" if is_degraded else "—"

                if is_degraded:
                    # Skip normal emotion pass/fail; use REVIEW or absolute FAIL
                    tts_conf_ok = t_t1c is not None and t_t1c >= config.SER_CONFIDENCE_THRESHOLD
                    if tts_conf_ok:
                        emotion_pass = "REVIEW"
                    else:
                        emotion_pass = "FAIL (Low_Conf_Abs)"
                else:
                    ref_set = {r_t1, r_t2}
                    tts_set = {t_t1, t_t2}

                    if r_t1 == t_t1:
                        emotion_pass = "PASS"
                    elif ref_set & tts_set:
                        emotion_pass = "NEAR_MISS"
                    else:
                        emotion_pass = "FAIL"

            ar_delta = round(abs(r_ar - t_ar), 4) if r_ar is not None and t_ar is not None else None
            if ar_delta is not None:
                arousal_pass = "PASS" if ar_delta <= config.AROUSAL_DELTA_THRESHOLD else "FAIL"
            else:
                arousal_pass = "ERROR"

            print(f"  → Emotion:{emotion_pass}  Arousal:{arousal_pass}  ar_Δ={ar_delta}  RefFlag:{ref_flag}")

            results.append({
                "Model"                                          : m,
                "Sample"                                         : sample_name,
                "Ref Top1"                                       : r_t1,
                "Ref Top1 Conf (degraded if<0.50)"               : r_t1c,
                "Ref Top2"                                       : r_t2,
                "Ref Top2 Conf"                                  : r_t2c,
                "Ref Arousal"                                    : r_ar,
                "Ref Valence"                                    : r_val,
                "TTS Top1"                                       : t_t1,
                "TTS Top1 Conf"                                  : t_t1c,
                "TTS Top2"                                       : t_t2,
                "TTS Top2 Conf"                                  : t_t2c,
                "TTS Arousal"                                    : t_ar,
                "TTS Valence"                                    : t_val,
                "Arousal Delta (threshold≤0.15)"                 : ar_delta,
                "Emotion Pass (PASS/NEAR_MISS/REVIEW/FAIL)"      : emotion_pass,
                "Arousal Pass (PASS/FAIL, threshold≤0.15)"       : arousal_pass,
                "Ref Flag (—=clean|LOW_CONF_REF=ref_conf<0.50)"  : ref_flag,
                "_is_degraded"                                   : is_degraded,
            })

    print("\n\nAll evaluations complete.")
    df = pd.DataFrame(results)

    summary_rows = []
    for m in model_folders:
        model_df    = df[df["Model"] == m]
        ep_col      = "Emotion Pass (PASS/NEAR_MISS/REVIEW/FAIL)"
        ap_col      = "Arousal Pass (PASS/FAIL, threshold≤0.15)"
        flag_col    = "Ref Flag (—=clean|LOW_CONF_REF=ref_conf<0.50)"

        # Scored = non-degraded rows with a real verdict (exclude SKIP/ERROR/REVIEW/FAIL(Low_Conf_Abs))
        scored_df   = model_df[model_df[ep_col].isin(["PASS", "NEAR_MISS", "FAIL"])]
        e_pass      = (scored_df[ep_col] == "PASS").sum()
        e_nm        = (scored_df[ep_col] == "NEAR_MISS").sum()
        e_fail      = (scored_df[ep_col] == "FAIL").sum()
        total       = len(scored_df)

        # REVIEW segments (degraded ref, TTS confidence OK)
        review_count = (model_df[ep_col] == "REVIEW").sum()

        # Degraded count
        degraded_count = model_df["_is_degraded"].sum() if "_is_degraded" in model_df.columns else 0

        ar_scored   = model_df[model_df[ap_col].isin(["PASS", "FAIL"])]
        ar_pass     = (ar_scored[ap_col] == "PASS").sum()
        ar_total    = len(ar_scored)

        fail_df = scored_df[scored_df[ep_col] == "FAIL"]
        common_mismatch = (
            fail_df["TTS Top1"].value_counts().index[0]
            if len(fail_df) > 0 else "—"
        )

        ar_delta_col = "Arousal Delta (threshold≤0.15)"
        summary_rows.append({
            "Model"                                           : m,
            "Total"                                           : len(model_df),
            "Emotion Pass Rate (PASS / scored)"               : f"{e_pass}/{total}",
            "Emotion Pass+NearMiss Rate (PASS+NEAR_MISS / scored)": f"{e_pass + e_nm}/{total}",
            "Arousal Pass Rate (PASS / scored)"               : f"{ar_pass}/{ar_total}",
            "Near Miss (top-2 label overlap)"             : int(e_nm),
            "Review (ref low conf, TTS conf OK)"              : int(review_count),
            "Degraded Segments"                               : int(degraded_count),
            "Median Arousal Δ"                                : round(scored_df[ar_delta_col].dropna().median(), 4) if total > 0 else None,
            "Common Mismatch"                                 : common_mismatch,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_sort"] = summary_df["Emotion Pass+NearMiss Rate (PASS+NEAR_MISS / scored)"].apply(
        lambda x: int(x.split("/")[0]) if "/" in str(x) else -1
    )
    summary_df = summary_df.sort_values("_sort", ascending=False).drop(columns=["_sort"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    cols = [
        "Model", "Sample",
        "Ref Top1", "Ref Top1 Conf (degraded if<0.50)", "Ref Top2",
        "TTS Top1", "TTS Top1 Conf", "TTS Top2",
        "Ref Arousal", "TTS Arousal", "Arousal Delta (threshold≤0.15)",
        "Emotion Pass (PASS/NEAR_MISS/REVIEW/FAIL)",
        "Arousal Pass (PASS/FAIL, threshold≤0.15)",
        "Ref Flag (—=clean|LOW_CONF_REF=ref_conf<0.50)",
    ]
    print(df[cols].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Emotion Pass  — PASS: top-1 match | NEAR_MISS: top-2 overlap | FAIL: no overlap")
    print("REVIEW        — ref confidence < 0.5 (degraded) but TTS confidence >= 0.5 (abs OK)")
    print("Arousal Pass  — PASS if |ref_arousal − tts_arousal| ≤ threshold")
    print(f"Arousal threshold: {config.AROUSAL_DELTA_THRESHOLD}")
    print(f"SER confidence threshold: {config.SER_CONFIDENCE_THRESHOLD}")
    print("Valence       — recorded only, not in any pass/fail (cross-lingual bias)")
    print()
    print("Model: MERaLiON-SER-v1 (Whisper-Medium + LoRA + ECAPA-TDNN)")
    print("Labels: neutral, happy, sad, angry, fearful, disgusted, surprised")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER gate — MERaLiON-SER-v1")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "ser"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Override config.REFERENCE_DIR")
    args = parser.parse_args()

    model_state = load_model()
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.ref_dir:
        model_state["ref_dir"] = os.path.abspath(args.ref_dir)

    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
