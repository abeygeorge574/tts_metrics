"""
Gate: Speech Emotion Recognition (SER)
Env : base (python 3.13)
Uses emotion2vec_plus_large to classify emotion in reference vs TTS audio.
Pass = emotions match. Segments where reference confidence is low are flagged
as degraded but still scored.

Files longer than _SER_CHUNK_S seconds are automatically split into chunks.
The final emotion label is chosen by majority vote across chunk labels;
the reported confidence is the mean of each chunk's top-label confidence.
"""

import os
import sys
import argparse
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_SER_CHUNK_S = 15   # emotion needs full utterance arc; training distribution ~5-15s


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    from funasr import AutoModel

    ser_model = AutoModel(
        model="emotion2vec/emotion2vec_plus_large",
        model_revision="v2.0.4",
        hub="hf",
        disable_update=True
    )

    print("emotion2vec loaded.")
    return {"ser_model": ser_model}


# ── Emotion extraction (single file, raw) ─────────────────────────────────────
def _get_emotion_raw(audio_path, ser_model):
    """Run emotion2vec on one file. Returns (label, confidence) or (None, None)."""
    try:
        res = ser_model.generate(
            audio_path,
            granularity="utterance",
            extract_embedding=False,
            disable_update=True
        )
        labels = res[0]["labels"]
        scores = res[0]["scores"]

        max_idx    = scores.index(max(scores))
        raw_label  = labels[max_idx]
        confidence = round(scores[max_idx], 4)

        # strip Chinese prefix — "生气/angry" → "angry"
        clean_label = raw_label.split("/")[-1]

        return clean_label, confidence

    except Exception as e:
        print(f"  emotion2vec error on {audio_path}: {e}")
        return None, None


# ── Emotion extraction with automatic chunking ─────────────────────────────────
def get_emotion(audio_path, ser_model):
    """
    Score one audio file with emotion2vec.

    If the file is ≤ _SER_CHUNK_S seconds it is scored directly.
    Longer files are split into _SER_CHUNK_S-second temp WAV clips and
    scored individually. The final label is the majority vote across chunks;
    the confidence is the mean of the winning label's per-chunk confidence.
    Temp files are deleted immediately after scoring.
    """
    import soundfile as sf

    data, sr = sf.read(audio_path, always_2d=False)
    duration = len(data) / sr

    if duration <= _SER_CHUNK_S:
        return _get_emotion_raw(audio_path, ser_model)

    import math
    total_frames     = len(data)
    max_chunk_frames = int(_SER_CHUNK_S * sr)
    n_chunks         = math.ceil(total_frames / max_chunk_frames)
    chunk_size       = total_frames / n_chunks   # equal-length, float → round at boundaries
    print(f"  [SER] {duration:.1f}s → {n_chunks} equal chunks (~{duration/n_chunks:.1f}s each)")

    chunk_labels = []
    chunk_confs  = []

    tmp_dir = tempfile.mkdtemp(prefix="ser_chunk_")
    try:
        for i in range(n_chunks):
            start = round(i       * chunk_size)
            end   = round((i + 1) * chunk_size)
            chunk = data[start:end]

            # Skip chunks shorter than 0.5 s
            if (end - start) / sr < 0.5:
                print(f"  [SER] chunk {i+1}/{n_chunks}: too short, skipping")
                continue

            tmp_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
            sf.write(tmp_path, chunk, sr)

            try:
                label, conf = _get_emotion_raw(tmp_path, ser_model)
                if label is not None:
                    chunk_labels.append(label)
                    chunk_confs.append(conf)
                    print(f"  [SER] chunk {i+1}/{n_chunks}: {label} ({conf})")
                else:
                    print(f"  [SER] chunk {i+1}/{n_chunks}: no result")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if not chunk_labels:
        return None, None

    # Majority vote on label
    from collections import Counter
    vote_counts  = Counter(chunk_labels)
    winner_label = vote_counts.most_common(1)[0][0]

    # Mean confidence of the winning label's chunks
    winner_confs = [c for lbl, c in zip(chunk_labels, chunk_confs) if lbl == winner_label]
    mean_conf    = round(sum(winner_confs) / len(winner_confs), 4)

    print(f"  [SER] vote result: {winner_label} ({vote_counts}) → {mean_conf} mean conf")
    return winner_label, mean_conf


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    ser_model = model_state["ser_model"]

    MODELS_DIR    = config.MODELS_DIR
    REFERENCE_DIR = config.REFERENCE_DIR

    CONFIDENCE_THRESHOLD = config.SER_CONFIDENCE_THRESHOLD

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")
    if not os.path.exists(REFERENCE_DIR):
        raise FileNotFoundError(
            f"Reference folder not found: {REFERENCE_DIR}\n"
            f"SER requires Hindi reference audio to compare against."
        )

    ref_files = sorted([f for f in os.listdir(REFERENCE_DIR) if f.endswith(".wav")])
    print(f"Reference folder found: {len(ref_files)} files")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models found: {model_folders}")

    model_samples = {}
    for model in model_folders:
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])
        model_samples[model] = wav_files
        print(f"   {model}: {len(wav_files)} samples")

    reference_filenames = set(model_samples[model_folders[0]])
    for model in model_folders[1:]:
        current_filenames = set(model_samples[model])
        if current_filenames != reference_filenames:
            missing = reference_filenames - current_filenames
            extra   = current_filenames - reference_filenames
            raise ValueError(
                f"Model '{model}' has mismatched filenames.\n"
                f"  Missing : {missing}\n"
                f"  Extra   : {extra}"
            )
    print("All models have identical filenames.")

    sample_names = model_samples[model_folders[0]]
    total        = len(model_folders) * len(sample_names)
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = {total} evaluations")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            print(f"\n  Sample : {sample_name}")

            if not os.path.exists(ref_path):
                print(f"  No reference file found — skipping")
                results.append({
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Ref Label"   : None,
                    "Ref Conf"    : None,
                    "TTS Label"   : None,
                    "TTS Conf"    : None,
                    "Match"       : None,
                    "Pass"        : "SKIP",
                    "Flag"        : "NO_REF",
                    "_is_degraded": False,
                })
                continue

            ref_label, ref_conf = get_emotion(ref_path, ser_model)
            print(f"  Ref    : {ref_label} ({ref_conf})")

            tts_label, tts_conf = get_emotion(tts_path, ser_model)
            print(f"  TTS    : {tts_label} ({tts_conf})")

            if ref_label is None or tts_label is None:
                flag        = "ERROR"
                match       = None
                passed      = "ERROR"
                is_degraded = False
            elif ref_conf < CONFIDENCE_THRESHOLD:
                flag        = "LOW_CONF_REF"
                match       = ref_label == tts_label
                passed      = "PASS" if match else "FAIL"
                is_degraded = True
            else:
                flag        = "—"
                match       = ref_label == tts_label
                passed      = "PASS" if match else "FAIL"
                is_degraded = False

            print(f"  Result : {passed} | Match: {match} | Flag: {flag}")

            results.append({
                "Model"       : model,
                "Sample"      : sample_name,
                "Ref Label"   : ref_label,
                "Ref Conf"    : ref_conf,
                "TTS Label"   : tts_label,
                "TTS Conf"    : tts_conf,
                "Match"       : match,
                "Pass"        : passed,
                "Flag"        : flag,
                "_is_degraded": is_degraded,
            })

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        model_df    = df[df["Model"] == model]
        clean_df    = model_df[
            ~model_df["_is_degraded"] &
            (model_df["Flag"] != "NO_REF") &
            (model_df["Flag"] != "ERROR")
        ]
        degraded_df = model_df[model_df["_is_degraded"]]
        total       = len(model_df)

        clean_total = len(clean_df)
        clean_pass  = (clean_df["Pass"] == "PASS").sum()

        deg_total   = len(degraded_df)
        deg_pass    = (degraded_df["Pass"] == "PASS").sum()

        fail_df              = clean_df[clean_df["Pass"] == "FAIL"]
        most_common_mismatch = (
            fail_df["TTS Label"].value_counts().index[0]
            if len(fail_df) > 0 else "—"
        )

        summary_rows.append({
            "Model"             : model,
            "Total Segments"    : total,
            "Clean Segments"    : clean_total,
            "Clean Pass Rate"   : f"{clean_pass}/{clean_total}"  if clean_total > 0 else "—",
            "Degraded Segments" : deg_total,
            "Degraded Pass Rate": f"{deg_pass}/{deg_total}"      if deg_total > 0 else "—",
            "Common Mismatch"   : most_common_mismatch,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_clean_pass_num"]    = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_degraded_pass_num"] = summary_df["Degraded Pass Rate"].apply(parse_rate)

    summary_df = summary_df.sort_values(
        by=["_clean_pass_num", "_degraded_pass_num"],
        ascending=[False, False]
    ).drop(columns=["_clean_pass_num", "_degraded_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Ref Label", "Ref Conf",
        "TTS Label", "TTS Conf", "Pass", "Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Degraded Pass Rate", "Common Mismatch"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate    → primary ranking — ref confidence >= 0.5 segments only")
    print("Degraded Pass Rate → segments where reference emotion was ambiguous")
    print("Common Mismatch    → what emotion TTS produces when it fails")
    print(f"\nMin confidence threshold: {config.SER_CONFIDENCE_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_df = df.drop(columns=["_is_degraded"], errors="ignore")
    seg_df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "ser"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
