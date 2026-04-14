"""
Gate: Speech Emotion Recognition (SER)
Env : base (python 3.13)
Uses emotion2vec_plus_large to classify emotion in reference vs TTS audio.

Pass/Near_Miss/Fail logic:
  PASS      — TTS top-1 label matches reference top-1 label
  NEAR_MISS — Labels mismatch, but either:
                (a) TTS had ref label as runner-up within SER_NEAR_MISS_MARGIN of winner, OR
                (b) Reference distribution itself was uncertain (top-1 vs top-2 within margin)
  FAIL      — Labels mismatch with clear confidence gap

Segments where reference confidence is low are flagged as degraded but still scored.
Files longer than _SER_CHUNK_S seconds are automatically split into equal chunks.
Final label = majority vote across chunks; confidence = mean of winning chunks.
"""

import os
import sys
import argparse
import tempfile

for _pv in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_pv, None)

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_SER_CHUNK_S = 15   # emotion needs full utterance arc; training distribution ~5-15s


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import logging
    logging.getLogger("funasr").setLevel(logging.WARNING)

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
    """
    Run emotion2vec on one file.
    Returns (top1_label, top1_conf, top2_label, top2_conf) or (None, None, None, None).
    """
    try:
        res = ser_model.generate(
            audio_path,
            granularity="utterance",
            extract_embedding=False,
            disable_update=True
        )
        labels = res[0]["labels"]
        scores = res[0]["scores"]

        # Rank all classes by score descending
        ranked = sorted(zip(scores, labels), reverse=True)

        def clean(raw_label):
            return raw_label.split("/")[-1]

        top1_label = clean(ranked[0][1])
        top1_conf  = round(ranked[0][0], 4)
        top2_label = clean(ranked[1][1]) if len(ranked) > 1 else None
        top2_conf  = round(ranked[1][0], 4) if len(ranked) > 1 else None

        return top1_label, top1_conf, top2_label, top2_conf

    except Exception as e:
        print(f"  emotion2vec error on {audio_path}: {e}")
        return None, None, None, None


# ── Emotion extraction with automatic chunking ─────────────────────────────────
def get_emotion(audio_path, ser_model):
    """
    Score one audio file with emotion2vec.
    Returns (top1_label, top1_conf, top2_label, top2_conf).

    Short files (≤ _SER_CHUNK_S s) are scored directly.
    Longer files are split into equal chunks, scored individually, and aggregated:
      - top1: majority vote label, mean conf of that label's chunks
      - top2: second most voted label, mean conf of that label's chunks
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
    chunk_size       = total_frames / n_chunks
    print(f"  [SER] {duration:.1f}s → {n_chunks} equal chunks (~{duration/n_chunks:.1f}s each)")

    chunk_labels = []
    chunk_confs  = []

    tmp_dir = tempfile.mkdtemp(prefix="ser_chunk_")
    try:
        for i in range(n_chunks):
            start = round(i       * chunk_size)
            end   = round((i + 1) * chunk_size)
            chunk = data[start:end]

            if (end - start) / sr < 0.5:
                print(f"  [SER] chunk {i+1}/{n_chunks}: too short, skipping")
                continue

            tmp_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
            sf.write(tmp_path, chunk, sr)

            try:
                lbl, conf, _, _ = _get_emotion_raw(tmp_path, ser_model)
                if lbl is not None:
                    chunk_labels.append(lbl)
                    chunk_confs.append(conf)
                    print(f"  [SER] chunk {i+1}/{n_chunks}: {lbl} ({conf})")
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
        return None, None, None, None

    from collections import Counter
    vote_counts = Counter(chunk_labels)
    top_two = vote_counts.most_common(2)

    winner_label = top_two[0][0]
    winner_confs = [c for lbl, c in zip(chunk_labels, chunk_confs) if lbl == winner_label]
    top1_conf    = round(sum(winner_confs) / len(winner_confs), 4)

    if len(top_two) > 1:
        runner_label = top_two[1][0]
        runner_confs = [c for lbl, c in zip(chunk_labels, chunk_confs) if lbl == runner_label]
        top2_conf    = round(sum(runner_confs) / len(runner_confs), 4)
    else:
        runner_label = None
        top2_conf    = None

    print(f"  [SER] vote result: {winner_label} ({vote_counts}) → {top1_conf} mean conf")
    return winner_label, top1_conf, runner_label, top2_conf


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    ser_model = model_state["ser_model"]

    MODELS_DIR    = model_state.get("models_dir") or config.MODELS_DIR
    REFERENCE_DIR = model_state.get("ref_dir")    or config.REFERENCE_DIR

    CONFIDENCE_THRESHOLD = config.SER_CONFIDENCE_THRESHOLD
    NEAR_MISS_MARGIN     = config.SER_NEAR_MISS_MARGIN

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
            import soundfile as sf
            sample_name = os.path.splitext(wav_file)[0]
            tts_path    = os.path.join(MODELS_DIR, model, wav_file)
            ref_path    = os.path.join(REFERENCE_DIR, wav_file)

            duration = sf.info(tts_path).duration
            is_short = duration < config.MIN_SEGMENT_DURATION
            if is_short:
                print(f"\n  Sample : {sample_name} [SHORT: {duration:.2f}s]")
            else:
                print(f"\n  Sample : {sample_name}")

            if not os.path.exists(ref_path):
                print(f"  No reference file found — skipping")
                results.append({
                    "Model"         : model,
                    "Sample"        : sample_name,
                    "Ref Label"     : None,
                    "Ref Conf"      : None,
                    "Ref Top2 Label": None,
                    "Ref Top2 Conf" : None,
                    "TTS Label"     : None,
                    "TTS Conf"      : None,
                    "TTS Top2 Label": None,
                    "TTS Top2 Conf" : None,
                    "Pass"          : "SKIP",
                    "Flag"          : "NO_REF",
                    "_is_degraded"  : False,
                })
                continue

            ref_label, ref_conf, ref_top2_label, ref_top2_conf = get_emotion(ref_path, ser_model)
            print(f"  Ref    : {ref_label} ({ref_conf})  runner-up: {ref_top2_label} ({ref_top2_conf})")

            tts_label, tts_conf, tts_top2_label, tts_top2_conf = get_emotion(tts_path, ser_model)
            print(f"  TTS    : {tts_label} ({tts_conf})  runner-up: {tts_top2_label} ({tts_top2_conf})")

            if ref_label is None or tts_label is None:
                flag        = "ERROR"
                passed      = "ERROR"
                is_degraded = False

            elif ref_conf < CONFIDENCE_THRESHOLD:
                flag        = "LOW_CONF_REF"
                is_degraded = True
                if ref_label == tts_label:
                    passed = "PASS"
                else:
                    # Still apply near-miss even on low-conf ref
                    passed = _classify_pass(
                        ref_label, ref_conf, ref_top2_conf,
                        tts_label, tts_conf, tts_top2_label, tts_top2_conf,
                        NEAR_MISS_MARGIN
                    )

            else:
                flag        = "—"
                is_degraded = False
                passed = _classify_pass(
                    ref_label, ref_conf, ref_top2_conf,
                    tts_label, tts_conf, tts_top2_label, tts_top2_conf,
                    NEAR_MISS_MARGIN
                )

            if is_short:
                flag        = "SHORT_SEGMENT"
                is_degraded = True

            print(f"  Result : {passed} | Flag: {flag}")

            results.append({
                "Model"         : model,
                "Sample"        : sample_name,
                "Ref Label"     : ref_label,
                "Ref Conf"      : ref_conf,
                "Ref Top2 Label": ref_top2_label,
                "Ref Top2 Conf" : ref_top2_conf,
                "TTS Label"     : tts_label,
                "TTS Conf"      : tts_conf,
                "TTS Top2 Label": tts_top2_label,
                "TTS Top2 Conf" : tts_top2_conf,
                "Pass"          : passed,
                "Flag"          : flag,
                "_is_degraded"  : is_degraded,
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

        clean_total     = len(clean_df)
        clean_pass      = (clean_df["Pass"] == "PASS").sum()
        clean_near_miss = (clean_df["Pass"] == "NEAR_MISS").sum()
        clean_fail      = (clean_df["Pass"] == "FAIL").sum()

        deg_total = len(degraded_df)
        deg_pass  = (degraded_df["Pass"] == "PASS").sum()

        fail_df              = clean_df[clean_df["Pass"] == "FAIL"]
        most_common_mismatch = (
            fail_df["TTS Label"].value_counts().index[0]
            if len(fail_df) > 0 else "—"
        )

        def rate(n, d):
            return f"{n}/{d}" if d > 0 else "—"

        summary_rows.append({
            "Model"             : model,
            "Total Segments"    : total,
            "Clean Segments"    : clean_total,
            "Clean Pass Rate"   : rate(clean_pass,      clean_total),
            "Clean Near Miss"   : rate(clean_near_miss, clean_total),
            "Clean Fail Rate"   : rate(clean_fail,      clean_total),
            "Degraded Segments" : deg_total,
            "Degraded Pass Rate": rate(deg_pass, deg_total),
            "Common Mismatch"   : most_common_mismatch,
        })

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_sort_pass"] = summary_df["Clean Pass Rate"].apply(parse_rate)
    summary_df["_sort_deg"]  = summary_df["Degraded Pass Rate"].apply(parse_rate)
    summary_df = summary_df.sort_values(
        by=["_sort_pass", "_sort_deg"], ascending=[False, False]
    ).drop(columns=["_sort_pass", "_sort_deg"])

    return df, summary_df


def _classify_pass(ref_label, ref_conf, ref_top2_conf,
                   tts_label, tts_conf, tts_top2_label, tts_top2_conf,
                   margin):
    """Return PASS, NEAR_MISS, or FAIL for one segment."""
    if ref_label == tts_label:
        return "PASS"

    # TTS near-miss: ref label was runner-up in TTS output and within margin
    tts_near = (
        tts_top2_label == ref_label and
        tts_top2_conf is not None and
        (tts_conf - tts_top2_conf) <= margin
    )

    # Ref near-miss: reference distribution itself was uncertain
    ref_near = (
        ref_top2_conf is not None and
        (ref_conf - ref_top2_conf) <= margin
    )

    return "NEAR_MISS" if (tts_near or ref_near) else "FAIL"


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample",
        "Ref Label", "Ref Conf", "Ref Top2 Label", "Ref Top2 Conf",
        "TTS Label", "TTS Conf", "TTS Top2 Label", "TTS Top2 Conf",
        "Pass", "Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Clean Pass Rate", "Clean Near Miss", "Clean Fail Rate",
        "Degraded Pass Rate", "Common Mismatch"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Clean Pass Rate  → primary rank — TTS emotion matches reference")
    print("Clean Near Miss  → FAILs where top-2 distributions were close (uncertain model)")
    print("Clean Fail Rate  → clear mismatches — TTS emotion diverges from reference")
    print("Common Mismatch  → what emotion TTS produces when it clearly fails")
    print(f"\nConfidence threshold : {config.SER_CONFIDENCE_THRESHOLD}")
    print(f"Near-miss margin     : {config.SER_NEAR_MISS_MARGIN}")


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
