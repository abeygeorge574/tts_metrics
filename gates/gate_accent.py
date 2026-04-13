"""
Gate: Accent Classification
Env : base (python 3.13)
Uses Jzuluaga/accent-id-commonaccent_ecapa (SpeechBrain ECAPA-TDNN, 16 labels) to
classify the accent of each TTS output. Pass = P(us) + P(canada) >= threshold.

16 accent labels: us, england, indian, australia, canada, scotland, wales,
                  singapore, hongkong, african, southatlantic, bermuda, malaysia,
                  newzealand, philippines, ireland

No reference audio needed.
Fallback: set env var ACCENT_USE_DIMA806=1 to use dima806/english_accents_classification
          (transformers pipeline, 5 labels) instead of ECAPA.
"""

import os
import sys
import argparse

import torch
import torchaudio
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    if os.environ.get("ACCENT_USE_DIMA806"):
        return _load_dima806()
    return _load_ecapa()


def _load_ecapa():
    import logging
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from speechbrain.inference.classifiers import EncoderClassifier

    savedir = os.path.join(os.path.expanduser("~"), ".cache", "accent_ecapa")
    print("Loading Jzuluaga/accent-id-commonaccent_ecapa (ECAPA-TDNN, 16 labels)...")
    clf = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir=savedir,
        run_opts={"device": "cpu"},
    )
    labels = list(clf.hparams.label_encoder.ind2lab.values())
    print(f"ECAPA loaded. Labels ({len(labels)}): {labels}")
    return {"clf": clf, "backend": "ecapa", "labels": labels}


def _load_dima806():
    from transformers import pipeline
    print("Loading dima806/english_accents_classification (fallback, 5 labels)...")
    clf = pipeline("audio-classification", model="dima806/english_accents_classification")
    labels = list(clf.model.config.id2label.values())
    print(f"dima806 loaded. Labels ({len(labels)}): {labels}")
    return {"clf": clf, "backend": "dima806", "labels": labels}


# ── Classification ─────────────────────────────────────────────────────────────
def classify_file(audio_path, model_state):
    """Return dict of {label: probability} for one file. Probabilities sum to 1."""
    backend = model_state["backend"]
    clf     = model_state["clf"]

    if backend == "ecapa":
        signal, sr = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)

        with torch.no_grad():
            out_prob, score, index, label = clf.classify_batch(signal)

        # out_prob is log-posteriors [batch=1, num_classes] → exp() → probabilities
        probs = torch.exp(out_prob[0])
        # Normalise to handle any floating-point drift
        probs = probs / probs.sum()

        ind2lab = clf.hparams.label_encoder.ind2lab
        return {ind2lab[i]: round(float(probs[i]), 4) for i in range(len(probs))}

    else:  # dima806 via transformers pipeline
        results = clf(audio_path)
        return {r["label"]: round(r["score"], 4) for r in results}


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    MODELS_DIR    = model_state.get("models_dir") or config.MODELS_DIR
    TARGET_LABELS = config.ACCENT_TARGET_LABELS       # ["us", "canada"]
    THRESHOLD     = config.ACCENT_TARGET_THRESHOLD
    NEAR_MISS_M   = config.ACCENT_NEAR_MISS_MARGIN
    LEAK_THRESH   = config.ACCENT_LEAK_THRESHOLD
    USE_TOP_LABEL = (model_state.get("backend") == "ecapa")  # ECAPA: rank-based, dima806: prob-based

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    model_folders = sorted([
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    ])
    if not model_folders:
        raise ValueError(f"No model folders found in {MODELS_DIR}")
    print(f"Models found: {model_folders}")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}\nModel: {model}\n{'='*50}")
        model_path = os.path.join(MODELS_DIR, model)
        wav_files  = sorted([f for f in os.listdir(model_path) if f.endswith(".wav")])

        for wav_file in wav_files:
            sample_name = os.path.splitext(wav_file)[0]
            audio_path  = os.path.join(model_path, wav_file)

            try:
                scores      = classify_file(audio_path, model_state)
                target_prob = round(sum(scores.get(l, 0.0) for l in TARGET_LABELS), 4)
                sorted_labels = sorted(scores.items(), key=lambda x: -x[1])
                top_label     = sorted_labels[0][0]
                second_label  = sorted_labels[1][0] if len(sorted_labels) > 1 else None
                label_str     = "+".join(TARGET_LABELS)

                if USE_TOP_LABEL:
                    # ECAPA: 16 labels dilute probability — use rank instead of threshold
                    # PASS if top-1 label is North American
                    # NEAR_MISS if top-2 is North American and gap from top-1 is tiny (<0.005)
                    top1_prob   = sorted_labels[0][1]
                    top2_prob   = sorted_labels[1][1] if len(sorted_labels) > 1 else 0
                    top1_is_na  = top_label in TARGET_LABELS
                    top2_is_na  = second_label in TARGET_LABELS

                    non_target_scores = {l: s for l, s in scores.items() if l not in TARGET_LABELS}
                    max_leak_label = max(non_target_scores, key=non_target_scores.get) if non_target_scores else None
                    max_leak_score = non_target_scores.get(max_leak_label, 0.0) if max_leak_label else 0.0

                    if top1_is_na and max_leak_score >= LEAK_THRESH:
                        final_pass = f"PASS_WARN (leak={max_leak_label} {max_leak_score:.3f})"
                    elif top1_is_na:
                        final_pass = "PASS"
                    elif top2_is_na and (top1_prob - top2_prob) < 0.005:
                        final_pass = f"NEAR_MISS (top={top_label} {top1_prob:.3f})"
                    else:
                        final_pass = f"FAIL (top={top_label} {top1_prob:.3f})"
                else:
                    # dima806: 5 labels, probability threshold works fine
                    passed    = target_prob >= THRESHOLD
                    near_miss = (not passed) and (target_prob >= THRESHOLD - NEAR_MISS_M)
                    non_target_scores = {l: s for l, s in scores.items() if l not in TARGET_LABELS}
                    max_leak_label    = max(non_target_scores, key=non_target_scores.get) if non_target_scores else None
                    max_leak_score    = non_target_scores.get(max_leak_label, 0.0) if max_leak_label else 0.0
                    accent_leak       = passed and max_leak_score >= LEAK_THRESH

                    if near_miss:
                        final_pass = f"NEAR_MISS (top={top_label} {scores.get(top_label,0):.2f})"
                    elif not passed:
                        final_pass = f"FAIL (top={top_label} {scores.get(top_label,0):.2f})"
                    elif accent_leak:
                        final_pass = f"PASS_WARN (leak={max_leak_label} {max_leak_score:.2f})"
                    else:
                        final_pass = "PASS"

                print(f"  {sample_name} | P({label_str})={target_prob:.4f} | top={top_label} → {final_pass}")

                row = {
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Top_Label"   : top_label,
                    "Target_Prob" : target_prob,
                    "Final_Pass"  : final_pass,
                }
                # Store top 4 label probabilities
                for lbl, prob in sorted(scores.items(), key=lambda x: -x[1])[:4]:
                    row[f"P_{lbl}"] = prob

            except Exception as e:
                print(f"  {sample_name} ERROR: {e}")
                row = {
                    "Model": model, "Sample": sample_name,
                    "Top_Label": "ERROR", "Target_Prob": None, "Final_Pass": "ERROR",
                }

            results.append(row)

    df = pd.DataFrame(results)

    summary_rows = []
    for model in model_folders:
        mdf       = df[df["Model"] == model]
        total     = len(mdf)
        passes     = mdf["Final_Pass"].isin(["PASS", "PASS_WARN"]).sum()
        warns      = mdf["Final_Pass"].str.startswith("PASS_WARN").fillna(False).sum()
        near_miss  = mdf["Final_Pass"].str.startswith("NEAR_MISS").fillna(False).sum()
        fails      = mdf["Final_Pass"].str.startswith("FAIL").fillna(False).sum()
        top_labels = mdf["Top_Label"].value_counts().to_dict()
        top_str    = ", ".join(f"{k}:{v}" for k, v in top_labels.items())

        # For dima806: keep median probability as additional signal
        med_prob = mdf["Target_Prob"].median() if not USE_TOP_LABEL else None

        row = {
            "Model"     : model,
            "Segments"  : total,
            "Pass_Rate" : f"{passes}/{total}",
            "Warn"      : warns,
            "Near_Miss" : near_miss,
            "Fails"     : fails,
            "Top_Labels": top_str,
        }
        if not USE_TOP_LABEL:
            row[f"Median_P({'+'.join(TARGET_LABELS)})"] = round(med_prob, 4) if med_prob is not None else None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_pass_num"] = summary_df["Pass_Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df = summary_df.sort_values(
        by=["_pass_num", "Fails"],
        ascending=[False, True]
    ).drop(columns=["_pass_num"])
    return df, summary_df


# ── Print / save ───────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    label_str = "+".join(config.ACCENT_TARGET_LABELS)
    ecapa = not os.environ.get("ACCENT_USE_DIMA806")
    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print(f"\nBackend      : {'ECAPA — Jzuluaga/accent-id-commonaccent_ecapa (16 labels, rank-based)' if ecapa else 'dima806 (5 labels, probability threshold)'}")
    if ecapa:
        print(f"Pass rule    : top-1 predicted label is {config.ACCENT_TARGET_LABELS}")
        print(f"NEAR_MISS    : top-2 is NA and gap from top-1 < 0.005")
        print(f"PASS_WARN    : passes but a non-NA label >= {config.ACCENT_LEAK_THRESHOLD:.0%}")
        print(f"NA_Consistency: fraction of segments where NA is top-1 (matches pass rate for clean models)")
    else:
        print(f"Pass rule    : P({label_str}) >= {config.ACCENT_TARGET_THRESHOLD}")
        print(f"Near miss    : FAIL but within {config.ACCENT_NEAR_MISS_MARGIN:.0%} of threshold")


def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accent classification gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "accent"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    args = parser.parse_args()

    model_state = load_model()
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)

    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
