"""
Gate: Accent Classification
Env : utmos (python 3.9)
Uses dima806/english_accents_classification (ViT on mel-spectrogram) to classify
the accent of each TTS output. Pass = P(target_label) >= threshold.

No reference audio needed — the classifier outputs label probabilities directly.
Labels: us, england, australia, canada, indian, bermuda, southatlandtic, HongKong, malaysia, newzealand, philippines, singapore, southafrica, wales, ireland, scotland
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_model():
    from transformers import pipeline
    print("Loading dima806/english_accents_classification...")
    clf = pipeline(
        "audio-classification",
        model="dima806/english_accents_classification",
    )
    print("Accent classifier ready.")
    return {"clf": clf}


def classify_file(audio_path, clf):
    """Return dict of {label: score} for one file."""
    results = clf(audio_path)
    return {r["label"]: round(r["score"], 4) for r in results}


def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    clf        = model_state["clf"]
    MODELS_DIR = model_state.get("models_dir") or config.MODELS_DIR
    TARGET     = config.ACCENT_TARGET          # e.g. "us"
    THRESHOLD  = config.ACCENT_TARGET_THRESHOLD

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
                scores      = classify_file(audio_path, clf)
                target_prob = scores.get(TARGET, 0.0)
                top_label   = max(scores, key=scores.get)
                passed      = target_prob >= THRESHOLD
                final_pass  = "PASS" if passed else f"FAIL (top={top_label} {scores.get(top_label,0):.2f})"

                print(f"  {sample_name} | {TARGET}={target_prob:.3f} | top={top_label} → {final_pass}")

                row = {
                    "Model"       : model,
                    "Sample"      : sample_name,
                    "Top_Label"   : top_label,
                    "Target_Prob" : target_prob,
                    "Final_Pass"  : final_pass,
                }
                # add top-3 label probs
                for r in sorted(scores.items(), key=lambda x: -x[1])[:3]:
                    row[f"P_{r[0]}"] = r[1]

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
        mdf        = df[df["Model"] == model]
        total      = len(mdf)
        passes     = (mdf["Final_Pass"] == "PASS").sum()
        fails      = total - passes
        med_prob   = mdf["Target_Prob"].median()
        top_labels = mdf["Top_Label"].value_counts().to_dict()
        top_str    = ", ".join(f"{k}:{v}" for k, v in top_labels.items())

        summary_rows.append({
            "Model"           : model,
            "Segments"        : total,
            "Pass_Rate"       : f"{passes}/{total}",
            "Fails"           : fails,
            f"Median_P_{TARGET}": round(med_prob, 4) if med_prob is not None else None,
            "Top_Labels"      : top_str,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=f"Median_P_{TARGET}", ascending=False
    )
    return df, summary_df


def print_results(df, summary_df):
    TARGET = config.ACCENT_TARGET
    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print(f"\nThreshold: P({TARGET}) >= {config.ACCENT_TARGET_THRESHOLD}")
    print(f"Target label: '{TARGET}'")


def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


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
