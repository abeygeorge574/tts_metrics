"""
Gate: Accent Classification
Env : utmos (python 3.9)
Uses wav2vec2-large-xlsr-53 to extract embeddings, then computes cosine
similarity to accent reference embeddings (american / british / indian).
Pass = target accent proximity >= threshold.
"""

import os
import sys
import argparse

import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"

    print(f"Loading {MODEL_NAME}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    wav2vec2          = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    wav2vec2.eval()

    print(f"wav2vec2 loaded: {MODEL_NAME}")
    return {"feature_extractor": feature_extractor, "wav2vec2": wav2vec2}


# ── Embedding and similarity ───────────────────────────────────────────────────
def get_accent_embedding(audio_path, feature_extractor, wav2vec2):
    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    if wav.shape[-1] < 160:
        raise ValueError(f"Audio too short: {audio_path}")

    inputs = feature_extractor(
        wav.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec2(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


def cosine_sim(emb1, emb2):
    e1 = emb1 / torch.norm(emb1)
    e2 = emb2 / torch.norm(emb2)
    return round(float(torch.dot(e1, e2)), 4)


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    feature_extractor = model_state["feature_extractor"]
    wav2vec2          = model_state["wav2vec2"]

    BASE_DIR   = config.ACCENT_BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    ACCENT_REFERENCES     = config.ACCENT_REFERENCES
    TARGET_ACCENT         = config.ACCENT_TARGET
    TARGET_THRESHOLD      = config.ACCENT_TARGET_THRESHOLD
    CHARACTER_MAP         = None   # set to a dict of {char_name: [sample_names]} to group by character

    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    for accent_name, ref_path in ACCENT_REFERENCES.items():
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Accent reference not found for '{accent_name}': {ref_path}\n"
                f"Provide a clean audio clip of any speaker with that accent."
            )
    print(f"All {len(ACCENT_REFERENCES)} accent reference files found.")

    if TARGET_ACCENT not in ACCENT_REFERENCES:
        raise ValueError(
            f"TARGET_ACCENT '{TARGET_ACCENT}' not in ACCENT_REFERENCES.\n"
            f"Available: {list(ACCENT_REFERENCES.keys())}"
        )

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

    total = len(model_folders) * len(model_samples[model_folders[0]])
    print(f"\nReady: {len(model_folders)} models × {len(model_samples[model_folders[0]])} samples = {total} evaluations")

    # pre-compute accent reference embeddings
    print("\nExtracting accent reference embeddings...")
    accent_ref_embeddings = {}
    for accent_name, ref_path in ACCENT_REFERENCES.items():
        try:
            emb = get_accent_embedding(ref_path, feature_extractor, wav2vec2)
            accent_ref_embeddings[accent_name] = emb
            print(f"  {accent_name}: {ref_path}")
        except Exception as e:
            print(f"  {accent_name} failed: {e}")
            accent_ref_embeddings[accent_name] = None
    print("Reference embeddings ready.")

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        embeddings = {}
        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            audio_path  = os.path.join(MODELS_DIR, model, wav_file)

            try:
                emb = get_accent_embedding(audio_path, feature_extractor, wav2vec2)
                embeddings[sample_name] = emb
                print(f"  Embedded: {sample_name}")
            except Exception as e:
                print(f"  Embedding failed: {sample_name} — {e}")
                embeddings[sample_name] = None

        if CHARACTER_MAP is not None:
            char_groups = CHARACTER_MAP
        else:
            char_groups = {"all": [os.path.splitext(f)[0] for f in model_samples[model]]}

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            emb         = embeddings.get(sample_name)

            if emb is None:
                row = {
                    "Model"         : model,
                    "Sample"        : sample_name,
                    "Character"     : "—",
                    "Closest Accent": "—",
                    "Target Pass"   : "—",
                    "Final Pass"    : "ERROR",
                }
                for accent_name in ACCENT_REFERENCES:
                    row[f"Proximity_{accent_name}"] = None
                results.append(row)
                continue

            proximities = {}
            for accent_name, ref_emb in accent_ref_embeddings.items():
                if ref_emb is not None:
                    proximities[accent_name] = cosine_sim(emb, ref_emb)
                else:
                    proximities[accent_name] = None

            target_proximity = proximities.get(TARGET_ACCENT)
            target_pass      = (target_proximity >= TARGET_THRESHOLD) if target_proximity is not None else None

            valid_prox     = {k: v for k, v in proximities.items() if v is not None}
            closest_accent = max(valid_prox, key=valid_prox.get) if valid_prox else "—"

            character = "unknown"
            for char_name, char_samples in char_groups.items():
                if sample_name in char_samples:
                    character = char_name
                    break

            if target_pass is False:
                final_pass = f"FAIL (Accent drift from {TARGET_ACCENT})"
            elif target_pass is None:
                final_pass = "ERROR"
            else:
                final_pass = "PASS"

            prox_str = " | ".join([f"{k}: {v}" for k, v in proximities.items() if v is not None])
            print(f"  {sample_name} | {prox_str} | Closest: {closest_accent} → {final_pass}")

            row = {
                "Model"         : model,
                "Sample"        : sample_name,
                "Character"     : character,
                "Closest Accent": closest_accent,
                "Target Pass"   : "PASS" if target_pass else "FAIL" if target_pass is not None else "—",
                "Final Pass"    : final_pass,
            }
            for accent_name in ACCENT_REFERENCES:
                row[f"Proximity_{accent_name}"] = proximities.get(accent_name)

            results.append(row)

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    proximity_cols = [f"Proximity_{a}" for a in ACCENT_REFERENCES]

    summary_rows = []
    for model in model_folders:
        model_df      = df[df["Model"] == model]
        total         = len(model_df)
        on_target     = (model_df["Closest Accent"] == TARGET_ACCENT).sum()
        label_consist = f"{on_target}/{total}"
        target_fails  = model_df["Final Pass"].str.contains("FAIL").sum()

        failing_df   = model_df[model_df["Final Pass"].str.contains("FAIL")]
        drift_counts = failing_df["Closest Accent"].value_counts().to_dict()
        drift_str    = ", ".join([f"{k}: {v}" for k, v in drift_counts.items()]) if drift_counts else "—"

        row = {
            "Model"             : model,
            "Segments"          : total,
            "Label Consistency" : label_consist,
            "Target Drift Fails": target_fails,
            "Drift Breakdown"   : drift_str,
        }
        for accent_name in ACCENT_REFERENCES:
            col  = f"Proximity_{accent_name}"
            vals = model_df[col].dropna()
            row[f"Median_{accent_name}"] = round(vals.median(), 4) if len(vals) > 0 else None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    def parse_rate(rate_str):
        if rate_str == "—":
            return -1
        return int(rate_str.split("/")[0])

    summary_df["_label_consist_num"] = summary_df["Label Consistency"].apply(parse_rate)
    summary_df["_median_target"]     = summary_df[f"Median_{TARGET_ACCENT}"].fillna(-999)

    summary_df = summary_df.sort_values(
        by=["_label_consist_num", "_median_target"],
        ascending=[False, False]
    ).drop(columns=["_label_consist_num", "_median_target"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    TARGET_ACCENT    = config.ACCENT_TARGET
    ACCENT_REFERENCES = config.ACCENT_REFERENCES
    proximity_cols   = [f"Proximity_{a}" for a in ACCENT_REFERENCES]

    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Character", "Closest Accent",
        *proximity_cols,
        "Final Pass"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    summary_cols = [
        "Model", "Label Consistency", f"Median_{TARGET_ACCENT}",
        "Target Drift Fails", "Drift Breakdown"
    ]
    print(summary_df[summary_cols].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print(f"Label Consistency   → segments where closest accent was {TARGET_ACCENT}")
    print(f"Proximity_{TARGET_ACCENT}  → cosine similarity — below {config.ACCENT_TARGET_THRESHOLD} = drifting")
    print(f"Drift Breakdown     → where failing segments drifted")
    print(f"\nThreshold: Proximity_{TARGET_ACCENT} >= {config.ACCENT_TARGET_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accent classification gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "accent"))
    args = parser.parse_args()

    model_state    = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
