"""
Gate: WER + Intelligibility
Env : utmos (python 3.9)
Uses Whisper (MLX on Apple Silicon / CUDA / CPU fallback) to transcribe TTS audio,
then computes WER via jiwer and intelligibility from Whisper log-probs.
"""

import os
import sys
import math
import gc
import time
import argparse

import pandas as pd
import numpy as np
import jiwer
from jiwer import Compose, ToLowerCase, SubstituteWords, RemovePunctuation, Strip, ReduceToListOfListOfWords

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Text normalisation ─────────────────────────────────────────────────────────
_TRANSFORM = Compose([
    ToLowerCase(),
    SubstituteWords({
        "alright": "all right",
        "ok"     : "okay",
    }),
    RemovePunctuation(),
    Strip(),
])

def normalize_text(text):
    return _TRANSFORM(text)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import torch
    print("Detecting hardware...")

    if torch.cuda.is_available():
        import whisper as openai_whisper
        ENGINE     = "cuda"
        MODEL_TIER = "medium"
        print(f"CUDA detected. Loading Whisper '{MODEL_TIER}' on GPU...")
        whisper_model = openai_whisper.load_model(MODEL_TIER, device="cuda")
        print("Whisper CUDA engine ready.")
    else:
        try:
            import mlx_whisper
            ENGINE        = "mlx"
            MODEL_TIER    = "mlx-community/whisper-medium-mlx"
            whisper_model = None   # mlx_whisper loaded lazily per call
            print("Apple Silicon detected. Using MLX Whisper.")
        except ImportError:
            import whisper as openai_whisper
            ENGINE     = "cpu"
            MODEL_TIER = "base"
            print(f"CPU fallback. Loading Whisper '{MODEL_TIER}'...")
            whisper_model = openai_whisper.load_model(MODEL_TIER, device="cpu")
            print("Whisper CPU engine ready.")

    return {"model": whisper_model, "engine": ENGINE, "model_tier": MODEL_TIER}


# ── Transcription ──────────────────────────────────────────────────────────────
def transcribe(audio_path, engine, model_tier, whisper_model):
    if engine == "cuda":
        result = whisper_model.transcribe(
            audio_path, language="en", word_timestamps=True, fp16=True
        )
    elif engine == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_tier,
            language="en",
            word_timestamps=True
        )
    else:  # cpu
        result = whisper_model.transcribe(
            audio_path, language="en", word_timestamps=True, fp16=False
        )
    return result


# ── WER computation ────────────────────────────────────────────────────────────
def compute_wer(reference_text, hypothesis_text):
    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(hypothesis_text)

    if not ref_norm or not hyp_norm:
        return {
            "WER"                : 1.0 if ref_norm else 0.0,
            "Substitutions"      : 0,
            "Deletions"          : 0,
            "Insertions"         : 0,
            "Hits"               : 0,
            "Substitution_Detail": "—",
            "Deleted_Words"      : "—",
            "Inserted_Words"     : "—",
        }

    _ref_transform = Compose([
        ToLowerCase(),
        SubstituteWords({"alright": "all right", "ok": "okay"}),
        RemovePunctuation(),
        Strip(),
        ReduceToListOfListOfWords()
    ])
    _hyp_transform = Compose([
        ToLowerCase(),
        SubstituteWords({"alright": "all right", "ok": "okay"}),
        RemovePunctuation(),
        Strip(),
        ReduceToListOfListOfWords()
    ])

    out = jiwer.process_words(
        reference_text, hypothesis_text,
        reference_transform=_ref_transform,
        hypothesis_transform=_hyp_transform,
    )

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    substitution_pairs = []
    deleted_words      = []
    inserted_words     = []

    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            ref_w = ref_words[chunk.ref_start_idx] if chunk.ref_start_idx < len(ref_words) else "?"
            hyp_w = hyp_words[chunk.hyp_start_idx] if chunk.hyp_start_idx < len(hyp_words) else "?"
            substitution_pairs.append(f"{ref_w}→{hyp_w}")
        elif chunk.type == "delete":
            if chunk.ref_start_idx < len(ref_words):
                deleted_words.append(ref_words[chunk.ref_start_idx])
        elif chunk.type == "insert":
            if chunk.hyp_start_idx < len(hyp_words):
                inserted_words.append(hyp_words[chunk.hyp_start_idx])

    return {
        "WER"                : round(out.wer, 4),
        "Substitutions"      : out.substitutions,
        "Deletions"          : out.deletions,
        "Insertions"         : out.insertions,
        "Hits"               : out.hits,
        "Substitution_Detail": ", ".join(substitution_pairs) if substitution_pairs else "—",
        "Deleted_Words"      : ", ".join(deleted_words)      if deleted_words      else "—",
        "Inserted_Words"     : ", ".join(inserted_words)     if inserted_words     else "—",
    }


# ── Intelligibility extraction ─────────────────────────────────────────────────
def extract_intelligibility(result, mumble_threshold):
    word_log_probs = []
    low_conf_words = []

    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            prob     = word_data.get("probability", 1.0)
            word_txt = word_data.get("word", "").strip()
            log_prob = math.log(prob) if prob > 0 else -10.0
            word_log_probs.append(log_prob)
            if log_prob < mumble_threshold:
                low_conf_words.append(f"{word_txt}({round(log_prob, 2)})")

    total           = len(word_log_probs)
    passed          = sum(1 for lp in word_log_probs if lp >= mumble_threshold)
    mean_log_prob   = round(float(np.mean(word_log_probs)), 4) if word_log_probs else None
    intel_pass_rate = round(passed / total, 4) if total > 0 else None

    return {
        "Mean_LogProb"   : mean_log_prob,
        "Intel_Pass_Rate": intel_pass_rate,
        "Low_Conf_Words" : ", ".join(low_conf_words) if low_conf_words else "—",
    }


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    engine        = model_state["engine"]
    model_tier    = model_state["model_tier"]
    whisper_model = model_state["model"]

    BASE_DIR       = config.WER_BASE_DIR
    REFERENCES_DIR = os.path.join(BASE_DIR, "references")
    MODELS_DIR     = os.path.join(BASE_DIR, "models")

    for folder in [BASE_DIR, REFERENCES_DIR, MODELS_DIR]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
    print("Top level folders found.")

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
    for wav_file in sample_names:
        txt_name = os.path.splitext(wav_file)[0] + ".txt"
        txt_path = os.path.join(REFERENCES_DIR, txt_name)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(
                f"Missing reference for {wav_file} — expected: {txt_path}"
            )
    print("All reference txt files found.")
    print(f"\nReady: {len(model_folders)} models × {len(sample_names)} samples = "
          f"{len(model_folders) * len(sample_names)} evaluations.")

    WER_THRESHOLD    = config.WER_THRESHOLD
    INTEL_THRESHOLD  = config.INTEL_THRESHOLD
    MUMBLE_THRESHOLD = config.MUMBLE_THRESHOLD

    results = []

    for model in model_folders:
        print(f"\n{'='*50}")
        print(f"Model: {model}")
        print(f"{'='*50}")

        for wav_file in model_samples[model]:
            sample_name = os.path.splitext(wav_file)[0]
            audio_path  = os.path.join(MODELS_DIR, model, wav_file)
            txt_path    = os.path.join(REFERENCES_DIR, sample_name + ".txt")

            with open(txt_path, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()

            print(f"\n  Sample : {sample_name}")
            print(f"  Ref    : {reference_text}")

            try:
                result          = transcribe(audio_path, engine, model_tier, whisper_model)
                hypothesis_text = result["text"].strip()
                print(f"  Whisper: {hypothesis_text}")

                wer_data   = compute_wer(reference_text, hypothesis_text)
                intel_data = extract_intelligibility(result, MUMBLE_THRESHOLD)

                wer_pass   = wer_data["WER"] <= WER_THRESHOLD
                intel_pass = (
                    intel_data["Intel_Pass_Rate"] is not None and
                    intel_data["Intel_Pass_Rate"] >= INTEL_THRESHOLD
                )
                deletion_flag = wer_data["Deletions"] > 0

                print(f"  WER    : {wer_data['WER']} {'PASS' if wer_pass else 'FAIL'} | "
                      f"Intel: {intel_data['Intel_Pass_Rate']} {'PASS' if intel_pass else 'FAIL'}")
                if wer_data["Substitution_Detail"] != "—":
                    print(f"  Subs   : {wer_data['Substitution_Detail']}")
                if wer_data["Deleted_Words"] != "—":
                    print(f"  Deleted: {wer_data['Deleted_Words']}")
                if wer_data["Inserted_Words"] != "—":
                    print(f"  Inserted: {wer_data['Inserted_Words']}")
                if intel_data["Low_Conf_Words"] != "—":
                    print(f"  Low Conf: {intel_data['Low_Conf_Words']}")

                results.append({
                    "Model"               : model,
                    "Sample"              : sample_name,
                    "Reference"           : reference_text,
                    "Whisper"             : hypothesis_text,
                    "WER"                 : wer_data["WER"],
                    "Substitutions"       : wer_data["Substitutions"],
                    "Deletions"           : wer_data["Deletions"],
                    "Insertions"          : wer_data["Insertions"],
                    "Hits"                : wer_data["Hits"],
                    "Substitution_Detail" : wer_data["Substitution_Detail"],
                    "Deleted_Words"       : wer_data["Deleted_Words"],
                    "Inserted_Words"      : wer_data["Inserted_Words"],
                    "Mean_LogProb"        : intel_data["Mean_LogProb"],
                    "Intel_Pass_Rate"     : intel_data["Intel_Pass_Rate"],
                    "Low_Conf_Words"      : intel_data["Low_Conf_Words"],
                    "WER_Pass"            : "PASS" if wer_pass   else "FAIL",
                    "Intel_Pass"          : "PASS" if intel_pass else "FAIL",
                    "Deletion_Flag"       : "FLAG" if deletion_flag else "OK",
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"               : model,
                    "Sample"              : sample_name,
                    "Reference"           : reference_text,
                    "Whisper"             : "ERROR",
                    "WER"                 : 1.0,
                    "Substitutions"       : 0,
                    "Deletions"           : 0,
                    "Insertions"          : 0,
                    "Hits"                : 0,
                    "Substitution_Detail" : "—",
                    "Deleted_Words"       : "—",
                    "Inserted_Words"      : "—",
                    "Mean_LogProb"        : None,
                    "Intel_Pass_Rate"     : None,
                    "Low_Conf_Words"      : "ERROR",
                    "WER_Pass"            : "FAIL",
                    "Intel_Pass"          : "FAIL",
                    "Deletion_Flag"       : "—",
                })

            finally:
                if "result" in dir():
                    del result
                gc.collect()
                if engine == "mlx":
                    time.sleep(0.5)

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)
    df["Both_Pass"] = df.apply(
        lambda row: "PASS" if (row["WER_Pass"] == "PASS" and row["Intel_Pass"] == "PASS") else "FAIL",
        axis=1
    )

    summary_rows = []
    for model in model_folders:
        model_df     = df[df["Model"] == model]
        total        = len(model_df)
        wer_vals     = model_df["WER"]
        logprob_vals = model_df["Mean_LogProb"].dropna()

        wer_pass_count   = (model_df["WER_Pass"]   == "PASS").sum()
        intel_pass_count = (model_df["Intel_Pass"] == "PASS").sum()
        both_pass_count  = (model_df["Both_Pass"]  == "PASS").sum()

        summary_rows.append({
            "Model"              : model,
            "Segments"           : total,
            "Both Pass Rate"     : f"{both_pass_count}/{total}",
            "WER Pass Rate"      : f"{wer_pass_count}/{total}",
            "Intel Pass Rate"    : f"{intel_pass_count}/{total}",
            "Median WER"         : round(wer_vals.median(), 4),
            "Max WER"            : round(wer_vals.max(), 4),
            "Median LogProb"     : round(logprob_vals.median(), 4) if len(logprob_vals) > 0 else None,
            "Total Deletions"    : model_df["Deletions"].sum(),
            "Total Substitutions": model_df["Substitutions"].sum(),
            "Total Insertions"   : model_df["Insertions"].sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_both_pass_num"]  = summary_df["Both Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df["_wer_pass_num"]   = summary_df["WER Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df["_intel_pass_num"] = summary_df["Intel Pass Rate"].apply(lambda x: int(x.split("/")[0]))
    summary_df = summary_df.sort_values(
        by=["_both_pass_num", "_wer_pass_num", "_intel_pass_num", "Total Deletions", "Median WER", "Max WER"],
        ascending=[False, False, False, True, True, True]
    ).drop(columns=["_both_pass_num", "_wer_pass_num", "_intel_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Reference", "Whisper",
        "WER", "Substitutions", "Deletions", "Insertions",
        "Substitution_Detail", "Deleted_Words", "Inserted_Words",
        "Mean_LogProb", "Intel_Pass_Rate", "Low_Conf_Words",
        "WER_Pass", "Intel_Pass", "Both_Pass", "Deletion_Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model", "Both Pass Rate", "WER Pass Rate", "Intel Pass Rate",
        "Median WER", "Max WER", "Median LogProb",
        "Total Deletions", "Total Substitutions"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Both Pass Rate   → primary ranking — must pass WER and Intelligibility")
    print("Median WER       → typical error level")
    print("Total Deletions  → words completely dropped — safety critical")
    print(f"\nThresholds: WER <= {config.WER_THRESHOLD} | Intel >= {config.INTEL_THRESHOLD} | Mumble log-prob < {config.MUMBLE_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WER + Intelligibility gate")
    parser.add_argument("--output-dir", default=os.path.join(config.OUTPUT_DIR, "wer"))
    args = parser.parse_args()

    model_state  = load_model()
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
