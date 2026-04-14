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
from jiwer import Compose, ToLowerCase, SubstituteWords, RemovePunctuation, Strip, ReduceToListOfListOfWords, SubstituteRegexes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Text normalisation ─────────────────────────────────────────────────────────
_TRANSFORM = Compose([
    ToLowerCase(),
    SubstituteWords({
        "alright": "all right",
        "ok"     : "okay",
    }),
    # Replace hyphens and em/en dashes with spaces BEFORE punctuation removal,
    # so "out-of-syllabus" → "out of syllabus" rather than "outofsyllabus".
    SubstituteRegexes({r"[-\u2013\u2014]": " "}),
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
            if not os.environ.get("WHISPER_FORCE_CPU"):
                import mlx_whisper
                ENGINE        = "mlx"
                MODEL_TIER    = "mlx-community/whisper-medium-mlx"
                whisper_model = None   # mlx_whisper loaded lazily per call
                print("Apple Silicon detected. Using MLX Whisper.")
            else:
                raise ImportError("WHISPER_FORCE_CPU set — skipping MLX")
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
            "WER (threshold≤0.10)"  : 1.0 if ref_norm else 0.0,
            "Substitutions"         : 0,
            "Deletions"             : 0,
            "Insertions"            : 0,
            "Hits"                  : 0,
            "Substitution_Detail"   : "—",
            "Deleted_Words"         : "—",
            "Inserted_Words"        : "—",
        }

    _ref_transform = Compose([
        ToLowerCase(),
        SubstituteWords({"alright": "all right", "ok": "okay"}),
        SubstituteRegexes({r"[-\u2013\u2014]": " "}),
        RemovePunctuation(),
        Strip(),
        ReduceToListOfListOfWords()
    ])
    _hyp_transform = Compose([
        ToLowerCase(),
        SubstituteWords({"alright": "all right", "ok": "okay"}),
        SubstituteRegexes({r"[-\u2013\u2014]": " "}),
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
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ref_w = ref_words[i] if i < len(ref_words) else "?"
                hyp_i = chunk.hyp_start_idx + (i - chunk.ref_start_idx)
                hyp_w = hyp_words[hyp_i] if hyp_i < len(hyp_words) else "?"
                substitution_pairs.append(f"{ref_w}→{hyp_w}")
        elif chunk.type == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if i < len(ref_words):
                    deleted_words.append(ref_words[i])
        elif chunk.type == "insert":
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                if i < len(hyp_words):
                    inserted_words.append(hyp_words[i])

    return {
        "WER (threshold≤0.10)"  : round(out.wer, 4),
        "Substitutions"         : out.substitutions,
        "Deletions"             : out.deletions,
        "Insertions"            : out.insertions,
        "Hits"                  : out.hits,
        "Substitution_Detail"   : ", ".join(substitution_pairs) if substitution_pairs else "—",
        "Deleted_Words"         : ", ".join(deleted_words)      if deleted_words      else "—",
        "Inserted_Words"        : ", ".join(inserted_words)     if inserted_words     else "—",
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
        "Mean_LogProb"                         : mean_log_prob,
        "Intelligibility Rate (threshold≥0.85)" : intel_pass_rate,
        "Low_Conf_Words"                        : ", ".join(low_conf_words) if low_conf_words else "—",
    }


# ── Combined verdict helper ────────────────────────────────────────────────────
def _combined_verdict(wer_verdict, intel_verdict):
    """
    PASS   — both pass
    NEAR_MISS — either is NEAR_MISS and neither hard-fails
    FAIL   — either hard-fails
    """
    if wer_verdict == "FAIL" or intel_verdict == "FAIL":
        return "FAIL"
    if wer_verdict == "NEAR_MISS" or intel_verdict == "NEAR_MISS":
        return "NEAR_MISS"
    return "PASS"


# ── Main gate ──────────────────────────────────────────────────────────────────
def run_gate(model_state=None):
    if model_state is None:
        model_state = load_model()

    engine        = model_state["engine"]
    model_tier    = model_state["model_tier"]
    whisper_model = model_state["model"]

    MODELS_DIR     = model_state.get("models_dir") or config.MODELS_DIR
    REFERENCES_DIR = model_state.get("refs_dir")   or config.TEXT_REFERENCE_DIR

    for folder in [REFERENCES_DIR, MODELS_DIR]:
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

    WER_THRESHOLD     = config.WER_THRESHOLD
    INTEL_THRESHOLD   = config.INTEL_THRESHOLD
    MUMBLE_THRESHOLD  = config.MUMBLE_THRESHOLD
    WER_NM_MARGIN     = config.WER_NEAR_MISS_MARGIN            # 0.20
    WER_NM_UPPER      = WER_THRESHOLD * (1 + WER_NM_MARGIN)    # 0.12
    INTEL_NM_LOWER    = INTEL_THRESHOLD * (1 - WER_NM_MARGIN)  # 0.68

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

                wer_val   = wer_data["WER (threshold≤0.10)"]
                intel_val = intel_data["Intelligibility Rate (threshold≥0.85)"]

                # WER verdict
                if wer_val <= WER_THRESHOLD:
                    wer_verdict = "PASS"
                elif wer_val <= WER_NM_UPPER:
                    wer_verdict = "NEAR_MISS"
                else:
                    wer_verdict = "FAIL"

                # Intel verdict
                if intel_val is None:
                    intel_verdict = "FAIL"
                elif intel_val >= INTEL_THRESHOLD:
                    intel_verdict = "PASS"
                elif intel_val >= INTEL_NM_LOWER:
                    intel_verdict = "NEAR_MISS"
                else:
                    intel_verdict = "FAIL"

                final_verdict = _combined_verdict(wer_verdict, intel_verdict)
                deletion_flag = wer_data["Deletions"] > 0

                print(f"  WER    : {wer_val} {wer_verdict} | "
                      f"Intel: {intel_val} {intel_verdict} | Final: {final_verdict}")
                if wer_data["Substitution_Detail"] != "—":
                    print(f"  Subs   : {wer_data['Substitution_Detail']}")
                if wer_data["Deleted_Words"] != "—":
                    print(f"  Deleted: {wer_data['Deleted_Words']}")
                if wer_data["Inserted_Words"] != "—":
                    print(f"  Inserted: {wer_data['Inserted_Words']}")
                if intel_data["Low_Conf_Words"] != "—":
                    print(f"  Low Conf: {intel_data['Low_Conf_Words']}")

                total_ref_words = wer_data["Hits"] + wer_data["Substitutions"] + wer_data["Deletions"]
                hit_rate = round(wer_data["Hits"] / total_ref_words, 4) if total_ref_words > 0 else None

                results.append({
                    "Model"                                    : model,
                    "Sample"                                   : sample_name,
                    "Reference"                                : reference_text,
                    "Whisper"                                  : hypothesis_text,
                    "WER (threshold≤0.10)"                     : wer_val,
                    "Substitutions"                            : wer_data["Substitutions"],
                    "Deletions"                                : wer_data["Deletions"],
                    "Insertions"                               : wer_data["Insertions"],
                    "Hit_Rate"                                 : hit_rate,
                    "Substitution_Detail"                      : wer_data["Substitution_Detail"],
                    "Deleted_Words"                            : wer_data["Deleted_Words"],
                    "Inserted_Words"                           : wer_data["Inserted_Words"],
                    "Mean_LogProb"                             : intel_data["Mean_LogProb"],
                    "Intelligibility Rate (threshold≥0.85)"    : intel_val,
                    "Low_Conf_Words"                           : intel_data["Low_Conf_Words"],
                    "WER Pass (threshold≤0.10)"                : wer_verdict,
                    "Intel Pass (threshold≥0.85)"              : intel_verdict,
                    "Final Pass (PASS/NEAR_MISS/FAIL)"         : final_verdict,
                    "Deletion_Flag"                            : "FLAG" if deletion_flag else "OK",
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "Model"                                    : model,
                    "Sample"                                   : sample_name,
                    "Reference"                                : reference_text,
                    "Whisper"                                  : "ERROR",
                    "WER (threshold≤0.10)"                     : 1.0,
                    "Substitutions"                            : 0,
                    "Deletions"                                : 0,
                    "Insertions"                               : 0,
                    "Hit_Rate"                                 : None,
                    "Substitution_Detail"                      : "—",
                    "Deleted_Words"                            : "—",
                    "Inserted_Words"                           : "—",
                    "Mean_LogProb"                             : None,
                    "Intelligibility Rate (threshold≥0.85)"    : None,
                    "Low_Conf_Words"                           : "ERROR",
                    "WER Pass (threshold≤0.10)"                : "FAIL",
                    "Intel Pass (threshold≥0.85)"              : "FAIL",
                    "Final Pass (PASS/NEAR_MISS/FAIL)"         : "FAIL",
                    "Deletion_Flag"                            : "—",
                })

            finally:
                if "result" in dir():
                    del result
                gc.collect()
                if engine == "mlx":
                    time.sleep(0.5)

    print("\n\nAll evaluations complete.")

    df = pd.DataFrame(results)

    fp_col    = "Final Pass (PASS/NEAR_MISS/FAIL)"
    wer_col   = "WER (threshold≤0.10)"
    intel_col = "Intelligibility Rate (threshold≥0.85)"
    wp_col    = "WER Pass (threshold≤0.10)"
    ip_col    = "Intel Pass (threshold≥0.85)"

    summary_rows = []
    for model in model_folders:
        model_df     = df[df["Model"] == model]
        total        = len(model_df)
        wer_vals     = model_df[wer_col]
        logprob_vals = model_df["Mean_LogProb"].dropna()

        wer_pass_count   = (model_df[wp_col] == "PASS").sum()
        intel_pass_count = (model_df[ip_col] == "PASS").sum()
        final_pass_count = (model_df[fp_col] == "PASS").sum()
        near_miss_count  = (model_df[fp_col] == "NEAR_MISS").sum()

        summary_rows.append({
            "Model"                              : model,
            "Segments"                           : total,
            "WER Pass Rate (PASS / total)"       : f"{wer_pass_count}/{total}",
            "Intel Pass Rate (PASS / total)"     : f"{intel_pass_count}/{total}",
            "Final Pass Rate (PASS / total)"     : f"{final_pass_count}/{total}",
            "Near Miss (WER or Intel marginal)"  : int(near_miss_count),
            "Median WER"                         : round(wer_vals.median(), 4),
            "Max WER"                            : round(wer_vals.max(), 4),
            "Median LogProb"                     : round(logprob_vals.median(), 4) if len(logprob_vals) > 0 else None,
            "Total Deletions"                    : model_df["Deletions"].sum(),
            "Total Substitutions"                : model_df["Substitutions"].sum(),
            "Total Insertions"                   : model_df["Insertions"].sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["_wer_pass_num"] = summary_df["WER Pass Rate (PASS / total)"].apply(lambda x: int(x.split("/")[0]))
    summary_df = summary_df.sort_values(
        # Dubbing priority: pass rate → dropped words → wrong words → worst segment →
        #                   typical quality → extra words → whisper confidence
        by=["_wer_pass_num", "Total Deletions", "Total Substitutions",
            "Median WER", "Max WER", "Total Insertions", "Median LogProb"],
        ascending=[False, True, True, True, True, True, False]
    ).drop(columns=["_wer_pass_num"])

    return df, summary_df


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(df, summary_df):
    print("\n========== FULL PER-SEGMENT RESULTS ==========")
    print(df[[
        "Model", "Sample", "Reference", "Whisper",
        "WER (threshold≤0.10)", "Substitutions", "Deletions", "Insertions", "Hit_Rate",
        "Substitution_Detail", "Deleted_Words", "Inserted_Words",
        "Mean_LogProb", "Intelligibility Rate (threshold≥0.85)", "Low_Conf_Words",
        "WER Pass (threshold≤0.10)", "Intel Pass (threshold≥0.85)",
        "Final Pass (PASS/NEAR_MISS/FAIL)", "Deletion_Flag"
    ]].to_string(index=False))

    print("\n========== MODEL COMPARISON SUMMARY ==========")
    print(summary_df[[
        "Model",
        "WER Pass Rate (PASS / total)",
        "Intel Pass Rate (PASS / total)",
        "Final Pass Rate (PASS / total)",
        "Near Miss (WER or Intel marginal)",
        "Median WER", "Max WER", "Median LogProb",
        "Total Deletions", "Total Substitutions"
    ]].to_string(index=False))

    print("\n========== WHAT TO LOOK FOR ==========")
    print("Final Pass Rate  → primary ranking — both WER and Intel must pass")
    print("NEAR_MISS        → WER in (0.10, 0.12] or Intel in [0.68, 0.85) — marginal")
    print("Hit_Rate         → correct words / total reference words (1.0 = perfect)")
    print("Median WER       → typical error level")
    print("Total Deletions  → words completely dropped — safety critical")
    print(f"\nThresholds: WER <= {config.WER_THRESHOLD} (NM <= {round(config.WER_THRESHOLD * (1 + config.WER_NEAR_MISS_MARGIN), 2)}) | "
          f"Intel >= {config.INTEL_THRESHOLD} (NM >= {round(config.INTEL_THRESHOLD * (1 - config.WER_NEAR_MISS_MARGIN), 2)}) | "
          f"Mumble log-prob < {config.MUMBLE_THRESHOLD}")


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(df, summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "per_segment_results.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)
    print(f"Results saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WER + Intelligibility gate")
    parser.add_argument("--output-dir",  default=os.path.join(config.OUTPUT_DIR, "wer"))
    parser.add_argument("--models-dir",  default=None, help="Override config.MODELS_DIR")
    parser.add_argument("--refs-dir",    default=None, help="Override config.TEXT_REFERENCE_DIR")
    parser.add_argument("--ref-dir",     default=None, help="Audio reference dir (unused by WER, accepted for pipeline compatibility)")
    args = parser.parse_args()

    model_state  = load_model()
    if args.models_dir:
        model_state["models_dir"] = os.path.abspath(args.models_dir)
    if args.refs_dir:
        model_state["refs_dir"] = os.path.abspath(args.refs_dir)
    df, summary_df = run_gate(model_state)
    print_results(df, summary_df)
    save_results(df, summary_df, args.output_dir)
