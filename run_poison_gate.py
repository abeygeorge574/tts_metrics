"""
run_poison_gate.py
------------------
Run gate_artifact.py against data/models_poison/ by temporarily patching
config.MODELS_DIR.  Does NOT modify config.py or gate_artifact.py.

Usage:
    conda run -n base python run_poison_gate.py --output-dir output/poison_test
"""

import os
import sys
import argparse

ROOT = "/Users/abey/Documents/tts_metrics"
sys.path.insert(0, ROOT)

import config

# ── Patch config in-memory (no file changes) ───────────────────────────────────
POISON_DIR = os.path.join(ROOT, "data", "models_poison")
config.MODELS_DIR = POISON_DIR

# Also set REFERENCE_DIR to something that exists (gate uses it for info only)
# gate_artifact.py reads config.REFERENCE_DIR but doesn't gate on it
config.REFERENCE_DIR = os.path.join(ROOT, "data", "reference")

print(f"[run_poison_gate] MODELS_DIR patched → {config.MODELS_DIR}")
print(f"[run_poison_gate] REFERENCE_DIR      → {config.REFERENCE_DIR}")

# ── Import gate AFTER config patch ────────────────────────────────────────────
# gate_artifact reads config at import-time only for _SR/_FRAME_SZ constants;
# MODELS_DIR is read inside run_gate(), so patching before import is safe.
sys.path.insert(0, os.path.join(ROOT, "gates"))
import gate_artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        default=os.path.join(ROOT, "output", "poison_test"))
    args = parser.parse_args()

    model_state    = gate_artifact.load_model()
    df, summary_df = gate_artifact.run_gate(model_state)
    gate_artifact.print_results(df, summary_df)
    gate_artifact.save_results(df, summary_df, args.output_dir)

    print(f"\n[run_poison_gate] Results saved to {args.output_dir}")
