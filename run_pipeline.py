"""
run_pipeline.py — main entry point for the TTS evaluation pipeline.

Gates and their environments:
  utmos env (python 3.9) : wer, utmos, accent
  base env  (python 3.13): nisqa, speaker_sim, ser, pitch, duration, vad, amplitude

Gates in the utmos env are invoked as subprocess calls using the conda env python,
so this script can run in either env (base is typical).

Usage:
  python run_pipeline.py                        # run all gates
  python run_pipeline.py --gates wer utmos      # run specific gates
  python run_pipeline.py --skip nisqa           # skip specific gates
  python run_pipeline.py --output-dir /path/to/out
"""

import os
import sys
import subprocess
import argparse

# ── locate config ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

# ── Gate registry ──────────────────────────────────────────────────────────────
# Each entry: (gate_key, module_path, env)
#   env = "utmos" → invoked via subprocess with the utmos conda python
#   env = "base"  → imported and called directly in this process
GATE_REGISTRY = [
    ("wer",        "gates/gate_wer.py",        "utmos"),
    ("nisqa",      "gates/gate_nisqa.py",       "base"),
    ("utmos",      "gates/gate_utmos.py",       "utmos"),
    ("speaker_sim","gates/gate_speaker_sim.py", "base"),
    ("ser",        "gates/gate_ser.py",         "base"),
    ("pitch",      "gates/gate_pitch.py",       "base"),
    ("duration",   "gates/gate_duration.py",    "base"),
    ("vad",        "gates/gate_vad.py",         "base"),
    ("amplitude",  "gates/gate_amplitude.py",   "base"),
    ("accent",     "gates/gate_accent.py",      "utmos"),
]

GATE_KEYS = [g[0] for g in GATE_REGISTRY]


def find_conda_python(env_name):
    """
    Locate the python executable for a named conda environment.
    Tries common miniconda/anaconda locations on macOS.
    """
    candidates = [
        os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python"),
        os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/python"),
        os.path.expanduser(f"~/opt/miniconda3/envs/{env_name}/bin/python"),
        os.path.expanduser(f"~/opt/anaconda3/envs/{env_name}/bin/python"),
        f"/opt/homebrew/Caskroom/miniconda/base/envs/{env_name}/bin/python",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # fallback: try conda run
    return None


def run_utmos_gate(gate_key, gate_script, output_dir):
    """Run a gate in the utmos conda environment via subprocess."""
    conda_python = find_conda_python(config.UTMOS_CONDA_ENV)
    gate_script_abs = os.path.join(ROOT, gate_script)
    gate_output_dir = os.path.join(output_dir, gate_key)

    if conda_python:
        cmd = [conda_python, gate_script_abs, "--output-dir", gate_output_dir]
        print(f"\n[{gate_key}] Running via {conda_python}")
    else:
        # fall back to conda run
        cmd = [
            "conda", "run", "-n", config.UTMOS_CONDA_ENV,
            "python", gate_script_abs, "--output-dir", gate_output_dir
        ]
        print(f"\n[{gate_key}] Running via conda run -n {config.UTMOS_CONDA_ENV}")

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[{gate_key}] FAILED (exit code {result.returncode})")
        return False

    print(f"\n[{gate_key}] DONE")
    return True


def run_base_gate(gate_key, gate_module_name, output_dir):
    """Import and run a gate directly in this process (base env)."""
    import importlib.util

    gate_script_abs = os.path.join(ROOT, f"gates/{gate_module_name}.py")
    spec   = importlib.util.spec_from_file_location(gate_module_name, gate_script_abs)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gate_output_dir = os.path.join(output_dir, gate_key)

    print(f"\n{'='*60}")
    print(f"Gate: {gate_key.upper()}")
    print(f"{'='*60}")

    try:
        model_state    = module.load_model()
        df, summary_df = module.run_gate(model_state)
        module.print_results(df, summary_df)
        module.save_results(df, summary_df, gate_output_dir)
        print(f"\n[{gate_key}] DONE")
        return True
    except Exception as e:
        print(f"\n[{gate_key}] FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TTS evaluation pipeline — runs all gates and saves results."
    )
    parser.add_argument(
        "--gates",
        nargs="*",
        choices=GATE_KEYS,
        default=None,
        help="Gate(s) to run. Default: all gates."
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=GATE_KEYS,
        default=[],
        help="Gate(s) to skip."
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Root output directory. Default: {config.OUTPUT_DIR}"
    )
    args = parser.parse_args()

    gates_to_run = args.gates if args.gates else GATE_KEYS
    gates_to_run = [g for g in gates_to_run if g not in (args.skip or [])]

    print(f"Pipeline starting.")
    print(f"Gates to run : {gates_to_run}")
    print(f"Output dir   : {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = {}

    for gate_key, gate_script, env in GATE_REGISTRY:
        if gate_key not in gates_to_run:
            continue

        gate_module = os.path.splitext(os.path.basename(gate_script))[0]

        if env == "utmos":
            success = run_utmos_gate(gate_key, gate_script, args.output_dir)
        else:
            success = run_base_gate(gate_key, gate_module, args.output_dir)

        results_summary[gate_key] = "PASS" if success else "FAIL"

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    for gate_key, status in results_summary.items():
        print(f"  {gate_key:<14} {status}")

    failed = [k for k, v in results_summary.items() if v == "FAIL"]
    if failed:
        print(f"\nFailed gates: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll gates completed successfully.")
        print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
