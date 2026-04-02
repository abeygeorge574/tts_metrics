"""
run_pipeline.py — main entry point for the TTS evaluation pipeline.

Gates and their environments:
  utmos env (python 3.9) : wer, utmos, accent
  base env  (python 3.13): nisqa, speaker_sim, ser, arousal_valence, pitch, duration, vad, amplitude

Gates in the utmos env are invoked as subprocess calls using the conda env python,
so this script must be run from the base env.

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
import logging
from datetime import datetime

# ── Run directory ──────────────────────────────────────────────────────────────
# Each pipeline run gets its own timestamped folder under output/runs/.
# All gate CSVs and the log file go inside that folder.
# e.g. output/runs/2026-04-03_10-00-00/
#        pipeline.log
#        wer/
#        nisqa/
#        ...

_RUN_ID  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
_RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "runs", _RUN_ID)
os.makedirs(_RUN_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
_LOG_FILE = os.path.join(_RUN_DIR, "pipeline.log")

_formatter = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)

# Filter to block known-noisy third-party messages regardless of which logger emits them.
# setLevel() on named loggers doesn't work here because FunaASR/SpeechBrain emit via
# the root logger or loggers not in their package namespace.
_NOISE_PREFIXES = (
    "init param, map:",          # FunaASR: ~200 lines per model load
    "scope_map:",                # FunaASR
    "excludes:",                 # FunaASR
    "ckpt:",                     # FunaASR
    "Loading pretrained params", # FunaASR
    "Loading ckpt:",             # FunaASR
    "download models from model hub",  # FunaASR
    "Registered checkpoint ",    # SpeechBrain DEBUG
    "Registered parameter transfer ",  # SpeechBrain DEBUG
    "Set local path in self.paths",    # SpeechBrain DEBUG
    "Fetching files for pretraining",  # SpeechBrain DEBUG
    "Redirecting (loading from local", # SpeechBrain DEBUG
    "Loaded categorical encoding",     # SpeechBrain DEBUG
    "Loading pretrained files for",    # SpeechBrain INFO
    "Fetch ",                    # SpeechBrain INFO: "Fetch hyperparams.yaml: Fetching from HF Hub..."
    "SpeechBrain could not find",      # SpeechBrain WARNING: torchaudio backend
    "Warning: You are sending unauthenticated",  # HuggingFace Hub WARNING
    "HTTP Request:",             # httpx
    "HTTP Response:",            # httpx
)

class _ThirdPartyFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(msg.startswith(p) for p in _NOISE_PREFIXES)

_noise_filter = _ThirdPartyFilter()
_console_handler.addFilter(_noise_filter)
_file_handler.addFilter(_noise_filter)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
log = logging.getLogger("pipeline")
log.info("Log file: %s", _LOG_FILE)

# ── Locate config ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config

# ── Gate registry ──────────────────────────────────────────────────────────────
# Each entry: (gate_key, module_path, env)
#   env = "utmos" → invoked via subprocess with the utmos conda python
#   env = "base"  → imported and called directly in this process
GATE_REGISTRY = [
    ("wer",              "gates/gate_wer.py",              "utmos"),
    ("nisqa",            "gates/gate_nisqa.py",            "base"),
    ("utmos",            "gates/gate_utmos.py",            "utmos"),
    ("speaker_sim",      "gates/gate_speaker_sim.py",      "base"),
    ("ser",              "gates/gate_ser.py",              "base"),
    ("arousal_valence",  "gates/gate_arousal_valence.py",  "base"),
    ("pitch",            "gates/gate_pitch.py",            "base"),
    ("duration",         "gates/gate_duration.py",         "base"),
    ("vad",              "gates/gate_vad.py",              "base"),
    ("amplitude",        "gates/gate_amplitude.py",        "base"),
    ("accent",           "gates/gate_accent.py",           "utmos"),
]

GATE_KEYS = [g[0] for g in GATE_REGISTRY]


def find_conda_python(env_name):
    """
    Locate the python executable for a named conda environment.

    Resolution order:
      1. CONDA_PREFIX env var (set when a conda env is active) — walks up to
         find envs/ sibling directory.
      2. `conda run --no-capture-output -n <env> which python` — works on any
         conda installation regardless of install path.
      3. Hard-coded common macOS paths as a last resort.
    """
    # 1. Derive root from active conda prefix
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        # CONDA_PREFIX is e.g. ~/miniconda3 (base) or ~/miniconda3/envs/utmos
        # Walk up until we find a directory that contains envs/<env_name>
        candidate_root = conda_prefix
        for _ in range(3):
            candidate = os.path.join(candidate_root, "envs", env_name, "bin", "python")
            if os.path.isfile(candidate):
                return candidate
            candidate_root = os.path.dirname(candidate_root)

    # 2. Ask conda itself
    try:
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", env_name, "which", "python"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if os.path.isfile(path):
                return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Common macOS fallback paths
    for base in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/opt/miniconda3"),
        os.path.expanduser("~/opt/anaconda3"),
        "/opt/homebrew/Caskroom/miniconda/base",
        "/opt/miniconda3",
    ]:
        candidate = os.path.join(base, "envs", env_name, "bin", "python")
        if os.path.isfile(candidate):
            return candidate

    return None


def run_utmos_gate(gate_key, gate_script, output_dir):
    """Run a gate inside the utmos conda environment via subprocess."""
    conda_python = find_conda_python(config.UTMOS_CONDA_ENV)
    gate_script_abs = os.path.join(ROOT, gate_script)
    gate_output_dir = os.path.join(output_dir, gate_key)

    if conda_python:
        cmd = [conda_python, gate_script_abs, "--output-dir", gate_output_dir]
        log.info("[%s] subprocess → %s", gate_key, conda_python)
    else:
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", config.UTMOS_CONDA_ENV,
            "python", gate_script_abs, "--output-dir", gate_output_dir,
        ]
        log.info("[%s] subprocess → conda run -n %s", gate_key, config.UTMOS_CONDA_ENV)

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        log.error("[%s] FAILED (exit code %d)", gate_key, result.returncode)
        return False

    log.info("[%s] DONE", gate_key)
    return True


def run_base_gate(gate_key, gate_module_name, output_dir):
    """Import and run a gate directly in this process (base env)."""
    import importlib.util

    gate_script_abs = os.path.join(ROOT, "gates", f"{gate_module_name}.py")
    spec   = importlib.util.spec_from_file_location(gate_module_name, gate_script_abs)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so subclasses of transformers.PreTrainedModel can
    # resolve their module via sys.modules[cls.__module__] (required by newer
    # transformers versions).
    sys.modules[gate_module_name] = module
    spec.loader.exec_module(module)

    gate_output_dir = os.path.join(output_dir, gate_key)

    log.info("")
    log.info("=" * 60)
    log.info("Gate: %s", gate_key.upper())
    log.info("=" * 60)

    try:
        model_state    = module.load_model()
        df, summary_df = module.run_gate(model_state)
        module.print_results(df, summary_df)
        module.save_results(df, summary_df, gate_output_dir)
        log.info("[%s] DONE", gate_key)
        return True
    except Exception as e:
        log.exception("[%s] FAILED: %s", gate_key, e)
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
        help="Gate(s) to run. Default: all gates.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=GATE_KEYS,
        default=[],
        help="Gate(s) to skip.",
    )
    parser.add_argument(
        "--output-dir",
        default=_RUN_DIR,
        help="Output directory for this run. Default: output/runs/<timestamp>/",
    )
    args = parser.parse_args()

    gates_to_run = args.gates if args.gates else GATE_KEYS
    gates_to_run = [g for g in gates_to_run if g not in (args.skip or [])]

    log.info("Pipeline starting.  Run ID: %s", _RUN_ID)
    log.info("Gates     : %s", gates_to_run)
    log.info("Output dir: %s", args.output_dir)
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
    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    for key, status in results_summary.items():
        log.info("  %-14s %s", key, status)

    failed = [k for k, v in results_summary.items() if v == "FAIL"]
    if failed:
        log.error("Failed gates: %s", failed)
        sys.exit(1)
    else:
        log.info("All gates completed successfully.")
        log.info("Results in: %s", args.output_dir)
        log.info("Run ID    : %s", _RUN_ID)

        # Generate visualizations and optional LLM report
        try:
            from generate_report import generate as _generate_report
            _generate_report(args.output_dir)
        except Exception as _e:
            log.warning("Report generation failed (non-fatal): %s", _e)


if __name__ == "__main__":
    main()
