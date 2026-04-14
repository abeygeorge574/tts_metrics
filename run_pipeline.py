"""
run_pipeline.py — main entry point for the TTS evaluation pipeline.

Gates and their environments:
  utmos env (python 3.9) : wer, utmos, accent
  base env  (python 3.13): nisqa, speaker_sim, ser, pitch, duration, vad, amplitude

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
import tempfile
from datetime import datetime

# librosa uses numba JIT which needs a writable cache dir.
# Without this, base env gates (amplitude, pitch, vad) fail with:
#   "cannot cache function '__o_fold': no locator available"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

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

# ── Stdout tee: everything printed to terminal is also written to the log file ──
# This captures both logging.info() calls AND raw print() calls from gate modules,
# giving a complete record of what appeared on the terminal.

_LOG_FILE = os.path.join(_RUN_DIR, "pipeline.log")

class _Tee:
    """
    Wrap a stream so every write goes to both the original stream and a log file.
    Assigned to sys.stdout so that logging (via StreamHandler) and gate print()
    calls all flow through the same path into the log file.
    """
    def __init__(self, stream, path: str):
        self._stream = stream
        self._file   = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        self._file.write(data)
        return n

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def fileno(self) -> int:
        # Expose the real fd so that subprocesses inherit the terminal, not the tee object.
        # Subprocess output is explicitly piped back through sys.stdout in run_utmos_gate().
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()

sys.stdout = _Tee(sys.__stdout__, _LOG_FILE)

# ── Logging ────────────────────────────────────────────────────────────────────
# Single console handler writing to sys.stdout (the tee).
# No separate FileHandler needed — the tee already writes everything to the file.

_formatter = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)

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

logging.basicConfig(level=logging.INFO, handlers=[_console_handler])
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
    ("speaker_sim",      "gates/gate_speaker_sim.py",      "base"),
    ("ser",              "gates/gate_ser.py",              "base"),
    ("pitch",            "gates/gate_pitch.py",            "base"),
    ("duration",         "gates/gate_duration.py",         "base"),
    ("vad",              "gates/gate_vad.py",              "base"),
    ("amplitude",        "gates/gate_amplitude.py",        "base"),
    ("accent",           "gates/gate_accent.py",           "base"),
    ("artifact",         "gates/gate_artifact.py",         "base"),
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


def run_utmos_gate(gate_key, gate_script, output_dir, extra_args=None):
    """
    Run a gate inside the utmos conda environment via subprocess.

    Subprocess stdout+stderr are piped back through sys.stdout (the tee) so
    that gate print() output appears on the terminal AND in the log file,
    matching exactly what a base-env gate produces.
    """
    conda_python = find_conda_python(config.UTMOS_CONDA_ENV)
    gate_script_abs = os.path.join(ROOT, gate_script)
    gate_output_dir = os.path.join(output_dir, gate_key)
    extra_args = extra_args or []

    log.info("")
    log.info("=" * 60)
    log.info("Gate: %s", gate_key.upper())
    log.info("=" * 60)

    if conda_python:
        cmd = [conda_python, gate_script_abs, "--output-dir", gate_output_dir] + extra_args
        log.info("[%s] subprocess → %s", gate_key, conda_python)
    else:
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", config.UTMOS_CONDA_ENV,
            "python", gate_script_abs, "--output-dir", gate_output_dir,
        ] + extra_args
        log.info("[%s] subprocess → conda run -n %s", gate_key, config.UTMOS_CONDA_ENV)

    # Merge stderr into stdout so everything comes through one pipe.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,          # line-buffered for real-time display
    )
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line)   # flows through the tee → terminal + log file
    sys.stdout.flush()
    proc.stdout.close()
    proc.wait()

    if proc.returncode != 0:
        log.error("[%s] FAILED (exit code %d)", gate_key, proc.returncode)
        return False

    log.info("[%s] DONE", gate_key)
    return True


def run_base_gate(gate_key, gate_module_name, output_dir, models_dir=None, ref_dir=None, refs_dir=None):
    """Import and run a gate directly in this process (base env)."""
    import importlib.util

    gate_output_dir = os.path.join(output_dir, gate_key)

    log.info("")
    log.info("=" * 60)
    log.info("Gate: %s", gate_key.upper())
    log.info("=" * 60)

    try:
        gate_script_abs = os.path.join(ROOT, "gates", f"{gate_module_name}.py")
        spec   = importlib.util.spec_from_file_location(gate_module_name, gate_script_abs)
        module = importlib.util.module_from_spec(spec)
        # Register before exec so subclasses of transformers.PreTrainedModel can
        # resolve their module via sys.modules[cls.__module__] (required by newer
        # transformers versions).
        sys.modules[gate_module_name] = module
        spec.loader.exec_module(module)

        model_state = module.load_model()
        if model_state is None:
            model_state = {}
        if models_dir:
            model_state["models_dir"] = models_dir
        if ref_dir:
            model_state["ref_dir"] = ref_dir
        if refs_dir:
            model_state["refs_dir"] = refs_dir
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
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Override MODELS_DIR from config (path to folder containing model sub-folders).",
    )
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Override REFERENCE_DIR from config (path to folder with reference .wav files).",
    )
    parser.add_argument(
        "--refs-dir",
        default=None,
        help="Override TEXT_REFERENCE_DIR from config (WER gate: folder with .txt transcripts).",
    )
    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir) if args.models_dir else None
    ref_dir    = os.path.abspath(args.ref_dir)    if args.ref_dir    else None
    refs_dir   = os.path.abspath(args.refs_dir)   if args.refs_dir   else None

    gates_to_run = args.gates if args.gates else GATE_KEYS
    gates_to_run = [g for g in gates_to_run if g not in (args.skip or [])]

    log.info("Pipeline starting.  Run ID: %s", _RUN_ID)
    log.info("Gates     : %s", gates_to_run)
    log.info("Output dir: %s", args.output_dir)
    if models_dir:
        log.info("Models dir: %s", models_dir)
    if ref_dir:
        log.info("Ref dir   : %s", ref_dir)
    if refs_dir:
        log.info("Refs dir  : %s (WER text transcripts)", refs_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build extra CLI args for utmos subprocess gates
    extra_args = []
    if models_dir:
        extra_args += ["--models-dir", models_dir]
    if ref_dir:
        extra_args += ["--ref-dir", ref_dir]
    if refs_dir:
        extra_args += ["--refs-dir", refs_dir]

    results_summary = {}

    for gate_key, gate_script, env in GATE_REGISTRY:
        if gate_key not in gates_to_run:
            continue

        gate_module = os.path.splitext(os.path.basename(gate_script))[0]

        if env == "utmos":
            success = run_utmos_gate(gate_key, gate_script, args.output_dir, extra_args=extra_args)
        else:
            success = run_base_gate(gate_key, gate_module, args.output_dir,
                                    models_dir=models_dir, ref_dir=ref_dir, refs_dir=refs_dir)

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
