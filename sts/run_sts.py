"""
STS Evaluation Pipeline
Evaluates one episode at a time. Loops over characters within the episode.

Episode folder structure expected:
  EP21/
  ├── Karthikeya/
  │   ├── Recorded_Segments/    ← input (dubbing artist)
  │   ├── Converted_Segments/   ← output (STS model, same filenames as input)
  │   └── Train_Data/
  │       └── Karthikeya.wav    ← target voice training clip
  ├── Rambo/
  │   └── ...
  └── ...

Usage:
  python run_sts.py --episode-dir /Users/abey/Downloads/EP21
  python run_sts.py --episode-dir /Users/abey/Downloads/EP21 --gates nisqa artifact pitch
  python run_sts.py --episode-dir /Users/abey/Downloads/EP21 --skip wer
"""

from __future__ import annotations

import os
import sys

# Python 3.13 + numba 0.64 bug: "no locator available" when numba tries to cache
# compiled functions from site-packages. Fix: point numba to a writable temp dir.
# Must be set before any librosa import.
import tempfile as _tempfile
os.environ["NUMBA_CACHE_DIR"] = _tempfile.mkdtemp(prefix="numba_sts_")
import argparse
import logging
import datetime
import traceback
import glob

import pandas as pd

_STS_DIR       = os.path.dirname(os.path.abspath(__file__))
_STS_GATES_DIR = os.path.join(_STS_DIR, "gates")
_TTS_ROOT      = os.path.dirname(_STS_DIR)

for _p in [_TTS_ROOT, _STS_DIR, _STS_GATES_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config_sts as config

# Gate imports (using sys.path that includes sts/gates/ directly)
import gate_duration, gate_amplitude, gate_nisqa, gate_artifact
import gate_ser, gate_pitch, gate_speaker_sim, gate_vad, gate_wer

# ── Gate registry ──────────────────────────────────────────────────────────────
# Gates that carry loaded models across characters (to avoid reloading per character)
_STATEFUL_GATES = {"ser", "speaker_sim", "nisqa"}

ALL_GATES = ["wer", "nisqa", "artifact", "duration", "amplitude",
             "ser", "vad", "pitch", "speaker_sim"]


# ── Character discovery ────────────────────────────────────────────────────────
def discover_characters(episode_dir: str) -> list[dict]:
    """
    Find all character directories with the expected subfolder structure.
    Returns list of dicts: {name, input_dir, output_dir, train_file}
    """
    characters = []
    for entry in sorted(os.listdir(episode_dir)):
        char_dir = os.path.join(episode_dir, entry)
        if not os.path.isdir(char_dir):
            continue

        input_dir  = os.path.join(char_dir, "Recorded_Segments")
        output_dir = os.path.join(char_dir, "Converted_Segments")

        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            continue  # not a character folder

        # Find train file
        train_wavs = sorted(glob.glob(os.path.join(char_dir, "Train_Data", "*.wav")))
        train_file = train_wavs[0] if train_wavs else None

        in_count  = len([f for f in os.listdir(input_dir)  if f.endswith(".wav")])
        out_count = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])

        print(f"  Character: {entry} | input={in_count} | output={out_count} | train={'yes' if train_file else 'MISSING'}")
        characters.append({
            "name"      : entry,
            "input_dir" : input_dir,
            "output_dir": output_dir,
            "train_file": train_file,
        })

    return characters


# ── Per-character gate runner ──────────────────────────────────────────────────
def run_character_gates(char: dict, run_dir: str, gates_to_run: list,
                        model_cache: dict, log: logging.Logger) -> dict:
    """
    Runs all gates for one character. Returns per-gate summary dicts.
    model_cache: shared dict for stateful models (SER, speaker_sim, NISQA weight)
    """
    name      = char["name"]
    input_dir = char["input_dir"]
    out_dir   = char["output_dir"]
    train_f   = char["train_file"]

    char_out  = os.path.join(run_dir, name)
    summaries = {}

    for gate_name in gates_to_run:
        gate_out = os.path.join(char_out, gate_name)
        log.info(f"  [{name}] Running gate: {gate_name}")

        try:
            if gate_name == "duration":
                df, summ = gate_duration.run_gate(input_dir, out_dir, train_f, name)
                gate_duration.save_results(df, summ, gate_out)

            elif gate_name == "amplitude":
                df, summ = gate_amplitude.run_gate(input_dir, out_dir, train_f, name)
                gate_amplitude.save_results(df, summ, gate_out)

            elif gate_name == "nisqa":
                w = model_cache.get("nisqa_weight", config.NISQA_WEIGHT)
                df, summ = gate_nisqa.run_gate(input_dir, out_dir, train_f, name, nisqa_weight=w)
                gate_nisqa.save_results(df, summ, gate_out)

            elif gate_name == "artifact":
                df, summ = gate_artifact.run_gate(input_dir, out_dir, train_f, name)
                gate_artifact.save_results(df, summ, gate_out)

            elif gate_name == "ser":
                ms = model_cache.get("ser_model_state")
                df, summ, ms = gate_ser.run_gate(input_dir, out_dir, train_f, name, model_state=ms)
                model_cache["ser_model_state"] = ms
                gate_ser.save_results(df, summ, gate_out)

            elif gate_name == "vad":
                df, summ = gate_vad.run_gate(input_dir, out_dir, train_f, name)
                gate_vad.save_results(df, summ, gate_out)

            elif gate_name == "pitch":
                df, summ = gate_pitch.run_gate(input_dir, out_dir, train_f, name)
                gate_pitch.save_results(df, summ, gate_out)

            elif gate_name == "speaker_sim":
                ms = model_cache.get("spkr_model_state")
                df, summ, ms = gate_speaker_sim.run_gate(input_dir, out_dir, train_f, name, model_state=ms)
                model_cache["spkr_model_state"] = ms
                gate_speaker_sim.save_results(df, summ, gate_out)

            elif gate_name == "wer":
                df, summ = gate_wer.run_gate(input_dir, out_dir, train_f, name)
                gate_wer.save_results(df, summ, gate_out)

            summaries[gate_name] = summ
            log.info(f"  [{name}] {gate_name} ✓")

        except Exception as e:
            log.error(f"  [{name}] {gate_name} FAILED: {e}")
            log.debug(traceback.format_exc())
            summaries[gate_name] = None

    return summaries


# ── Episode-level summary ──────────────────────────────────────────────────────
def build_episode_summary(all_summaries: dict, gates_to_run: list, run_dir: str) -> None:
    """
    Aggregate per-character summaries into an episode-level CSV per gate.
    """
    for gate_name in gates_to_run:
        gate_summs = []
        for char_name, char_summs in all_summaries.items():
            if char_summs and gate_name in char_summs and char_summs[gate_name] is not None:
                gate_summs.append(char_summs[gate_name])

        if not gate_summs:
            continue

        try:
            ep_df = pd.concat(gate_summs, ignore_index=True)
            out_path = os.path.join(run_dir, f"episode_summary_{gate_name}.csv")
            ep_df.to_csv(out_path, index=False)
            print(f"Episode summary [{gate_name}] → {out_path}")
        except Exception as e:
            print(f"Could not build episode summary for {gate_name}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="STS Evaluation Pipeline")
    parser.add_argument("--episode-dir", required=True,
                        help="Path to episode folder (e.g. /Users/abey/Downloads/EP21)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: sts_output/runs/<episode>/<timestamp>)")
    parser.add_argument("--gates", nargs="+", default=None,
                        help=f"Gates to run (default: all). Choices: {ALL_GATES}")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Gates to skip")
    parser.add_argument("--characters", nargs="+", default=None,
                        help="Only run for specific characters (default: all)")
    args = parser.parse_args()

    episode_dir  = os.path.abspath(args.episode_dir)
    episode_name = os.path.basename(episode_dir)

    if not os.path.isdir(episode_dir):
        print(f"ERROR: episode-dir not found: {episode_dir}")
        sys.exit(1)

    # Output directory
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir or os.path.join(config.OUTPUT_DIR, "runs", episode_name, ts)
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(run_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("sts_pipeline")
    log.info(f"Episode   : {episode_name}")
    log.info(f"Input dir : {episode_dir}")
    log.info(f"Output dir: {run_dir}")

    # Gate selection
    gates_to_run = args.gates if args.gates else ALL_GATES
    gates_to_run = [g for g in gates_to_run if g not in args.skip]
    log.info(f"Gates     : {gates_to_run}")

    # Discover characters
    log.info("Discovering characters ...")
    characters = discover_characters(episode_dir)
    if not characters:
        log.error(f"No character folders found in {episode_dir}")
        sys.exit(1)

    if args.characters:
        characters = [c for c in characters if c["name"] in args.characters]
        log.info(f"Filtered to: {[c['name'] for c in characters]}")

    log.info(f"Characters: {[c['name'] for c in characters]}")

    # Model cache (shared across characters — avoids reloading per character)
    model_cache = {}

    # Pre-load NISQA weight path
    if "nisqa" in gates_to_run:
        model_cache["nisqa_weight"] = config.NISQA_WEIGHT

    # Run gates for each character
    all_summaries = {}
    for char in characters:
        log.info(f"\n{'='*60}")
        log.info(f"Character: {char['name']} ({len([f for f in os.listdir(char['input_dir']) if f.endswith('.wav')])} segments)")
        log.info(f"{'='*60}")
        all_summaries[char["name"]] = run_character_gates(
            char, run_dir, gates_to_run, model_cache, log
        )

    # Episode-level summary
    log.info("\nBuilding episode summaries ...")
    build_episode_summary(all_summaries, gates_to_run, run_dir)

    log.info(f"\nAll done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
