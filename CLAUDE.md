# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

A TTS/STS (Text-To-Speech / Speech-To-Speech) audio quality evaluation pipeline for Hindi-English dubbing. It runs a set of "gates" — modular quality checks — against synthesized speech to determine production readiness. Each gate evaluates a different dimension: intelligibility, naturalness, speaker similarity, pitch, duration, pause alignment, loudness, emotion, emotional intensity, and accent.

## Running the pipeline

```bash
# Run all gates
python run_pipeline.py

# Run specific gates only
python run_pipeline.py --gates wer utmos nisqa

# Skip specific gates
python run_pipeline.py --skip nisqa pitch

# Custom output directory
python run_pipeline.py --output-dir /path/to/results
```

Valid gate names: `wer`, `nisqa`, `utmos`, `speaker_sim`, `ser`, `arousal_valence`, `pitch`, `duration`, `vad`, `amplitude`, `accent`, `artifact`

## Environment setup

Two conda environments are required:
- **`utmos`** (Python 3.9): used for `wer`, `utmos`, `accent` gates — these run as subprocesses
- **`base`** (Python 3.13): used for all other gates — these run as direct imports in the current process

The pipeline auto-detects the `utmos` conda python at common paths (`~/miniconda3/envs/utmos/bin/python`, etc.) and falls back to `conda run -n utmos`. Run `run_pipeline.py` from the base (Python 3.13) environment.

`ffmpeg` must be installed and on PATH (required by the VAD gate for silence detection).

## Architecture

### Two-environment design

Gates that depend on older ML models (Whisper, UTMOS scorer, wav2vec2 for accent) require Python 3.9 and are invoked as subprocesses. All other gates are imported and called directly. The `GATE_REGISTRY` in `run_pipeline.py` tracks which environment each gate belongs to.

### Gate interface contract

Every gate module in `gates/` must expose four functions:
```python
model_state = load_model()
df, summary_df = run_gate(model_state)
print_results(df, summary_df)
save_results(df, summary_df, output_dir)
```

`run_gate()` accepts an optional `model_state` parameter so models can be injected without reloading (useful for testing individual gates).

### Data layout expected by gates

All gates share a single unified data directory:

```
data/
├── models/
│   ├── model_name1/
│   │   ├── sample1.wav
│   │   └── sample2.wav
│   └── model_name2/
│       └── ...
├── reference/          # reference audio (optional for some gates)
│   ├── sample1.wav
│   └── sample2.wav
├── enrollment/         # optional, for speaker_sim gate
│   └── speaker.wav
├── text_references/    # WER gate — one .txt per sample, filename must match .wav
│   ├── sample1.txt
│   └── sample2.txt
└── accent_reference/   # fixed accent reference clips (not per-segment)
    ├── american.wav
    ├── british.wav
    └── indian.wav
```

Base directories are set in `config.py` (all rooted at `ROOT = /Users/abey/Documents/tts_metrics`).

### Output structure

Each run creates a timestamped folder: `output/runs/YYYY-MM-DD_HH-MM-SS/`

Inside, each gate writes two CSVs to `<run_dir>/<gate_key>/`:
- `per_segment_results.csv` — per audio file metrics + pass/fail
- `model_summary.csv` — aggregated pass rate per TTS/STS model

A `pipeline.log` captures full terminal output. A Gemini LLM report and visualizations (radar chart + heatmap) are generated at the end.

### Pass/fail logic

- Most gates compare TTS vs reference using delta thresholds
- When no reference is available (or reference quality is degraded), absolute thresholds are used as fallback
- NISQA uses a hybrid: PASS if absolute threshold met OR (absolute met AND delta met)
- SER uses three-way: PASS / NEAR_MISS / FAIL — near-miss when top-2 label distributions are close on either side

### Language limitations

This pipeline targets **English TTS/STS output** evaluated against Hindi reference audio.

- **Unreliable for non-English output:** WER, UTMOS, Accent
- **Partially reliable:** SER (LOW_CONF_REF degraded bucket handles uncertain Hindi references), Arousal/Valence (no confidence signal — treat as soft)
- **Fully reliable regardless of language:** Pitch, Duration, VAD, Amplitude, Speaker Sim, NISQA (relative comparisons valid)

## Configuration

All thresholds and paths live in `config.py`. Key thresholds:

| Gate | Key threshold |
|------|--------------|
| WER | max 10% word error rate |
| NISQA | MOS ≥ 3.0, delta ≥ −0.5 |
| UTMOS | score ≥ 3.0 |
| Speaker similarity | cosine ≥ 0.75 |
| SER | confidence ≥ 0.5, near-miss margin 0.10 |
| Arousal/Valence | delta ≤ 0.15 on both axes |
| Duration | within ±10% of reference |
| Pitch | median delta ≤ 30 Hz, std ≥ 0.5× reference |
| Amplitude | LUFS ±4, LRA ±3, spectral centroid ±500 Hz |
| Accent | "american" class proximity ≥ 0.75 |

Model weight paths (gitignored, must be present locally):
- NISQA: `weights/nisqa/weights/nisqa.tar`
- UTMOS: `weights/utmos/simple/epoch=3-step=7459.ckpt`

## Adding a new gate

1. Create `gates/gate_<name>.py` implementing the four-function interface above
2. Add an entry to `GATE_REGISTRY` in `run_pipeline.py` with the env (`"base"` or `"utmos"`)
3. Add base directory, thresholds, and any weight paths to `config.py`
