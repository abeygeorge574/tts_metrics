# TTS Evaluation Pipeline

Automated quality-gate pipeline for Hindi-English dubbing TTS evaluation.
Runs 10 objective metrics across any number of TTS models and produces
per-segment and per-model summary CSVs.

---

## Requirements

Two conda environments are required.

### 1. base (Python 3.13) — most gates
```bash
conda env create -f environment_base.yml
```

### 2. utmos (Python 3.9) — WER, UTMOS, accent gates
```bash
conda env create -f environment_utmos.yml
```

> **Always run `run_pipeline.py` from the `base` env.**
> The three utmos-env gates are automatically subprocessed into the `utmos` env.

---

## Folder structure

Every gate expects the same layout inside its own base directory:

```
<gate_folder>/
├── models/
│   ├── model_1/
│   │   ├── sample_01.wav
│   │   └── sample_02.wav
│   └── model_2/
│       ├── sample_01.wav
│       └── sample_02.wav
└── reference/             ← required by most gates (see per-gate notes)
    ├── sample_01.wav
    └── sample_02.wav
```

**Rules:**
- All model folders must contain **identical filenames**.
- Reference filenames must match model filenames exactly.
- All audio must be `.wav`.

### Gate-specific paths (all rooted at `/Users/abey/Documents/tts_metrics/`)

| Gate | Folder | Reference needed? | Notes |
|---|---|---|---|
| WER | `WER_PER_Production/WER_TEST/` | `.txt` per sample in `references/` | Text files, not audio |
| NISQA | `NISQA_prod/` | Optional audio in `reference/` | Delta checks skipped if absent |
| UTMOS | `UTMOS/` | None | Absolute scoring only |
| Speaker Sim | `speaker_similarity/` | Audio in `reference/` and/or `enrollment/speaker.wav` | Falls back to enrollment if per-utterance ref missing |
| SER | `SER/` | Required audio in `reference/` | Hindi reference audio |
| Pitch | `pitch/` | Optional audio in `reference/` | Absolute floor check always runs |
| Duration | `duration_ratio/` | Required audio in `reference/` | |
| VAD | `pause_alignment/` | Required audio in `reference/` | |
| Amplitude | `amplitude/` | Required audio in `reference/` | |
| Accent | `accent_classification/` | `reference/american.wav`, `reference/british.wav`, `reference/indian.wav` | Fixed accent reference clips |

### WER folder layout (different from others)
```
WER_PER_Production/WER_TEST/
├── references/
│   ├── sample_01.txt      ← plain text, one file per sample
│   └── sample_02.txt
└── models/
    ├── model_1/
    │   ├── sample_01.wav
    │   └── sample_02.wav
    └── model_2/
        ├── sample_01.wav
        └── sample_02.wav
```

### Model weights (not included in repo)

| Gate | Weight location |
|---|---|
| NISQA | `NISQA_prod/model/weights/nisqa.tar` |
| UTMOS | `UTMOS/model/simple/epoch=3-step=7459.ckpt` + `wav2vec_small.pt` |

---

## Running the pipeline

```bash
conda activate base

# Run all gates
python run_pipeline.py

# Run specific gates only
python run_pipeline.py --gates wer utmos accent

# Skip a gate
python run_pipeline.py --skip nisqa

# Custom output directory
python run_pipeline.py --output-dir /path/to/results
```

Each gate can also be run standalone:
```bash
python gates/gate_pitch.py --output-dir output/pitch
```

---

## Output

Results are written to `output/<gate>/`:

| File | Contents |
|---|---|
| `per_segment_results.csv` | One row per model × sample with all scores and pass/fail flags |
| `model_summary.csv` | One row per model with aggregated pass rates and key metrics |

---

## Gates summary

| Gate | Env | What it measures | Key threshold(s) |
|---|---|---|---|
| `gate_wer` | utmos | Word Error Rate + Whisper intelligibility | WER ≤ 0.10, Intel ≥ 0.85 |
| `gate_nisqa` | base | NISQA MOS, Noisiness, Discontinuity, Coloration, Loudness | MOS ≥ 3.0, ΔMOS ≥ −0.5 |
| `gate_utmos` | utmos | UTMOS naturalness (1–5 scale) | ≥ 3.0 |
| `gate_speaker_sim` | base | ECAPA-TDNN cosine speaker similarity | ≥ 0.75 |
| `gate_ser` | base | emotion2vec emotion match vs reference | Label must match |
| `gate_pitch` | base | Pitch register + expressiveness (librosa pyin) | Median Δ ≤ 30 Hz, Std ≥ 20 Hz, Std ratio ≥ 0.5× ref |
| `gate_duration` | base | TTS/reference duration ratio | Within ±10% |
| `gate_vad` | base | Pause count, position, duration (Hungarian matching) | Count Δ ≤ 20, Pos ≤ 0.5 s, Dur ratio 0.75–1.25 |
| `gate_amplitude` | base | LUFS, LRA, spectral centroid, true peak | LUFS Δ ≤ 4, LRA Δ ≤ 3, Centroid Δ ≤ 500 Hz, Peak < −1 dBFS |
| `gate_accent` | utmos | wav2vec2 accent proximity (American target) | Proximity ≥ 0.75 |

All thresholds are in `config.py` and can be adjusted without touching gate code.
