# STS Evaluation Pipeline — Setup & Run Guide

Evaluates Speech-To-Speech (voice conversion) quality for Hindi dubbing.
Measures content preservation, speaker identity, naturalness, and more.

---

## 1. Prerequisites

- macOS with Apple Silicon (M1/M2/M3) — MLX Whisper used for WER gate
- OR Linux with CUDA GPU — CUDA Whisper used automatically
- `conda` installed (Miniconda or Anaconda)
- `ffmpeg` on PATH (`brew install ffmpeg` on Mac, `apt install ffmpeg` on Linux)
- Git (to clone the repo)

---

## 2. Clone the repo

```bash
git clone https://github.com/abeygeorge574/tts_metrics.git
cd tts_metrics
git checkout add-sts-pipeline
```

---

## 3. Create environments

### Environment A — `base` (Python 3.13) — runs all gates except WER

```bash
conda create -n base python=3.13 -y
conda activate base
pip install torch torchaudio
pip install speechbrain transformers
pip install parselmouth soundfile librosa pyloudnorm jiwer
pip install pandas numpy scipy
pip install ffmpeg-python
pip install pyannote.audio   # VAD gate
pip install numba
```

### Environment B — `utmos` (Python 3.9) — WER gate only (Whisper)

```bash
conda create -n utmos python=3.9 -y
conda activate utmos
pip install torch torchaudio
pip install openai-whisper jiwer soundfile

# Apple Silicon only — MLX Whisper (10-50x faster than CPU):
pip install mlx-whisper

# Linux/CUDA — openai-whisper with CUDA already works above
```

---

## 4. Download model weights

These are large files — must be downloaded manually. Place them as shown.

### NISQA (naturalness MOS)
```bash
mkdir -p weights/nisqa/weights
# Download nisqa.tar from https://github.com/gabrielmittag/NISQA
# Place at: weights/nisqa/weights/nisqa.tar
```

### Speaker Similarity (ECAPA-TDNN)
```bash
mkdir -p weights/speaker_sim
# SpeechBrain ECAPA-TDNN — will auto-download on first run IF HuggingFace is accessible
# If behind proxy: manually download speechbrain/spkrec-ecapa-voxceleb → weights/speaker_sim/
```

### SER — MERaLiON-SER-v1 (emotion recognition)
```bash
# In Python:
from huggingface_hub import snapshot_download
snapshot_download('MERaLiON/MERaLiON-SER-v1', local_dir='weights/meralion_ser')
```

### Accent (ECAPA, auto-downloads on first run)
No manual step needed — downloads to `~/.cache/huggingface/` automatically.

---

## 5. Configure paths

Edit `sts/config_sts.py`:

```python
ROOT            = "/path/to/tts_metrics"        # absolute path to repo
NISQA_WEIGHT    = f"{ROOT}/weights/nisqa/weights/nisqa.tar"
MERALION_LOCAL_PATH = f"{ROOT}/weights/meralion_ser"
OUTPUT_DIR      = f"{ROOT}/sts_output"
```

Speaker sim weight path in `sts/gates/gate_speaker_sim.py`:
```python
# Change the local_dir to your weights/speaker_sim/ path
```

---

## 6. Episode folder structure

Your episode data must follow this layout:

```
EP21/
├── Karthikeya/
│   ├── Recorded_Segments/     ← dubbing artist WAV files (input)
│   ├── Converted_Segments/    ← STS model output WAVs (same filenames as input)
│   └── Train_Data/
│       └── Karthikeya.wav     ← target speaker training audio
├── Rambo/
│   └── ...
└── ...
```

Filenames in `Recorded_Segments/` and `Converted_Segments/` must match exactly.

---

## 7. Run the pipeline

**Always run from the `base` conda environment:**

```bash
conda activate base
cd /path/to/tts_metrics

# Run all gates on an episode
python sts/run_sts.py --episode-dir /path/to/EP21

# Run specific gates only
python sts/run_sts.py --episode-dir /path/to/EP21 --gates nisqa artifact pitch

# Skip gates
python sts/run_sts.py --episode-dir /path/to/EP21 --skip wer

# Specific characters only
python sts/run_sts.py --episode-dir /path/to/EP21 --characters Karthikeya Rambo

# Custom output directory
python sts/run_sts.py --episode-dir /path/to/EP21 --output-dir /path/to/results
```

### WER gate — must run separately from utmos environment

The WER gate uses Whisper (Python 3.9 + MLX/CUDA). Run it separately:

```bash
conda activate utmos
cd /path/to/tts_metrics
python sts/run_sts.py --episode-dir /path/to/EP21 --gates wer --output-dir sts_output/runs/EP21/wer_results
```

**Important:** Run WER directly from terminal (not via subprocess/notebook) — MLX needs direct Metal GPU access on Apple Silicon.

---

## 8. Output structure

Each run creates:

```
sts_output/runs/EP21/<timestamp>/
├── pipeline.log                        ← full run log
├── episode_summary_<gate>.csv          ← per-gate episode summary
├── Karthikeya/
│   ├── duration/
│   │   ├── per_segment_results.csv    ← one row per WAV file
│   │   └── summary.csv               ← character-level summary
│   ├── pitch/
│   ├── amplitude/
│   └── ...
├── Rambo/
└── ...
```

To build combined episode-level segment CSVs (all characters in one file):
```bash
python sts_output/runs/EP21/build_episode_segments.py
# Creates: sts_output/runs/EP21/episode_segments/<gate>_all_segments.csv
```

---

## 9. Active gates and what they measure

| Gate | What it checks | Environment |
|------|---------------|-------------|
| `wer` | Content preservation — does output say the same words as input? | `utmos` |
| `nisqa` | Naturalness MOS (1–5 scale) | `base` |
| `artifact` | HNR, pause silence floor, spectral artifacts | `base` |
| `speaker_sim` | Is output voice similar to target speaker (train data)? | `base` |
| `ser` | Emotion match between input and output | `base` |
| `pitch` | Pitch register, contour correlation, expressiveness | `base` |
| `duration` | Output duration matches input within ±10% | `base` |
| `vad` | Pause count, position, duration alignment | `base` |
| `amplitude` | Loudness (LUFS, LRA) alignment | `base` |

---

## 10. Key thresholds (edit in `sts/config_sts.py`)

| Gate | Threshold |
|------|-----------|
| WER | ≤ 10% word error rate |
| NISQA | MOS ≥ 3.75 |
| Speaker sim | cosine ≥ 0.50 |
| Pitch register | delta ≤ 30 Hz vs train data |
| Duration | within ±10% of input |
| Amplitude | LUFS ±6.5, LRA ±3.0 |

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'pyloudnorm'`**
→ You're running in utmos env. Switch to base env, or use `--gates wer` in utmos.

**WER: `NSRangeException` crash on Apple Silicon**
→ MLX needs direct terminal access. Don't run WER via Jupyter or subprocess. Run directly from terminal with `conda activate utmos`.

**WER very slow (30+ sec/file)**
→ Kill background processes consuming GPU memory. Check with `ps aux | grep python`.

**NISQA: `numba` cache error**
→ Set `NUMBA_CACHE_DIR` before import (already handled in `run_sts.py`).

**Speaker sim / MERaLiON download blocked by proxy**
→ Download weights on a machine with internet access, transfer via scp/rsync to `weights/`.
