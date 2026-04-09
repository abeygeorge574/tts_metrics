# TTS Evaluation Pipeline

Automated quality-gate pipeline for Hindi-English dubbing TTS/STS evaluation.
Runs 11 objective metrics across any number of TTS models and produces
per-segment and per-model summary CSVs, plus a visual HTML-style report
(radar chart + heatmap) and an optional Gemini LLM quality summary.

---

## Requirements

Two conda environments are required.

### 1. base (Python 3.13) — most gates + report generation
```bash
conda env create -f environment_base.yml
```

### 2. utmos (Python 3.9) — WER, UTMOS, accent gates
```bash
conda env create -f environment_utmos.yml
```

> **Always run `run_pipeline.py` from the `base` env.**
> The three utmos-env gates are automatically subprocessed into the `utmos` env.

### Optional: Gemini LLM report
```bash
pip install google-genai        # in base env
export GEMINI_API_KEY="your_key_here"   # or add to ~/.zshrc / ~/.bashrc
```
Set the environment variable before running the pipeline. The LLM report
is silently skipped if the key is not set.

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

All audio gates share the unified `data/` folder. The only gate with a separate layout is WER (text references) and Accent (fixed reference clips in `data/accent_reference/`).

| Gate | Audio source | Reference needed? | Notes |
|---|---|---|---|
| WER | `data/models/` | `.txt` per sample in `data/text_references/` | Text files, not audio |
| NISQA | `data/models/` | Optional audio in `data/reference/` | Delta checks skipped if absent |
| UTMOS | `data/models/` | None | Absolute scoring only |
| Speaker Sim | `data/models/` | Audio in `data/reference/` and/or `data/enrollment/speaker.wav` | Falls back to enrollment if per-utterance ref missing |
| SER | `data/models/` | Required audio in `data/reference/` | Hindi reference audio |
| Arousal/Valence | `data/models/` | Optional audio in `data/reference/` | Dimensional emotion (arousal + valence) |
| Pitch | `data/models/` | Optional audio in `data/reference/` | Absolute floor check always runs |
| Duration | `data/models/` | Required audio in `data/reference/` | |
| VAD | `data/models/` | Required audio in `data/reference/` | |
| Amplitude | `data/models/` | Required audio in `data/reference/` | |
| Accent | `data/models/` | Fixed clips in `data/accent_reference/` (american/british/indian) | Not per-segment — one clip per accent class |

### WER text references

Place one `.txt` file per sample in `data/text_references/`. The filename must match the `.wav` filename (e.g. `sample_01.txt` ↔ `sample_01.wav`). The WER gate reads audio from the same `data/models/` folder as all other gates.

### Model weights (not included in repo)

| Gate | Weight location |
|---|---|
| NISQA | `weights/nisqa/weights/nisqa.tar` |
| UTMOS | `weights/utmos/simple/epoch=3-step=7459.ckpt` + `weights/utmos/simple/wav2vec_small.pt` |

#### Download commands

**NISQA** — clone the repo directly into the expected path:
```bash
git clone https://github.com/gabrielmittag/NISQA.git weights/nisqa
```
The weights file `weights/nisqa.tar` ships with the repo.

**UTMOS** — download checkpoint and wav2vec backbone:
```bash
mkdir -p weights/utmos/simple

# UTMOS checkpoint (from the UTMOS GitHub release)
curl -L "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt" \
     -o weights/utmos/simple/epoch=3-step=7459.ckpt

# wav2vec 2.0 small backbone (from Facebook Research)
curl -L "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt" \
     -o weights/utmos/simple/wav2vec_small.pt
```

Also clone the UTMOS inference code:
```bash
git clone https://github.com/sarulab-speech/UTMOS22.git weights/utmos
```

### Enrollment file (Speaker Similarity gate)

The speaker similarity gate supports two reference modes:

1. **Per-utterance reference** — place matching `.wav` files in `speaker_similarity/reference/` (one per sample, same filename as the model output). Best for measuring identity per line.
2. **Enrollment** — place a single reference clip at `speaker_similarity/enrollment/speaker.wav`. Used as fallback when a per-utterance reference is missing, or as the sole reference if no `reference/` folder exists.

```
speaker_similarity/
├── enrollment/
│   └── speaker.wav        ← single reference clip for the target speaker
├── reference/             ← optional: per-utterance ground-truth audio
│   ├── sample_01.wav
│   └── sample_02.wav
└── models/
    ├── model_1/
    └── model_2/
```

At least one of `enrollment/speaker.wav` or `reference/` must be present; if neither exists the gate skips all segments.

---

## Hardware / device support

All gates that use a neural model perform automatic device detection at runtime:

```
CUDA (NVIDIA GPU)  →  MPS (Apple Silicon)  →  CPU
```

| Gate | GPU accelerated? | Notes |
|---|---|---|
| WER | Yes (Whisper) | MLX on Apple Silicon via `whisper-mlx`, CUDA via `openai-whisper`, CPU fallback |
| NISQA | CPU only | PyTorch inference is fast enough on CPU |
| UTMOS | Yes | CUDA or MPS picked up automatically |
| Speaker Sim | Yes | ECAPA-TDNN moves to detected device |
| SER | CPU only | emotion2vec runs on CPU |
| Arousal/Valence | Yes | wav2vec2-large-robust on detected device |
| Pitch | CPU only | librosa/pyin — no GPU path |
| Duration | CPU only | header read only |
| VAD | CPU only | ffmpeg + scipy |
| Amplitude | CPU only | pyloudnorm + librosa |
| Accent | Yes | wav2vec2-large-xlsr-53 moves to detected device |

No manual configuration is needed — the gates detect the best available device automatically.

### One-time setup: PyTorch checkpoint compatibility

UTMOS uses a `lightning_fabric` checkpoint. On PyTorch ≥ 2.x the default `weights_only=True` in `torch.load` breaks loading. The gate patches `torch.load` automatically at runtime — **no manual sed edits required**.

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

# Custom output directory (disables auto-timestamping)
python run_pipeline.py --output-dir /path/to/results
```

Each gate can also be run standalone:
```bash
python gates/gate_pitch.py --output-dir output/pitch
```

Report generation can also be triggered standalone on any completed run:
```bash
python generate_report.py --run-dir output/runs/2026-04-03_10-00-00/
```

---

## Output

Each run creates a timestamped folder under `output/runs/`:

```
output/
└── runs/
    └── 2026-04-03_10-00-00/      ← one folder per run
        ├── pipeline.log          ← full run log
        ├── radar.png             ← per-model gate pass-rate radar chart
        ├── heatmap.png           ← segment × gate PASS/FAIL heatmap
        ├── llm_report.txt        ← Gemini quality summary (if API key set)
        ├── wer/
        │   ├── per_segment_results.csv
        │   └── model_summary.csv
        ├── nisqa/
        ├── utmos/
        ├── speaker_sim/
        ├── ser/
        ├── arousal_valence/
        ├── pitch/
        ├── duration/
        ├── vad/
        ├── amplitude/
        └── accent/
```

Each gate folder contains:

| File | Contents |
|---|---|
| `per_segment_results.csv` | One row per model × sample with all scores and PASS/NEAR_MISS/FAIL flags |
| `model_summary.csv` | One row per model with aggregated pass rates and key metrics |

---

## Gates summary

| Gate | Env | What it measures | Key threshold(s) |
|---|---|---|---|
| `gate_wer` | utmos | Word Error Rate + Whisper intelligibility | WER ≤ 0.10, Intel ≥ 0.85 |
| `gate_nisqa` | base | NISQA MOS, Noisiness, Discontinuity, Coloration, Loudness | MOS ≥ 3.0, ΔMOS ≥ −0.5 |
| `gate_utmos` | utmos | UTMOS naturalness (1–5 scale) | ≥ 3.0 |
| `gate_speaker_sim` | base | ECAPA-TDNN cosine speaker similarity | ≥ 0.75 |
| `gate_ser` | base | emotion2vec emotion match vs reference (see near-miss below) | Label must match; near-miss within 10% confidence gap |
| `gate_arousal_valence` | base | Dimensional emotion: arousal + valence delta vs reference | Δ arousal ≤ 0.15, Δ valence ≤ 0.15 |
| `gate_pitch` | base | Pitch register + expressiveness (librosa pyin, 16 kHz) | Median Δ ≤ 30 Hz, Std ≥ 20 Hz, Std ratio ≥ 0.5× ref |
| `gate_duration` | base | TTS/reference duration ratio | Within ±10% |
| `gate_vad` | base | Pause count, position, duration (Hungarian matching) | Count Δ ≤ 20, Pos ≤ 0.2 s, Dur ratio 0.75–1.25 |
| `gate_amplitude` | base | LUFS, LRA, spectral centroid, true peak | LUFS Δ ≤ 4, LRA Δ ≤ 3, Centroid Δ ≤ 500 Hz, Peak < −1 dBFS |
| `gate_accent` | utmos | wav2vec2 accent proximity (American target) | Proximity ≥ 0.75 |
| `gate_artifact` | base | HNR tonal buzz (parselmouth) + median dBFS of real pause frames | HNR ≥ 8 dB, pause median ≤ −55 dBFS |

All thresholds are in `config.py` and can be adjusted without touching gate code.

### WER model summary ranking

The `model_summary.csv` for the WER gate ranks models by the following tiebreaker chain (dubbing priority order):

| Priority | Column | Direction | Reasoning |
|---|---|---|---|
| 1 | Both Pass Rate | Higher = better | Must pass both WER and Intelligibility |
| 2 | Total Deletions | Lower = better | Dropped words = character skips script lines — hardest to catch in post |
| 3 | Total Substitutions | Lower = better | Wrong words said = character says something off-script |
| 4 | Median WER | Lower = better | Typical quality — consistent degradation across many lines is worse than one bad segment |
| 5 | Max WER | Lower = better | Worst single segment — one outlier failure |
| 6 | Total Insertions | Lower = better | Extra words added — less critical than deletions |
| 7 | Median LogProb | Closer to 0 = better | Whisper confidence — higher (less negative) means Whisper is more certain about its transcription |

**Median LogProb interpretation**: Whisper assigns a log-probability to each word. The segment mean is reported. Clean, natural TTS clusters around −0.01 to −0.03. Values below −0.5 indicate Whisper is guessing. speecht5_griffinlim at −1.58 means Whisper has almost no confidence — it is transcribing noise.

**Intel_Pass_Rate** is the per-segment fraction of words with log-prob above `MUMBLE_THRESHOLD` (default −1.0). A segment passes intelligibility if `Intel_Pass_Rate ≥ INTEL_THRESHOLD` (default 0.85) — i.e. at least 85% of words are high-confidence.

---

### SER near-miss detection

The SER gate uses a three-way classification for each segment:

| Status | Meaning |
|---|---|
| `PASS` | TTS emotion label matches reference label |
| `NEAR_MISS` | Labels differ, but the model was uncertain: either the reference label appeared as the TTS runner-up within 10% of the winning confidence, or the reference distribution itself was close (top-1 vs top-2 within 10%) |
| `FAIL` | Labels differ with a clear confidence gap — unambiguous mismatch |

The model summary shows `Clean Pass Rate`, `Clean Near Miss`, and `Clean Fail Rate` separately.
The near-miss margin is controlled by `SER_NEAR_MISS_MARGIN = 0.10` in `config.py`.

### Short segment handling

Segments shorter than `MIN_SEGMENT_DURATION` (default 2.0 s) in `config.py` are:
- Still scored normally — the metric value is real
- Flagged as `SHORT_SEGMENT` in the `Flag` column
- Counted as degraded in the pass-rate calculation

This is relevant for fast-switching dialogue lines.

### Long segment chunking

Gates that run neural models with memory or training-distribution constraints
automatically split long audio into equal-length chunks and aggregate:

| Gate | Max chunk | Aggregation |
|---|---|---|
| NISQA | 15 s | Mean of all 5 NISQA metrics across chunks |
| SER | 15 s | Majority vote on label; mean confidence of winning chunks |
| UTMOS | 10 s | Mean UTMOS score across chunks |
| Arousal/Valence | 15 s | Mean arousal/valence across chunks |
| Accent | 30 s | Mean cosine similarity per chunk against reference embedding |

---

## Configuration reference (`config.py`)

| Key | Default | Description |
|---|---|---|
| `WER_THRESHOLD` | 0.10 | Max acceptable word error rate |
| `INTEL_THRESHOLD` | 0.85 | Min fraction of high-confidence words |
| `NISQA_THRESHOLDS["MOS"]` | 3.0 | Min absolute NISQA MOS |
| `NISQA_DELTA_THRESHOLDS["MOS"]` | −0.5 | Max MOS drop vs reference |
| `UTMOS_THRESHOLD` | 3.0 | Min UTMOS score |
| `SPEAKER_SIM_THRESHOLD` | 0.75 | Min cosine similarity to reference speaker |
| `SER_CONFIDENCE_THRESHOLD` | 0.5 | Min reference confidence to use as ground truth |
| `SER_NEAR_MISS_MARGIN` | 0.10 | Confidence gap within which a FAIL becomes NEAR_MISS |
| `AROUSAL_DELTA_THRESHOLD` | 0.15 | Max arousal delta vs reference |
| `VALENCE_DELTA_THRESHOLD` | 0.15 | Max valence delta vs reference |
| `PITCH_MEDIAN_THRESHOLD` | 30.0 Hz | Max pitch median delta |
| `PITCH_STD_ABS_THRESHOLD` | 20.0 Hz | Min TTS pitch std (expressiveness floor) |
| `PITCH_STD_RATIO_THRESHOLD` | 0.5 | TTS std must be ≥ 0.5× reference std |
| `DURATION_TOLERANCE` | 0.10 | ±10% duration ratio tolerance |
| `LUFS_TOLERANCE` | 6.5 | Max LUFS delta (±6 dB passes, ±9 dB fails — perceptually calibrated) |
| `LRA_TOLERANCE` | 3.0 | Max LRA delta |
| `CENTROID_TOLERANCE` | 500 Hz | Max spectral centroid delta |
| `ACCENT_TARGET_THRESHOLD` | 0.75 | Min American accent proximity |
| `MIN_SEGMENT_DURATION` | 2.0 s | Segments below this are flagged SHORT_SEGMENT |
| `ARTIFACT_HNR_ABS_THRESHOLD` | 8.0 dB | HNR below this → ERR_VOICE_BUZZ (tonal/metallic buzz) |
| `ARTIFACT_SILENCE_DETECT_DB` | −30.0 dBFS | Frame energy threshold for pause detection (separate from VAD gate's −40 dB) |
| `ARTIFACT_MIN_PAUSE_FRAMES` | 10 frames | Minimum contiguous frames to count as a real pause (≥ 160 ms) |
| `ARTIFACT_SILENCE_MEDIAN_DB` | −55.0 dBFS | Pause median above this → ERR_BACKGROUND_STATIC (broadband noise floor) |

---

## Language limitations

This pipeline was built for **English TTS/STS output** evaluated against reference audio. Gates fall into three categories depending on whether the audio being tested is English or not.

### Fully reliable for any language (signal processing only)

| Gate | Why |
|---|---|
| Pitch | librosa pyin operates on the raw audio signal — no language assumption |
| Duration | reads file header only |
| VAD | amplitude-based silence detection via ffmpeg |
| Amplitude | pyloudnorm + librosa spectral analysis — purely acoustic |

### Unreliable for non-English audio

| Gate | Problem | Detail |
|---|---|---|
| WER | English only | Whisper transcribes into whatever language it detects; text references are English. WER will be near 100% on Hindi audio. |
| UTMOS | English only | Trained on English TTS naturalness ratings (LJSpeech-style MOS). Scores on non-English speech have no calibrated meaning. |
| Accent | English only | Classifies English accent types (American / British / Indian-English). Hindi speech produces arbitrary similarity scores — results are meaningless. |
| Arousal/Valence | English-biased | Fine-tuned on MSP-IMPROV and MSP-Podcast, which are English-only. Dimensional emotion predictions degrade noticeably on non-English prosody. |

### Partially reliable for non-English audio

| Gate | Reliability | Detail |
|---|---|---|
| NISQA | Mostly reliable | Acoustic dimensions (noisiness, discontinuity, coloration, loudness) are language-agnostic. Absolute MOS scores are calibrated on English data so exact values may be slightly biased, but relative comparisons between models remain valid. **When using a Hindi reference: NISQA on Hindi reference gives useful relative comparisons, not calibrated absolute scores.** Noisiness, Discontinuity, and Loudness deltas are valid cross-lingually. MOS and Coloration deltas carry a baseline offset from the language difference itself — a negative ΔMOS partly reflects language mismatch, not just quality difference. Treat MOS/Coloration deltas as directional signals only. |
| SER | Moderate | emotion2vec_plus_large was trained on multilingual data and handles several languages. Hindi is not a primary training language — emotion label accuracy will be lower than on English, but coarse emotion groupings (happy vs sad vs angry) remain usable. Treat NEAR_MISS results with extra skepticism. |
| Speaker Sim | Reliable | ECAPA-TDNN on VoxCeleb captures voice identity independent of language content. Speaker similarity scores are valid across languages as long as the same speaker is compared. |

### Summary table

| Gate | English output | Hindi output | Cross-lingual ref→output |
|---|---|---|---|
| WER | ✅ | ❌ | ❌ |
| NISQA | ✅ | ✅ (scores slightly biased) | ✅ |
| UTMOS | ✅ | ❌ | — |
| Speaker Sim | ✅ | ✅ | ✅ |
| SER | ✅ | ⚠️ | ⚠️ |
| Arousal/Valence | ✅ | ⚠️ | ⚠️ |
| Pitch | ✅ | ✅ | ✅ |
| Duration | ✅ | ✅ | ✅ |
| VAD | ✅ | ✅ | ✅ |
| Amplitude | ✅ | ✅ | ✅ |
| Accent | ✅ (English accent check) | ❌ | ❌ |

✅ reliable  ⚠️ use with caution  ❌ do not use
