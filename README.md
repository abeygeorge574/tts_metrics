# TTS Evaluation Pipeline

Automated quality-gate pipeline for Hindi-English dubbing TTS/STS evaluation.
Runs 12 objective metrics across any number of TTS models and produces
per-segment and per-model summary CSVs, plus a visual report
(radar chart + heatmap) and an optional Gemini LLM quality summary.

---

## Quick start

```bash
# 1. Create environments
conda env create -f environment_base.yml   # Python 3.13 — most gates
conda env create -f environment_utmos.yml  # Python 3.9  — WER, UTMOS

# 2. Download model weights (speaker-sim ECAPA ~87 MB; skip chatterbox if not doing voice cloning)
conda activate base
python download_weights.py --skip-chatterbox

# 3. Run all gates
python run_pipeline.py
```

> Always run `run_pipeline.py` from the **base** env.
> WER and UTMOS gates are automatically subprocessed into the `utmos` env.

### Optional: Gemini LLM report
```bash
export GEMINI_API_KEY="your_key_here"
```
Silently skipped if key is not set.

---

## Data layout

All gates share the same `data/` folder:

```
data/
├── models/                    ← TTS/STS outputs — one subfolder per model
│   ├── model_a/
│   │   ├── sample_1.wav
│   │   └── sample_2.wav
│   └── model_b/
│       ├── sample_1.wav
│       └── sample_2.wav
├── reference/                 ← per-segment reference audio (same filenames as models/)
│   ├── sample_1.wav
│   └── sample_2.wav
├── enrollment/
│   └── speaker.wav            ← single target-speaker clip (speaker_sim gate)
├── text_references/           ← ground-truth transcripts for WER gate
│   ├── sample_1.txt
│   └── sample_2.txt
└── accent_reference/          ← fixed accent reference clips (accent gate)
    ├── american.wav
    ├── british.wav
    └── indian.wav
```

**Rules:**
- All model folders must contain identical filenames.
- Reference filenames must match model filenames exactly.
- All audio must be `.wav`.

---

## Model weights

Large binary weights are gitignored. Run `python download_weights.py` to fetch them.

| Weight | Gate | Size | Auto-downloaded? |
|--------|------|------|-----------------|
| `weights/speaker_sim/` | Speaker Sim (ECAPA-TDNN) | ~87 MB | Yes — `download_weights.py` |
| `weights/chatterbox/` | Chatterbox TTS (voice cloning) | ~3 GB | Yes — `download_weights.py` (optional) |
| `weights/nisqa/` | NISQA | ~40 MB | Manual — see below |
| `weights/utmos/` | UTMOS | ~360 MB | Manual — see below |

### NISQA
```bash
git clone https://github.com/gabrielmittag/NISQA.git weights/nisqa
```
The `.tar` weight file ships inside the cloned repo.

### UTMOS
```bash
mkdir -p weights/utmos/simple
curl -L "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/epoch%3D3-step%3D7459.ckpt" \
     -o weights/utmos/simple/epoch=3-step=7459.ckpt
curl -L "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt" \
     -o weights/utmos/simple/wav2vec_small.pt
git clone https://github.com/sarulab-speech/UTMOS22.git weights/utmos
```

---

## Running the pipeline

```bash
# All gates
python run_pipeline.py

# Specific gates only
python run_pipeline.py --gates wer utmos accent artifact

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

## Gates

| Gate | Env | What it measures | Pass condition |
|------|-----|-----------------|----------------|
| `wer` | utmos | Word Error Rate + Whisper word confidence | WER ≤ 10%, ≥ 85% high-confidence words |
| `nisqa` | base | MOS, Noisiness, Discontinuity, Coloration, Loudness | MOS ≥ 3.75, each Δ within threshold |
| `utmos` | utmos | TTS naturalness MOS (1–5) | Score ≥ 3.0 |
| `speaker_sim` | base | ECAPA-TDNN cosine speaker similarity | Cosine ≥ 0.55 |
| `ser` | base | Emotion label match (emotion2vec) | Top-1 label matches reference |
| `arousal_valence` | base | Dimensional emotion delta vs reference | \|Δarousal\| ≤ 0.15, \|Δvalence\| ≤ 0.15 |
| `pitch` | base | F0 register, expressiveness, std ratio | Mean Δ ≤ 30 Hz, std ≥ 20 Hz, std ratio ≥ 0.5× ref |
| `duration` | base | TTS/reference duration ratio | Within ±10% |
| `vad` | base | Pause count, position, duration alignment | Count Δ ≤ 20, position ≤ 0.2 s, dur ratio 0.75–1.25 |
| `amplitude` | base | LUFS, LRA, spectral centroid, true peak | LUFS Δ ≤ 6.5, LRA Δ ≤ 3, centroid Δ ≤ 500 Hz, peak < −1 dBFS |
| `accent` | base | SpeechBrain ECAPA accent (16 labels, rank-based) | Top-1 predicted accent = US or Canada |
| `artifact` | base | Vocoder buzz (HNR) + background static + spectral artifacts | HNR ≥ 8 dB, no spectral artifact pattern |

All thresholds are in `config.py`.

### Speaker Similarity — enrollment vs reference

The `speaker_sim` gate supports two reference modes:

| Mode | When used | Best for |
|------|-----------|----------|
| **UTTERANCE_REF** | Matching `.wav` file found in `data/reference/` | Measuring per-line voice identity preservation |
| **ENROLLMENT_REF** | No matching file in `data/reference/` — falls back to `data/enrollment/speaker.wav` | Voice cloning validation (single target speaker, any content) |

For a voice cloning pipeline: put the target speaker's clip at `data/enrollment/speaker.wav` and leave `data/reference/` empty. Every segment compares against the enrollment embedding.

For a dubbing pipeline with per-line reference audio: put reference segments in `data/reference/` with matching filenames — each TTS segment compares against its corresponding reference line.

### Accent gate — ECAPA-TDNN (16 labels, rank-based)

Uses `Jzuluaga/accent-id-commonaccent_ecapa` (SpeechBrain ECAPA-TDNN, 16 English accent labels). With 16 labels, probability is diluted (~0.16 for clearly American speech), so the gate uses **rank** not probability:

- **PASS** — top-1 predicted label is `us` or `canada`
- **PASS_WARN** — passes but a non-target label ≥ 20% (potential accent leak)
- **NEAR_MISS** — top-2 is US/Canada and gap from top-1 < 0.005
- **FAIL** — top-1 is neither US nor Canada

### Artifact gate — three-step detection

1. **HNR** (tonal/metallic buzz): voiced frames below 8 dB → `ERR_VOICE_BUZZ`
2. **Pause floor** (background static): median dBFS of real pause frames above −35 dBFS → `ERR_BACKGROUND_STATIC`
3. **Spectral fingerprint** (vocoder artifacts): combined score from H1/H2 ratio, cepstral mid-quefrency energy, HF spectral flatness → `ERR_SPECTRAL_ARTIFACT`

---

## Voice cloning generation

To generate voice-cloned audio for evaluation:

```bash
# First download chatterbox weights (3 GB)
python download_weights.py   # without --skip-chatterbox

# Generate 5 test sentences in target speaker's voice (uses data/enrollment/speaker.wav)
python generate_chatterbox.py

# Generate episode-length cloned audio from per-segment references
python generate_episode_cloned.py     # Chatterbox — uses each reference as voice prompt
python generate_episode_f5tts.py      # F5TTS — same, different model
```

Generated files go to:
- `data/models/chatterbox/` — 5-sentence generic test
- `data/hindi_eval/models/chatterbox_cloned/` — episode with Chatterbox
- `data/hindi_eval/models/f5tts_cloned/` — episode with F5TTS

**F5TTS note:** provide `ref_text` (even a placeholder) to avoid Whisper download on first run.

### Running gates on episode/custom data

Every gate accepts CLI overrides for data paths:

```bash
python gates/gate_speaker_sim.py \
  --models-dir  data/hindi_eval/models \
  --ref-dir     data/hindi_eval/reference \
  --output-dir  output/episode_speaker_sim

python gates/gate_pitch.py \
  --models-dir  data/hindi_eval/models \
  --ref-dir     data/hindi_eval/reference \
  --output-dir  output/episode_pitch
```

`--ref-dir` is optional. If omitted, speaker_sim falls back to `--enrollment-file`; pitch runs in absolute-threshold-only mode.

### Pitch gate — PRAAT estimator

Uses PRAAT (via `praat-parselmouth`) for F0 estimation. Install: `pip install praat-parselmouth`.

- **Per segment**: mean F0 of voiced frames (PRAAT's autocorrelation resolves octave ambiguity, so mean is reliable)
- **Cross-segment summary**: median of per-segment mean deltas + Max Δ (worst case)
- **Voiced frames**: frames where PRAAT detects a fundamental frequency. Unvoiced frames (fricatives, stops, silence) are excluded from F0 computation
- **Degraded reference**: if reference voiced ratio < 0.2, delta comparisons are skipped for that segment
- **Validated**: estimator choice confirmed by isolated comparison against pyin and CREPE, with perceptual ground truth

---

## Output structure

```
output/runs/YYYY-MM-DD_HH-MM-SS/
├── pipeline.log
├── radar.png
├── heatmap.png
├── llm_report.txt            ← requires GEMINI_API_KEY
└── <gate>/
    ├── per_segment_results.csv
    └── model_summary.csv
```

---

## Hardware

All neural gates auto-detect: CUDA → MPS (Apple Silicon) → CPU.

| Gate | GPU accelerated |
|------|----------------|
| WER (Whisper) | Yes — MLX on Apple Silicon, CUDA elsewhere |
| UTMOS | Yes |
| Speaker Sim | Yes (set `FORCE_SPEAKER_CPU=1` to override) |
| Accent | Yes |
| Arousal/Valence | Yes |
| SER | CPU only |
| NISQA, Pitch, Duration, VAD, Amplitude, Artifact | CPU only |

---

## Language limitations

| Gate | English output | Hindi output | Cross-lingual (Hindi ref → English TTS) |
|------|---------------|-------------|----------------------------------------|
| WER | ✅ | ❌ | ❌ |
| NISQA | ✅ | ✅ (slightly biased) | ✅ |
| UTMOS | ✅ | ❌ | — |
| Speaker Sim | ✅ | ✅ | ✅ |
| SER | ✅ | ⚠️ | ⚠️ |
| Arousal/Valence | ✅ | ⚠️ | ⚠️ |
| Pitch / Duration / VAD / Amplitude | ✅ | ✅ | ✅ |
| Accent | ✅ | ❌ | ❌ |
| Artifact | ✅ | ✅ | — |

✅ reliable  ⚠️ use with caution  ❌ do not use

---

## Configuration reference (`config.py`)

| Key | Default | Gate |
|-----|---------|------|
| `WER_THRESHOLD` | 0.10 | WER |
| `INTEL_THRESHOLD` | 0.85 | WER |
| `NISQA_THRESHOLDS["MOS"]` | 3.75 | NISQA |
| `NISQA_DELTA_THRESHOLDS["MOS"]` | −0.5 | NISQA |
| `UTMOS_THRESHOLD` | 3.0 | UTMOS |
| `SPEAKER_SIM_THRESHOLD` | 0.50 | Speaker Sim |
| `SER_NEAR_MISS_MARGIN` | 0.10 | SER |
| `AROUSAL_DELTA_THRESHOLD` | 0.15 | Arousal/Valence |
| `VALENCE_DELTA_THRESHOLD` | 0.15 | Arousal/Valence |
| `PITCH_MEDIAN_THRESHOLD` | 30 Hz | Pitch |
| `PITCH_STD_ABS_THRESHOLD` | 20 Hz | Pitch |
| `PITCH_STD_RATIO_THRESHOLD` | 0.5 | Pitch |
| `DURATION_TOLERANCE` | 0.10 | Duration |
| `LUFS_TOLERANCE` | 6.5 | Amplitude |
| `CENTROID_TOLERANCE` | 500 Hz | Amplitude |
| `ACCENT_NEAR_MISS_MARGIN` | 0.07 | Accent |
| `ACCENT_LEAK_THRESHOLD` | 0.20 | Accent |
| `ARTIFACT_HNR_ABS_THRESHOLD` | 8.0 dB | Artifact |
| `MIN_SEGMENT_DURATION` | 2.0 s | All gates |
