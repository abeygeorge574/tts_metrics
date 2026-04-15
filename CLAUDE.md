# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

A TTS/STS (Text-To-Speech / Speech-To-Speech) audio quality evaluation pipeline for Hindi-English dubbing. It runs a set of "gates" — modular quality checks — against synthesized speech to determine production readiness. Each gate evaluates a different dimension: intelligibility, naturalness, speaker similarity, pitch, duration, pause alignment, loudness, emotion, and accent.

## Running the pipeline

```bash
# Run all gates
python run_pipeline.py

# Run specific gates only
python run_pipeline.py --gates wer nisqa

# Skip specific gates
python run_pipeline.py --skip nisqa pitch

# Custom output directory
python run_pipeline.py --output-dir /path/to/results
```

Active gate names: `wer`, `nisqa`, `speaker_sim`, `ser`, `pitch`, `duration`, `vad`, `amplitude`, `accent`, `artifact`

Dropped gates (files exist but not in pipeline):
- `utmos` — same verdicts as NISQA, redundant; not in GATE_REGISTRY
- `arousal_valence` — merged into `ser` gate (arousal delta + arousal_pass columns in SER output)

## Environment setup

Two conda environments are required:
- **`utmos`** (Python 3.9): used for `wer` gate only — invoked as subprocess
- **`base`** (Python 3.13): all other gates run as direct imports in the current process

The pipeline auto-detects the `utmos` conda python at common paths (`~/miniconda3/envs/utmos/bin/python`, etc.) and falls back to `conda run -n utmos`. Run `run_pipeline.py` from the base (Python 3.13) environment.

`ffmpeg` must be installed and on PATH (required by the VAD gate for silence detection).

## Architecture

### Two-environment design

Gates that depend on older ML models (Whisper for WER) require Python 3.9 and are invoked as subprocesses. All other gates are imported and called directly. The `GATE_REGISTRY` in `run_pipeline.py` tracks which environment each gate belongs to.

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
- SER uses three-way: PASS / NEAR_MISS / FAIL — near-miss when top-2 emotion label sets overlap

### Language limitations

This pipeline targets **English TTS/STS output** evaluated against Hindi reference audio.

- **Unreliable for non-English output:** WER (Whisper ASR is English-only)
- **Partially reliable:** SER (LOW_CONF_REF degraded bucket handles uncertain Hindi references — MERaLiON-SER-v1 used)
- **Fully reliable regardless of language:** Pitch, Duration, VAD, Amplitude, Speaker Sim, NISQA (relative comparisons valid)
- **Accent is meaningful:** ECAPA classifier correctly catches Indian-accented TTS — voice-cloned models from Hindi speakers score low (chatterbox_cloned: 2/6 PASS, 4/6 FAIL, classified as Indian). This is the correct and expected behaviour for English dubbing quality gates.

## Configuration

All thresholds and paths live in `config.py`. Key thresholds:

| Gate | Key threshold |
|------|--------------|
| WER | max 10% word error rate, intelligibility ≥ 85% |
| NISQA | MOS ≥ 3.75, Noisiness ≥ 3.5, Discontinuity ≥ 3.5, Coloration ≥ 4.0, Loudness ≥ 3.4 |
| Speaker similarity | cosine ≥ 0.50 |
| SER | top-1 confidence ≥ 0.5; near-miss = top-2 label set overlap |
| Duration | within ±10% of reference |
| Pitch | median delta ≤ 30 Hz, std ≥ 0.5× reference |
| Amplitude | LUFS ±6.5, LRA ±3.0, spectral centroid ±500 Hz |
| Accent | top-1 predicted label must be "us" or "canada" (ECAPA-TDNN, 16 labels) |
| Artifact | HNR ≥ 8 dB, pause median ≤ −35 dBFS, SA combined ≤ 0.15 |

Model weight paths (gitignored, must be present locally):
- NISQA: `weights/nisqa/weights/nisqa.tar`
- Speaker sim (ECAPA-TDNN): `weights/speaker_sim/`
- SER (MERaLiON-SER-v1): path set in `config.MERALION_LOCAL_PATH` (must be manually downloaded — proxy blocks HF large blobs)

## Model evolution history

This section documents why each gate uses its current model. Do not change models without reading this first.

### SER gate — emotion2vec → MERaLiON-SER-v1

**Original (V1):** Two separate models:
- `FunAudioLLM/emotion2vec_plus_large` for categorical emotion (SER gate)
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` for arousal/valence (separate `arousal_valence` gate)

**Problems with V1:**
- emotion2vec is trained on short utterances; chunking at 15 s helped but emotion only became clear mid-utterance on longer dubbing segments
- audeering valence produced a systematic ~0.25 valence gap when comparing Hindi reference to English TTS — not a real quality signal, just cross-lingual bias (model trained on English MSP-Podcast data only). Valence was dropped from pass/fail scoring before the switch.

**Current (V2): MERaLiON-SER-v1** (`MERaLiON/MERaLiON-SER-v1`)
- Architecture: Whisper-Medium + LoRA + ECAPA-TDNN
- 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
- Single inference pass gives both categorical label AND dimensional arousal/valence/dominance
- arousal_valence gate merged into SER — no separate gate needed
- Valence kept as diagnostic column only (cross-lingual bias documented, not used for pass/fail)
- NEAR_MISS = top-2 predicted label sets overlap between reference and TTS (more lenient than exact match — accounts for adjacent emotion confusion)
- Must be manually downloaded: `huggingface_hub.snapshot_download('MERaLiON/MERaLiON-SER-v1', local_dir=config.MERALION_LOCAL_PATH)` — proxy blocks large HF blobs

### Pitch gate — pyin → PRAAT (parselmouth)

**Original (V1):** `librosa.pyin` with voiced_probs filtering

**Problems with V1:** Produced 1500–2000 Hz estimates on expressive voices — clearly broken. Vocal prob filters couldn't reliably distinguish genuine high-pitch frames from estimation errors.

**Current (V2): PRAAT via parselmouth** (autocorrelation algorithm)
- PRAAT handles octave ambiguity internally; no post-hoc correction needed
- Validated against listening: chatterbox PSTSBF gave 45 Hz FAIL with pyin → 21 Hz PASS with PRAAT (ear agreed with PRAAT)
- f5tts improved from 4/6 → 6/6; chatterbox 4/6 → 5/6
- F0 summary: **median** of voiced frames per segment (robust to octave errors; standard approach)
- Several iterations tried: mean, octave-corrected mean — median was most consistent

### Accent gate — wav2vec2 cosine → dima806 classifier → ECAPA rank-based

**Original (V1):** wav2vec2 cosine similarity against english/hebrew/hindi .mp3 reference clips
- Problem: noisy, threshold unclear, required reference clips

**V2:** `dima806` softmax classifier, target P(us) ≥ 0.60
- Clear separation: english1 = 85% us, hindi1 = 99% indian
- Problem: US+Canada should both count as North American for dubbing

**V3:** North American criterion: P(us) + P(canada) ≥ 0.75
- Canada and US are same accent class for English dubbing

**Current (V4): `Jzuluaga/accent-id-commonaccent_ecapa`** (ECAPA-TDNN, 16 accent labels)
- Rank-based pass: top-1 predicted label must be "us" or "canada"
- Moved from `utmos` conda env (Python 3.9) to `base` env (Python 3.13) — runs faster as direct import
- NEAR_MISS: top-1 non-target but within 7% of passing
- PASS_WARN: passes but a non-target label exceeds 20% (accent leak)
- **Validated on real data:** chatterbox_cloned (voice-cloned from Hindi speaker) → 2/6 PASS, 4/6 FAIL (classified Indian). This is correct and expected. gtts → similar pattern. Pure English TTS (kokoro, edge-tts, parler) → all PASS.

### Speaker Sim gate — threshold calibration history

Model unchanged (SpeechBrain ECAPA-TDNN), but loaded from local weights (`weights/speaker_sim/`) after SOCKS proxy blocked HF hub downloads.

**Threshold evolution:**
- 0.75 (initial) → 0.55 (first calibration) → **0.50 (current)**
- Calibrated on real cloning data: chatterbox_cloned vs Hindi reference segments → 0.53–0.78 cosine (median 0.66)
- Generic TTS (not cloning target speaker) → −0.13–0.24 (all far below threshold)
- 0.50 correctly captures borderline clones while staying well above the generic TTS ceiling (0.24)

### Artifact gate — threshold calibration history

Three independent sub-metrics, each with their own threshold evolution:

- **HNR:** ≥ 8 dB (unchanged). Validated: kokoro ~12 dB (PASS), Telugu STS ~7.3 dB (FAIL).
- **Pause silence floor (ARTIFACT_SILENCE_FAIL_DB):** raised from −45 → **−35 dBFS**. At −45, fastspeech2 and parler_mini both failed — but parler_mini was user-rated as WARN not FAIL. −35 correctly puts parler_mini in WARN band and fastspeech2 in hard FAIL.
- **Spectral artifact combined score (ARTIFACT_COMBINED_THRESHOLD):** lowered from 0.30 → **0.15**. At 0.30 some fastspeech2 samples slipped through. At 0.15, all fastspeech2 samples fail with margin and no clean models are affected (max clean score: kokoro 0.073).

## Adding a new gate

1. Create `gates/gate_<name>.py` implementing the four-function interface above
2. Add an entry to `GATE_REGISTRY` in `run_pipeline.py` with the env (`"base"` or `"utmos"`)
3. Add base directory, thresholds, and any weight paths to `config.py`
