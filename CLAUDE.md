# CLAUDE.md

## What this project does

TTS/STS audio quality evaluation pipeline for Hindi-English dubbing. Runs quality gates against synthesized speech to determine production readiness. Each gate evaluates one dimension: intelligibility, naturalness, speaker similarity, pitch, duration, pause alignment, loudness, emotion, and artifacts.

## Running the pipeline

```bash
python run_pipeline.py                        # all gates
python run_pipeline.py --gates wer utmos      # specific gates
python run_pipeline.py --skip nisqa           # skip gates
python run_pipeline.py --output-dir /path/to/out
```

Valid gates: `wer`, `nisqa`, `utmos`, `speaker_sim`, `ser`, `arousal_valence`, `pitch`, `duration`, `vad`, `amplitude`, `accent`, `artifact`

## Environment

Two conda envs:
- **`base`** (Python 3.13): most gates, run pipeline from here
- **`utmos`** (Python 3.9): `wer`, `utmos`, `accent` — auto-subprocessed

## Architecture

### Gate interface (every gate must expose)
```python
model_state = load_model()
df, summary_df = run_gate(model_state)
print_results(df, summary_df)
save_results(df, summary_df, output_dir)
```

### Data layout
```
data/
├── models/<model_name>/<sample>.wav
├── reference/<sample>.wav
├── enrollment/speaker.wav
├── text_references/<sample>.txt
└── accent_reference/american.wav
```

### Output
Each run: `output/runs/YYYY-MM-DD_HH-MM-SS/` with per-gate CSVs + pipeline.log

## Configuration
All thresholds in `config.py`. **Calibrated values (updated Apr 2026):**

| Gate | Key threshold |
|------|--------------|
| WER | WER ≤ 0.10, Intel ≥ 0.85 |
| NISQA | MOS ≥ **3.75**, Noisiness ≥ 3.5, Coloration ≥ **4.0**, Loudness ≥ 3.4, ΔMOS ≥ −0.5 |
| UTMOS | ≥ 3.0 (redundant — same verdicts as NISQA; consider removing from hard gates) |
| Speaker sim | cosine ≥ 0.75 |
| Duration | ±10% |
| Pitch | median Δ ≤ 30 Hz, std ≥ 0.5× ref, abs std ≥ 20 Hz |
| Amplitude | LUFS ±6.5, LRA ±3, centroid ±500 Hz |
| Accent | "american" proximity ≥ 0.75 (language detector only, not accent classifier) |
| Artifact | HNR ≥ 8 dB, pause median ≤ −35 dBFS (FAIL) / −58 dBFS (WARN), SA_combined ≤ 0.30 |

Model weights (gitignored, must be local):
- NISQA: `weights/nisqa/weights/nisqa.tar`
- UTMOS: `weights/utmos/simple/epoch=3-step=7459.ckpt`

## Datasets

Two separate datasets — always clarify which one is being used:

| Dataset | Path | Models | Samples | Purpose |
|---------|------|--------|---------|---------|
| Test set | `data/models/` | 12 models | 5 synthetic (sample_1–5.wav) | Gate calibration |
| Episode data | `data/hindi_eval/models/` | 5 models | 6 real Hindi-dubbed segments (PSTSBF-… IDs) | Production eval |

- Episode text refs: `data/hindi_eval/text_refs/`
- Episode reference audio: `data/hindi_eval/reference/`
- Episode output goes to: `output/hindi_eval/<gate>/`
- Test output goes to: `output/<gate>/`

Episode models (5): edge_tts_andrew, edge_tts_ava, gtts, kokoro, parler_mini

## Known gate limitations

- **Duration, Pitch (register), Amplitude (dynamics/EQ), SER, Arousal/Valence**: unreliable with gtts proxy reference. gtts is the slowest, flattest, most muffled TTS — all comparisons skew. Meaningful only with real Hindi reference.
- **Accent gate**: detects English vs non-English only. All English TTS score ~1.0 for american/british/indian equally.
- **UTMOS**: misses vocoder buzz (fastspeech2, mms score 4.0+ despite audible artifacts). Artifact gate catches what UTMOS misses.
- **WER**: intelligibility only — fastspeech2/speecht5_hifigan pass despite buzz because Whisper transcribes correctly.
- **Artifact gate verdicts (12 test models)**: PASS: kokoro, kokoro_v1, edge_tts_ava, edge_tts_andrew, gtts, parler_mini, samantha. FAIL: f5tts, fastspeech2, mms, speecht5_hifigan, speecht5_griffinlim.

## Run commands

Always `cd /Users/abey/Documents/tts_metrics` before running — CWD resets to worktree otherwise.

```bash
# Base env gates (amplitude, pitch, vad, etc.) need numba cache:
cd /Users/abey/Documents/tts_metrics
NUMBA_CACHE_DIR=$TMPDIR/numba_cache /Users/abey/miniconda3/bin/python run_pipeline.py --gates <gate>

# WER/UTMOS/Accent (utmos env) — add MPS fallback to prevent Metal GPU crash:
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/abey/miniconda3/envs/utmos/bin/python gates/gate_wer.py

# Run on episode data (after adding --models-dir/--refs-dir CLI args):
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/abey/miniconda3/envs/utmos/bin/python gates/gate_wer.py \
  --models-dir data/hindi_eval/models --refs-dir data/hindi_eval/text_refs \
  --output-dir output/hindi_eval/wer
```

## Dual threshold (pending work)
Reference audio (Hindi) may score lower on NISQA due to cross-lingual penalty. Proposed: `NISQA_REF_THRESHOLDS` (permissive) separate from `NISQA_THRESHOLDS` (strict for TTS output). Not yet implemented.

---

## Communication rules

**Format:**
- Short sentences. No filler, no preamble, no pleasantries.
- Lead with tool call or result, not with explanation.
- Don't explain unless asked.
- Code: write normally. English prose: compress aggressively.
- Use tables over prose for comparisons and data.
- No "I'll now...", "Let me...", "As you mentioned..." constructions.
- Don't summarize what you just did — the diff speaks for itself.

**Reasoning (preserve, don't cut):**
- Still think through tradeoffs before acting on ambiguous requests.
- Flag risks or mismatches between what's asked and what makes sense — one sentence.
- If something is a known limitation or gap, say so directly.
- State assumptions when they're non-obvious.

**Token efficiency:**
- Omit restatement of the user's message.
- Omit confirmation of completed tool calls unless the result is surprising.
- Collapse lists of similar items; expand only what differs.
- Don't pad responses to seem thorough — short and correct beats long and complete.
