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
All thresholds in `config.py`. Key ones:

| Gate | Key threshold |
|------|--------------|
| WER | WER ≤ 0.10, Intel ≥ 0.85 |
| NISQA | MOS ≥ 3.0, ΔMOS ≥ −0.5 |
| UTMOS | ≥ 3.0 |
| Speaker sim | cosine ≥ 0.75 |
| Duration | ±10% |
| Pitch | median Δ ≤ 30 Hz, std ≥ 0.5× ref |
| Amplitude | LUFS ±4, LRA ±3, centroid ±500 Hz |
| Accent | "american" proximity ≥ 0.75 |
| Artifact | HNR ≥ 8 dB, pause median ≤ −45 dBFS (FAIL) / −58 dBFS (WARN), SA_combined ≤ 0.30 |

Model weights (gitignored, must be local):
- NISQA: `weights/nisqa/weights/nisqa.tar`
- UTMOS: `weights/utmos/simple/epoch=3-step=7459.ckpt`

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
