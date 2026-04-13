#!/bin/bash
# Speaker similarity — episode data
# Compares each TTS model segment against the matching Hindi reference audio
# (UTTERANCE_REF, not enrollment fallback)
cd /Users/abey/Documents/tts_metrics
echo "Running speaker similarity — episode models vs Hindi reference segments..."
FORCE_SPEAKER_CPU=1 NUMBA_CACHE_DIR=/tmp/claude/numba conda run -n base python3 gates/gate_speaker_sim.py \
    --models-dir data/hindi_eval/models \
    --ref-dir    data/hindi_eval/reference \
    --output-dir output/hindi_eval/speaker_sim
