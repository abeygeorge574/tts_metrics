#!/bin/bash
# Speaker similarity — voice cloning validation
# Compares ALL models against the enrollment speaker (speaker.wav).
# No per-segment reference dir — enrollment is the correct reference for cloning.
# Expected: cloning models (chatterbox, f5tts) score HIGH; generic TTS score LOW.
cd /Users/abey/Documents/tts_metrics
echo "Running speaker similarity — all models vs enrollment speaker..."
FORCE_SPEAKER_CPU=1 NUMBA_CACHE_DIR=/tmp/claude/numba /Users/abey/miniconda3/bin/python3 gates/gate_speaker_sim.py \
    --models-dir      data/models \
    --enrollment-file data/enrollment/speaker.wav \
    --output-dir      output/speaker_sim_cloning_test
