#!/bin/bash
cd /Users/abey/Documents/tts_metrics
echo "Running accent gate (ECAPA) on episode data..."
NUMBA_CACHE_DIR=/tmp/claude/numba /Users/abey/miniconda3/bin/python3 gates/gate_accent.py \
    --models-dir data/hindi_eval/models \
    --output-dir output/hindi_eval/accent
