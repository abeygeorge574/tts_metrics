#!/bin/bash
# Speaker similarity cloning test
# Compares ALL test models against the target enrollment speaker (speaker.wav)
# f5tts was cloned from this speaker — it should score HIGH
# chatterbox / xtts (if generated) should also score HIGH
# All other models are generic TTS — they should score LOW
# This validates the gate works for voice cloning use cases.

cd /Users/abey/Documents/tts_metrics

# Build a combined models dir: link f5tts + cloning models + generic TTS
STAGING="$TMPDIR/cloning_test_models"
rm -rf "$STAGING" && mkdir -p "$STAGING"

# Always include: f5tts (cloning) + all generic models
for model in data/models/*/; do
    name=$(basename "$model")
    ln -s "$(pwd)/$model" "$STAGING/$name"
done

# Include chatterbox if generated
if [ -d "data/models/chatterbox" ]; then
    echo "Including chatterbox samples."
fi

# Include xtts if generated
if [ -d "data/models/xtts" ]; then
    echo "Including xtts samples."
fi

echo "Running speaker similarity — all models vs enrollment speaker..."
echo "Models dir: $STAGING"
ls "$STAGING"

FORCE_SPEAKER_CPU=1 NUMBA_CACHE_DIR=/tmp/claude/numba conda run -n base python3 gates/gate_speaker_sim.py \
    --models-dir data/models \
    --ref-dir    /tmp/empty_nonexistent_ref \
    --enrollment-file data/enrollment/speaker.wav \
    --output-dir output/speaker_sim_cloning_test
