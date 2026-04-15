"""
STS Pipeline Configuration
All paths and thresholds for the STS evaluation pipeline.
Weights are shared with the TTS pipeline (same models, different comparison logic).
"""
import os

# ── Roots ─────────────────────────────────────────────────────────────────────
TTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tts_metrics/
STS_ROOT = os.path.dirname(os.path.abspath(__file__))                   # tts_metrics/sts/

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(TTS_ROOT, "sts_output")

# ── Shared weights (same models as TTS pipeline) ──────────────────────────────
NISQA_REPO          = os.path.join(TTS_ROOT, "weights", "nisqa")
NISQA_WEIGHT        = os.path.join(NISQA_REPO, "weights", "nisqa.tar")
SPEAKER_SIM_WEIGHTS = os.path.join(TTS_ROOT, "weights", "speaker_sim")
MERALION_LOCAL_PATH = os.path.join(TTS_ROOT, "weights", "meralion_ser")
UTMOS_CONDA_ENV     = "utmos"

# ── Short segment ─────────────────────────────────────────────────────────────
MIN_SEGMENT_DURATION = 2.0   # seconds — flagged but still scored

# ── WER (content preservation — Hindi ASR, input vs output transcripts) ───────
WER_THRESHOLD        = 0.10  # max acceptable content-preservation WER; calibrate after run
WER_NEAR_MISS_MARGIN = 0.20  # in (0.10, 0.12] → NEAR_MISS

# ── NISQA ─────────────────────────────────────────────────────────────────────
# Absolute thresholds — same as TTS (both sides Hindi, no cross-lingual MOS penalty)
NISQA_THRESHOLDS = {
    "MOS"          : 3.75,
    "Noisiness"    : 3.5,
    "Discontinuity": 3.5,
    "Coloration"   : 4.0,
    "Loudness"     : 3.4,
}
NISQA_DELTA_THRESHOLDS = {
    "MOS"          : -0.5,
    "Noisiness"    : -0.5,
    "Discontinuity": -0.5,
    "Coloration"   : -0.5,
    "Loudness"     : -0.6,
}
# Input quality floor — tighter than TTS because both sides are Hindi (no cross-lingual penalty)
NISQA_REF_THRESHOLDS = {
    "MOS"          : 3.5,
    "Noisiness"    : 3.0,
    "Discontinuity": 3.0,
    "Coloration"   : 3.5,
    "Loudness"     : 3.0,
}

# ── Duration (output vs input) ────────────────────────────────────────────────
# Start at ±10% — may tighten to ±5% after seeing data.
# STS should preserve timing (same-language, same words) but real vocoders
# add/trim silence at boundaries, so tight tolerance may produce false failures.
DURATION_TOLERANCE        = 0.10
DURATION_NEAR_MISS_MARGIN = 0.20

# ── Amplitude (output vs input) ───────────────────────────────────────────────
LUFS_TOLERANCE     = 6.5     # LUFS delta — same as TTS; calibrate after run
LRA_TOLERANCE      = 3.0
CENTROID_TOLERANCE = 500     # Hz
PEAK_LIMIT         = -1.0    # dBFS — clipping threshold
CLIP_RATE_WARN     = 0.001
AMPLITUDE_NEAR_MISS_MARGIN = 0.20
REF_LUFS_MIN       = -40.0
REF_LUFS_MAX       = -5.0
TTS_LUFS_ABS_MIN   = -40.0
TTS_LUFS_ABS_MAX   = -5.0
TTS_LRA_ABS_MIN    = 0.5
TTS_LRA_ABS_MAX    = 20.0

# ── Pitch ─────────────────────────────────────────────────────────────────────
# Preservation axis (output vs input)
PITCH_STD_ABS_THRESHOLD   = 20.0   # Hz absolute floor — same as TTS
PITCH_STD_RATIO_THRESHOLD = 0.70   # output/input — tighter than TTS (0.50) for same-language
PITCH_RANGE_RATIO_MIN     = 0.60   # output/input range ratio
PITCH_NEAR_MISS_MARGIN    = 0.20
# Contour correlation (output vs input F0 frame-by-frame)
PITCH_CONTOUR_CORR_THRESHOLD  = 0.70   # Pearson r
PITCH_CONTOUR_CORR_NEAR_MISS  = 0.55
# Absolute sanity (used when input is unvoiced/degraded)
TTS_VOICED_ABS_MIN   = 0.10
TTS_PITCH_STD_ABS_MIN = 5.0
# Register axis (output vs train — target voice characteristic F0)
PITCH_REGISTER_DELTA_THRESHOLD    = 30.0   # Hz — max |output_median - train_median|
PITCH_REGISTER_NEAR_MISS_MARGIN   = 0.20

# ── VAD / Pause alignment (output vs input — soft gate) ───────────────────────
SILENCE_DB                 = -40     # dBFS threshold for silence detection
MIN_SILENCE_DURATION       = 0.5    # seconds — minimum pause duration
POSITION_HARD_LIMIT        = 3.0    # seconds — pairs beyond this never assigned
POSITION_WEIGHT            = 0.3
DURATION_WEIGHT            = 0.7
POSITION_SCALE             = 3.0
PAUSE_COUNT_THRESHOLD      = 20     # absolute cap
PAUSE_COUNT_RATE_THRESHOLD = 0.5    # pauses/sec for dynamic threshold
POSITION_OFFSET_THRESHOLD  = 0.15   # seconds — tighter than TTS (0.20)
VAD_DURATION_RATIO_MIN     = 0.80   # tighter than TTS (0.75)
VAD_DURATION_RATIO_MAX     = 1.20   # tighter than TTS (1.25)
REF_PAUSES_PER_SECOND_LIMIT = 1.0
TTS_PAUSES_PER_SEC_MAX     = 2.0
VAD_NEAR_MISS_MARGIN       = 0.20

# ── Speaker similarity ────────────────────────────────────────────────────────
# Primary: output vs train (does it sound like target voice?)
SPEAKER_SIM_THRESHOLD      = 0.50
SPEAKER_SIM_NEAR_MISS_MARGIN = 0.20
# Conversion check: output vs input (should be LOW — conversion happened)
CONVERSION_SIM_MAX  = 0.70   # above this → CONVERSION_WEAK warning
CONVERSION_SIM_WARN = 0.60   # in [0.60, 0.70) → note (borderline)

# ── SER (emotion transfer — output vs input) ──────────────────────────────────
SER_CONFIDENCE_THRESHOLD = 0.5
AROUSAL_DELTA_THRESHOLD  = 0.15
VALENCE_DELTA_THRESHOLD  = 0.15   # recorded as diagnostic, not used in pass/fail

# ── Artifact (output standalone + input baseline) ─────────────────────────────
ARTIFACT_HNR_ABS_THRESHOLD  = 8.0
ARTIFACT_SILENCE_DETECT_DB  = -30.0
ARTIFACT_MIN_PAUSE_FRAMES   = 10
ARTIFACT_SILENCE_FAIL_DB    = -35.0
ARTIFACT_SILENCE_WARN_DB    = -58.0
ARTIFACT_H1H2_THRESHOLD     = 1.5
ARTIFACT_CEP_MIDQ_THRESHOLD = 0.022
ARTIFACT_SFM_HF_THRESHOLD   = 0.16
ARTIFACT_CLEAN_H1H2         = 1.9
ARTIFACT_CLEAN_SFM_HF       = 0.134
ARTIFACT_CLEAN_CEP          = 0.017
ARTIFACT_COMBINED_THRESHOLD = 0.15
ARTIFACT_NEAR_MISS_MARGIN   = 0.20
