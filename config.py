import os

ROOT = "/Users/abey/Documents/tts_metrics"

# ── Shared data directory ──────────────────────────────────────────────────────
# All audio gates read from the same folder.  Put your files here once and
# every gate evaluates them — no duplication across gate-specific folders.
#
#   data/
#   ├── models/            ← TTS / STS model outputs, one sub-folder per model
#   │   ├── model_1/
#   │   └── model_2/
#   ├── reference/         ← per-segment human reference audio (same filenames)
#   ├── enrollment/        ← single speaker clip for speaker similarity gate
#   │   └── speaker.wav
#   └── text_references/   ← plain-text ground-truth transcripts for WER gate
#       └── sample_01.txt
#
DATA_DIR       = os.path.join(ROOT, "data")
MODELS_DIR     = os.path.join(DATA_DIR, "models")
REFERENCE_DIR  = os.path.join(DATA_DIR, "reference")
ENROLLMENT_DIR = os.path.join(DATA_DIR, "enrollment")

# ── WER gate paths ─────────────────────────────────────────────────────────────
TEXT_REFERENCE_DIR = os.path.join(DATA_DIR, "text_references")

# ── Accent class reference clips (NOT per-segment — one clip per accent class) ─
ACCENT_BASE_DIR = os.path.join(ROOT, "data")

# ── Model / weights paths ──────────────────────────────────────────────────────
NISQA_REPO   = os.path.join(ROOT, "weights", "nisqa")
NISQA_WEIGHT = os.path.join(NISQA_REPO, "weights", "nisqa.tar")

UTMOS_MODEL_DIR = os.path.join(ROOT, "weights", "utmos", "simple")
UTMOS_CKPT      = os.path.join(UTMOS_MODEL_DIR, "epoch=3-step=7459.ckpt")

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(ROOT, "output")

# ── WER thresholds ─────────────────────────────────────────────────────────────
WER_THRESHOLD    = 0.10   # max acceptable word error rate
INTEL_THRESHOLD  = 0.85   # min fraction of words above mumble threshold
MUMBLE_THRESHOLD = -1.0   # log prob below this = low confidence word

# ── NISQA thresholds ───────────────────────────────────────────────────────────
NISQA_THRESHOLDS = {
    "MOS"           : 3.0,
    "Noisiness"     : 3.5,
    "Discontinuity" : 3.5,
    "Coloration"    : 3.0,
    "Loudness"      : 3.0,
}
NISQA_DELTA_THRESHOLDS = {
    "MOS"           : -0.5,
    "Noisiness"     : -0.5,
    "Discontinuity" : -0.5,
    "Coloration"    : -0.5,
    "Loudness"      : -0.6,
}

# ── UTMOS thresholds ───────────────────────────────────────────────────────────
UTMOS_THRESHOLD = 3.0

# ── Short segment threshold ───────────────────────────────────────────────────
# Files shorter than this are still scored but flagged as SHORT_SEGMENT and
# counted as degraded in all gates. Fast-switching dialogue lines can be <1 s
# so we flag rather than skip — the score is real, just higher-uncertainty.
MIN_SEGMENT_DURATION = 2.0   # seconds

# ── Artifact / Vocoder Buzz gate thresholds ────────────────────────────────────
# Step 1: HNR — tonal / metallic buzz on voiced speech
ARTIFACT_HNR_ABS_THRESHOLD  = 8.0    # dB — below this → ERR_VOICE_BUZZ
                                      # validated: kokoro 12 dB (PASS), Telugu STS 7.3 dB (FAIL)

# Step 2: Real pause detection — contiguous frames below this threshold
# NOTE: separate from SILENCE_DB (-40) used by the VAD gate.
# -30 dB captures the noisy silence frames we want to measure.
ARTIFACT_SILENCE_DETECT_DB  = -30.0  # dBFS — frame qualifies as silence if below this
ARTIFACT_MIN_PAUSE_FRAMES   = 10     # frames at hop=256/sr=16k → 160 ms minimum pause
                                      # samantha_2 has 0 real pauses → N/A (correct)

# Step 3: Median dBFS of real pause frames — background static
ARTIFACT_SILENCE_MEDIAN_DB  = -55.0  # dBFS — above this → ERR_BACKGROUND_STATIC
                                      # validated: kokoro −65 dBFS (PASS, 10 dB margin)
                                      #            fastspeech2 −44 dBFS (FAIL, 11 dB margin)

# Step 4: Spectral shape artifact detection (catches f5tts-class artifacts)
# These three metrics combined detect subtle vocoder/flow-matching artifacts.
# H1/H2 Ratio: ratio of first-harmonic to second-harmonic amplitude in voiced frames.
#   Low ratio = flattened harmonic slope → vocoder spectral shaping artifact.
ARTIFACT_H1H2_THRESHOLD     = 1.5    # below this → ERR_SPECTRAL_ARTIFACT
                                      # f5tts: 1.27 (FAIL), clean: 1.65-2.89 (PASS)
# Cepstral mid-quefrency energy: normalized energy in pitch-period quefrency range.
#   High value = excessive periodicity structure in spectrum → tonal artifact.
ARTIFACT_CEP_MIDQ_THRESHOLD = 0.022  # above this → ERR_SPECTRAL_ARTIFACT
                                      # f5tts: 0.028 (FAIL), clean: 0.015-0.019 (PASS)
# SFM 4-8kHz (voiced): spectral flatness measure of high-freq in voiced frames.
#   High value = broadband noise in HF → broadband artifact.
ARTIFACT_SFM_HF_THRESHOLD   = 0.16   # above this → ERR_SPECTRAL_ARTIFACT
                                      # f5tts: 0.172 (FAIL), clean: 0.130-0.138 (PASS)
# Combined score normalization reference values (derived from clean model means)
ARTIFACT_CLEAN_H1H2   = 1.9          # reference H1H2 for normalisation
ARTIFACT_CLEAN_SFM_HF = 0.134        # reference SFM_4-8k for normalisation
ARTIFACT_CLEAN_CEP    = 0.017        # reference Cep_MidQ for normalisation
ARTIFACT_COMBINED_THRESHOLD = 0.25   # combined score above this → ERR_SPECTRAL_ARTIFACT

# ── Speaker similarity thresholds ─────────────────────────────────────────────
SPEAKER_SIM_THRESHOLD = 0.75

# ── SER thresholds ─────────────────────────────────────────────────────────────
SER_CONFIDENCE_THRESHOLD = 0.5
# A FAIL becomes NEAR_MISS if the confidence gap between top-1 and the reference
# label (on the TTS side) OR between top-1 and top-2 (on the reference side) is
# within this margin. Both checks use the same margin.
SER_NEAR_MISS_MARGIN = 0.10

# ── Pitch thresholds ───────────────────────────────────────────────────────────
PITCH_MEDIAN_THRESHOLD    = 30.0   # Hz — max acceptable |ref_median - tts_median|
PITCH_STD_ABS_THRESHOLD   = 20.0   # Hz — minimum TTS pitch std (expressiveness floor)
PITCH_STD_RATIO_THRESHOLD = 0.5    # TTS std must be >= 0.5x reference std

# ── Duration thresholds ────────────────────────────────────────────────────────
DURATION_TOLERANCE = 0.10   # ±10%

# ── VAD / Pause alignment thresholds ──────────────────────────────────────────
SILENCE_DB                  = -40    # dB threshold for silence detection
MIN_SILENCE_DURATION        = 0.5    # seconds — minimum pause duration
POSITION_HARD_LIMIT         = 3.0   # seconds — pairs beyond this never assigned
POSITION_WEIGHT             = 0.3   # weight for position component of cost
DURATION_WEIGHT             = 0.7   # weight for duration component of cost
POSITION_SCALE              = 3.0   # seconds — position diff of this size = cost 1.0
PAUSE_COUNT_THRESHOLD       = 20    # max acceptable pause count difference
POSITION_OFFSET_THRESHOLD   = 0.5   # seconds — max acceptable median position offset
VAD_DURATION_RATIO_MIN      = 0.75  # TTS pause at least 75% as long as reference
VAD_DURATION_RATIO_MAX      = 1.25  # TTS pause at most 125% as long as reference
REF_PAUSES_PER_SECOND_LIMIT = 1.0   # pauses/sec above this = degraded

# ── Amplitude thresholds ───────────────────────────────────────────────────────
LUFS_TOLERANCE     = 4.0     # LUFS delta tolerance
LRA_TOLERANCE      = 3.0     # LUFS LRA delta tolerance
CENTROID_TOLERANCE = 500     # Hz spectral centroid delta tolerance
PEAK_LIMIT         = -1.0    # dBFS — TTS clipping threshold
REF_LUFS_MIN       = -40.0   # below this = degraded reference
REF_LUFS_MAX       = -5.0    # above this = degraded reference

# ── Arousal / Valence thresholds ──────────────────────────────────────────────
AROUSAL_DELTA_THRESHOLD = 0.15   # max |ref_arousal - out_arousal| — above = emotion intensity lost
VALENCE_DELTA_THRESHOLD = 0.15   # max |ref_valence - out_valence| — above = tone polarity shifted

# ── Accent thresholds ──────────────────────────────────────────────────────────
ACCENT_TARGET           = "american"
ACCENT_TARGET_THRESHOLD = 0.75
ACCENT_REFERENCES = {
    "american": os.path.join(ACCENT_BASE_DIR, "accent_reference", "american.wav"),
    "british" : os.path.join(ACCENT_BASE_DIR, "accent_reference", "british.wav"),
    "indian"  : os.path.join(ACCENT_BASE_DIR, "accent_reference", "indian.wav"),
}

# ── Conda environment names ────────────────────────────────────────────────────
UTMOS_CONDA_ENV = "utmos"   # python 3.9 — used for WER, UTMOS, accent gates
