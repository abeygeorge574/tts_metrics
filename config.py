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
    "MOS"           : 3.75,  # raised 3.0→3.75: floor calibrated from data (min passing MOS = 4.11)
    "Noisiness"     : 3.5,   # calibrated: catches background noise (SNR<50dB). Vocoder noise caught by artifact gate separately.
    "Discontinuity" : 3.5,
    "Coloration"    : 4.0,   # raised 3.0→4.0: hp350Hz/hp400Hz perceptually unacceptable for dubbing
    "Loudness"      : 3.4,   # raised 3.0→3.4: loud+9dB now fails (perceptually correct)
}
NISQA_DELTA_THRESHOLDS = {
    "MOS"           : -0.5,
    "Noisiness"     : -0.5,
    "Discontinuity" : -0.5,
    "Coloration"    : -0.5,
    "Loudness"      : -0.6,
}
# Permissive thresholds for reference audio quality.
# If reference fails these, delta comparison is skipped (segment → degraded bucket).
# Lower than NISQA_THRESHOLDS because:
#   - Hindi reference gets a cross-lingual NISQA penalty (trained on English)
#   - We only want to flag genuinely bad reference (noisy recording, clipping, etc.)
NISQA_REF_THRESHOLDS = {
    "MOS"           : 3.0,
    "Noisiness"     : 3.0,
    "Discontinuity" : 3.0,
    "Coloration"    : 3.0,
    "Loudness"      : 3.0,
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
ARTIFACT_SILENCE_FAIL_DB    = -35.0  # dBFS — above this → ERR_BACKGROUND_STATIC (hard FAIL)
                                      # raised -45→-35: fastspeech2/parler separation done by
                                      # ERR_SPECTRAL_ARTIFACT, not silence floor alone.
                                      # parler_mini (-36 to -40) user-rated WARN → moved to WARN band.
ARTIFACT_SILENCE_WARN_DB    = -58.0  # dBFS — above this → WARN_SILENCE_FLOOR (soft, not a FAIL)
                                      # kokoro_v1 −46 to −55 dBFS (WARN), kokoro −64 dBFS (PASS)

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
ARTIFACT_COMBINED_THRESHOLD = 0.15   # combined score above this → ERR_SPECTRAL_ARTIFACT
                                      # clean models max: 0.073 (kokoro); artifact min: 0.179 (fastspeech2)
                                      # H1/H2 individual gate removed (caused false positives on edge-tts)
                                      # lowered 0.30→0.15: catches all fastspeech2 samples w/ ample margin

# MERaLiON-SER-v1 — must be manually downloaded (proxy blocks HF hub for large blobs)
# Download: huggingface_hub.snapshot_download('MERaLiON/MERaLiON-SER-v1', local_dir=MERALION_LOCAL_PATH)
MERALION_LOCAL_PATH = "/tmp/claude/hf_cache/meralion-ser-v1"

SPEAKER_SIM_WEIGHTS = os.path.join(ROOT, "weights", "speaker_sim")

# ── Speaker similarity thresholds ─────────────────────────────────────────────
# Calibrated from episode cloning (chatterbox_cloned vs Hindi references):
#   cloning models: 0.53–0.78 (median 0.66)
#   generic TTS:   -0.13–0.24 (all fail)
# 0.50 captures borderline clones while staying well above generic TTS ceiling (0.24).
SPEAKER_SIM_THRESHOLD = 0.50

# ── SER thresholds ─────────────────────────────────────────────────────────────
SER_CONFIDENCE_THRESHOLD = 0.5

# ── Pitch thresholds ───────────────────────────────────────────────────────────
PITCH_MEDIAN_THRESHOLD    = 30.0   # Hz — max acceptable |ref_median - tts_median|
PITCH_STD_ABS_THRESHOLD   = 20.0   # Hz — minimum TTS pitch std (expressiveness floor)
PITCH_STD_RATIO_THRESHOLD = 0.5    # TTS std must be >= 0.5x reference std
PITCH_NEAR_MISS_MARGIN    = 0.20   # 20% beyond threshold → NEAR_MISS not FAIL
TTS_VOICED_ABS_MIN        = 0.10   # TTS voiced ratio below this → Unvoiced_Abs fail when degraded
TTS_PITCH_STD_ABS_MIN     = 5.0    # Hz — TTS F0 std below this → Flat_Abs fail when degraded

# ── Duration thresholds ────────────────────────────────────────────────────────
DURATION_TOLERANCE        = 0.10   # ±10%

# ── VAD / Pause alignment thresholds ──────────────────────────────────────────
SILENCE_DB                  = -40    # dB threshold for silence detection
MIN_SILENCE_DURATION        = 0.5    # seconds — minimum pause duration
POSITION_HARD_LIMIT         = 3.0   # seconds — pairs beyond this never assigned
POSITION_WEIGHT             = 0.3   # weight for position component of cost
DURATION_WEIGHT             = 0.7   # weight for duration component of cost
POSITION_SCALE              = 3.0   # seconds — position diff of this size = cost 1.0
PAUSE_COUNT_THRESHOLD       = 20    # max acceptable pause count difference
POSITION_OFFSET_THRESHOLD   = 0.2   # seconds — tightened 0.5→0.2: dubbing sync requires tight pause alignment
VAD_DURATION_RATIO_MIN      = 0.75  # TTS pause at least 75% as long as reference
VAD_DURATION_RATIO_MAX      = 1.25  # TTS pause at most 125% as long as reference
REF_PAUSES_PER_SECOND_LIMIT = 1.0   # pauses/sec above this = degraded

# ── Amplitude thresholds ───────────────────────────────────────────────────────
LUFS_TOLERANCE     = 6.5     # LUFS delta tolerance — raised 4→6.5: ±6dB passes, ±9dB fails
LRA_TOLERANCE      = 3.0     # LRA delta tolerance
CENTROID_TOLERANCE = 500     # Hz spectral centroid delta tolerance
PEAK_LIMIT         = -1.0    # dBFS — clipping threshold (applied to both ref and TTS)
REF_LUFS_MIN       = -40.0   # below this = degraded reference
REF_LUFS_MAX       = -5.0    # above this = degraded reference

# Near-miss margin: delta in (threshold, threshold × (1 + margin)] → NEAR_MISS not FAIL
AMPLITUDE_NEAR_MISS_MARGIN = 0.20   # 20% beyond threshold

# Clipping rate: fraction of samples at or above PEAK_LIMIT
# 0 < rate < CLIP_RATE_WARN  → NEAR_CLIP (warn, not fail)
# rate >= CLIP_RATE_WARN     → CLIPPING (hard fail)
CLIP_RATE_WARN = 0.001   # 0.1% of samples — ~110 samples in a 5s/22kHz file

# Absolute TTS bounds (used when ref is degraded — sanity checks not quality gates)
TTS_LUFS_ABS_MIN = -40.0   # below this = near-silent TTS
TTS_LUFS_ABS_MAX = -5.0    # above this = dangerously loud TTS
TTS_LRA_ABS_MIN  = 0.5     # below this = completely flat/robotic delivery
TTS_LRA_ABS_MAX  = 20.0    # above this = erratic dynamics

# ── Arousal / Valence thresholds ──────────────────────────────────────────────
AROUSAL_DELTA_THRESHOLD = 0.15   # max |ref_arousal - out_arousal| — above = emotion intensity lost
VALENCE_DELTA_THRESHOLD = 0.15   # max |ref_valence - out_valence| — above = tone polarity shifted

# ── Accent thresholds ──────────────────────────────────────────────────────────
ACCENT_TARGET_LABELS    = ["us", "canada"]   # North American English — treated as same accent for dubbing
ACCENT_TARGET_THRESHOLD = 0.75              # P(us) + P(canada) >= 0.75 to pass
ACCENT_NEAR_MISS_MARGIN = 0.07             # FAIL within this margin of threshold → NEAR_MISS
ACCENT_LEAK_THRESHOLD   = 0.20             # non-target label above this while PASS → WARN_ACCENT_LEAK
ACCENT_TARGET           = "us"              # legacy key, not used by gate logic
ACCENT_REFERENCES = {  # kept for reference, not used by classifier-based gate
    "english": os.path.join(ACCENT_BASE_DIR, "accent_reference", "english1.mp3"),
    "hebrew" : os.path.join(ACCENT_BASE_DIR, "accent_reference", "hebrew1.mp3"),
    "hindi"  : os.path.join(ACCENT_BASE_DIR, "accent_reference", "hindi1.mp3"),
}

# ── SER additions ────────────────────────────────────────────────────────────
# (SER_CONFIDENCE_THRESHOLD already exists at 0.5)

# ── VAD additions ────────────────────────────────────────────────────────────
VAD_NEAR_MISS_MARGIN    = 0.20   # 20% beyond threshold → NEAR_MISS
TTS_PAUSES_PER_SEC_MAX  = 2.0    # above this = implausibly dense TTS (abs bound)

# ── WER additions ────────────────────────────────────────────────────────────
WER_NEAR_MISS_MARGIN    = 0.20   # WER in (0.10, 0.12] → NEAR_MISS

# ── Speaker similarity additions ─────────────────────────────────────────────
SPEAKER_SIM_NEAR_MISS_MARGIN = 0.20  # score in [0.40, 0.50) → NEAR_MISS

# ── Artifact additions ───────────────────────────────────────────────────────
ARTIFACT_NEAR_MISS_MARGIN = 0.20  # metrics within 20% of threshold → NEAR_MISS

# ── Conda environment names ────────────────────────────────────────────────────
UTMOS_CONDA_ENV = "utmos"   # python 3.9 — used for WER, UTMOS, accent gates
