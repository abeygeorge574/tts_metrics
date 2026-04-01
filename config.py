import os

ROOT = "/Users/abey/Documents/tts_metrics"

# ── Gate base directories ──────────────────────────────────────────────────────
WER_BASE_DIR      = os.path.join(ROOT, "WER_PER_Production", "WER_TEST")
NISQA_BASE_DIR    = os.path.join(ROOT, "NISQA_prod")
UTMOS_BASE_DIR    = os.path.join(ROOT, "UTMOS")
SPEAKER_SIM_DIR   = os.path.join(ROOT, "speaker_similarity")
SER_BASE_DIR      = os.path.join(ROOT, "SER")
PITCH_BASE_DIR    = os.path.join(ROOT, "pitch")
DURATION_BASE_DIR = os.path.join(ROOT, "duration_ratio")
VAD_BASE_DIR      = os.path.join(ROOT, "pause_alignment")
AMPLITUDE_BASE_DIR = os.path.join(ROOT, "amplitude")
ACCENT_BASE_DIR   = os.path.join(ROOT, "accent_classification")

# ── Model / weights paths ──────────────────────────────────────────────────────
NISQA_REPO  = os.path.join(NISQA_BASE_DIR, "model")
NISQA_WEIGHT = os.path.join(NISQA_REPO, "weights", "nisqa.tar")

UTMOS_MODEL_DIR = os.path.join(UTMOS_BASE_DIR, "model", "simple")
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

# ── Speaker similarity thresholds ─────────────────────────────────────────────
SPEAKER_SIM_THRESHOLD = 0.75

# ── SER thresholds ─────────────────────────────────────────────────────────────
SER_CONFIDENCE_THRESHOLD = 0.5

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

# ── Accent thresholds ──────────────────────────────────────────────────────────
ACCENT_TARGET           = "american"
ACCENT_TARGET_THRESHOLD = 0.75
ACCENT_REFERENCES = {
    "american": os.path.join(ACCENT_BASE_DIR, "reference", "american.wav"),
    "british" : os.path.join(ACCENT_BASE_DIR, "reference", "british.wav"),
    "indian"  : os.path.join(ACCENT_BASE_DIR, "reference", "indian.wav"),
}

# ── Conda environment names ────────────────────────────────────────────────────
UTMOS_CONDA_ENV = "utmos"   # python 3.9 — used for WER, UTMOS, accent gates
