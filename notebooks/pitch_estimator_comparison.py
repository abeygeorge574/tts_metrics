"""
Pitch estimator comparison — isolated experiment.

Runs four approaches on the same audio files and reports F0 estimates:
  1. pyin + median          (current gate approach)
  2. pyin + voiced_probs>0.8 filter + median
  3. CREPE (deep learning)
  4. PRAAT via parselmouth

Test files: 3 Hindi reference segments + their chatterbox_cloned and f5tts_cloned outputs.
That gives 9 files, covering reference audio and two cloning models.

Run:
    python notebooks/pitch_estimator_comparison.py
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REF_DIR       = os.path.join(ROOT, "data", "hindi_eval", "reference")
MODELS_DIR    = os.path.join(ROOT, "data", "hindi_eval", "models")

# Use 3 segments that had interesting verdicts in the gate run
TEST_SEGMENTS = [
    "PSTSBF-34931-S375335-R 2.wav",   # chatterbox register fail (Δ=45 Hz)
    "PSTSCX-34931-S375354-R.wav",     # chatterbox register fail (Δ=40 Hz)
    "PSTSHI-34931-S375341-R.wav",     # chatterbox PASS (Δ=4 Hz)
]
TEST_MODELS   = ["chatterbox_cloned", "f5tts_cloned"]

_SR = 16000


# ── Approach 1: pyin + median (current gate) ──────────────────────────────────
def f0_pyin_median(path):
    import librosa
    audio, sr = librosa.load(path, sr=_SR, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=1024,
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None
    return round(float(np.median(voiced_f0)), 1)


# ── Approach 2: pyin + voiced_probs > 0.8 filter + median ────────────────────
def f0_pyin_conffilter(path, conf_threshold=0.8):
    import librosa
    audio, sr = librosa.load(path, sr=_SR, mono=True)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=1024,
    )
    high_conf = voiced_flag & (voiced_probs > conf_threshold)
    voiced_f0 = f0[high_conf]
    if len(voiced_f0) == 0:
        # fallback to standard voiced frames if nothing passes threshold
        voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        return None
    return round(float(np.median(voiced_f0)), 1)


# ── Approach 3: CREPE (runs via utmos Python 3.9 env) ────────────────────────
_UTMOS_PYTHON = os.path.expanduser("~/miniconda3/envs/utmos/bin/python")

_CREPE_WORKER = '''
import sys, json, numpy as np
path = sys.argv[1]
import crepe, soundfile as sf
audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
time, frequency, confidence, _ = crepe.predict(
    audio, sr, viterbi=True, step_size=10, verbose=0)
voiced = confidence > 0.7
if voiced.sum() == 0:
    print("null")
else:
    print(round(float(np.median(frequency[voiced])), 1))
'''

def f0_crepe(path):
    try:
        import subprocess, tempfile
        script_path = os.path.join("/tmp/claude", "_crepe_worker.py")
        os.makedirs("/tmp/claude", exist_ok=True)
        with open(script_path, "w") as f:
            f.write(_CREPE_WORKER)
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"
        env["MPLBACKEND"]   = "Agg"
        os.makedirs("/tmp/claude/matplotlib", exist_ok=True)
        result = subprocess.run(
            [_UTMOS_PYTHON, script_path, os.path.abspath(path)],
            capture_output=True, text=True, timeout=120, env=env
        )
        # take first line only — TF prints "No supported GPU was found." to stdout
        out = result.stdout.strip().split('\n')[0].strip()
        if not out or out == "null":
            return None
        return float(out)
    except Exception as e:
        return f"ERR:{str(e)[:20]}"


# ── Approach 4: PRAAT via parselmouth ────────────────────────────────────────
def f0_praat(path):
    try:
        import parselmouth
        snd   = parselmouth.Sound(path)
        pitch = snd.to_pitch(
            time_step=0.01,
            pitch_floor=60.0,    # Hz — lower bound (covers male voices)
            pitch_ceiling=600.0, # Hz — upper bound (covers high female)
        )
        f0_values = pitch.selected_array["frequency"]
        voiced_f0 = f0_values[f0_values > 0]  # 0 = unvoiced frame in PRAAT
        if len(voiced_f0) == 0:
            return None
        return round(float(np.median(voiced_f0)), 1)
    except ImportError:
        return "parselmouth not installed"
    except Exception as e:
        return f"ERROR: {e}"


# ── Run comparison ─────────────────────────────────────────────────────────────
def run():
    if not os.environ.get("NUMBA_CACHE_DIR"):
        os.environ["NUMBA_CACHE_DIR"] = "/tmp/claude/numba"
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

    rows = []   # (label, pyin_med, pyin_conf, crepe, praat)

    all_files = []
    for seg in TEST_SEGMENTS:
        all_files.append(("REF", os.path.join(REF_DIR, seg), seg))
        for model in TEST_MODELS:
            p = os.path.join(MODELS_DIR, model, seg)
            if os.path.exists(p):
                all_files.append((model, p, seg))

    col_w = 30
    print(f"\n{'Label':<22} {'Segment':<36}  {'pyin+med':>10}  {'pyin+conf0.8':>13}  {'CREPE':>10}  {'PRAAT':>10}")
    print("-" * 107)

    for label, path, seg in all_files:
        seg_short = seg[:34]
        pm  = f0_pyin_median(path)
        pc  = f0_pyin_conffilter(path)
        cr  = f0_crepe(path)
        pr  = f0_praat(path)

        def fmt(v):
            if v is None:
                return "—"
            if isinstance(v, str):
                return v[:10]
            return f"{v} Hz"

        print(f"  {label:<20} {seg_short:<36}  {fmt(pm):>10}  {fmt(pc):>13}  {fmt(cr):>10}  {fmt(pr):>10}")
        rows.append({"label": label, "segment": seg, "pyin_median": pm, "pyin_conf08": pc, "crepe": cr, "praat": pr})

    print("\n\n--- DELTA vs REFERENCE (|TTS - REF| per segment) ---\n")
    print(f"{'Model':<22} {'Segment':<36}  {'pyin+med':>10}  {'pyin+conf0.8':>13}  {'CREPE':>10}  {'PRAAT':>10}")
    print("-" * 107)

    ref_rows = {r["segment"]: r for r in rows if r["label"] == "REF"}
    for r in rows:
        if r["label"] == "REF":
            continue
        ref = ref_rows.get(r["segment"])
        if not ref:
            continue

        def delta(a, b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return f"{abs(a - b):.1f} Hz"
            return "—"

        seg_short = r["segment"][:34]
        print(f"  {r['label']:<20} {seg_short:<36}  "
              f"{delta(r['pyin_median'], ref['pyin_median']):>10}  "
              f"{delta(r['pyin_conf08'], ref['pyin_conf08']):>13}  "
              f"{delta(r['crepe'], ref['crepe']):>10}  "
              f"{delta(r['praat'], ref['praat']):>10}")

    print("\n--- AGREEMENT BETWEEN METHODS (on REF files) ---\n")
    print(f"{'Segment':<36}  {'pyin+med':>10}  {'pyin+conf0.8':>13}  {'CREPE':>10}  {'PRAAT':>10}")
    print("-" * 85)
    for r in rows:
        if r["label"] != "REF":
            continue
        def fmt(v):
            if v is None: return "—"
            if isinstance(v, str): return v[:10]
            return f"{v} Hz"
        seg_short = r["segment"][:34]
        print(f"  {seg_short:<34}  {fmt(r['pyin_median']):>10}  {fmt(r['pyin_conf08']):>13}  {fmt(r['crepe']):>10}  {fmt(r['praat']):>10}")

    print("\n")
    print("HOW TO READ:")
    print("  Agreement section: all methods on the same REF file — closer together = more consistent.")
    print("  Delta section: |TTS - REF| — lower = better pitch match.")
    print("  If all methods agree on the delta ranking, the gate verdict is robust.")
    print("  If they disagree significantly (>15 Hz delta difference), the estimator choice matters.")


if __name__ == "__main__":
    run()
