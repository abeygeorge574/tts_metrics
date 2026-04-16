"""
STS Gates — Multi-Episode Runner
Runs NISQA, Pitch, Amplitude, Artifact, Duration, VAD (and optionally SER)
on all episodes (EP21–EP35), comparing Converted_Segments vs Recorded_Segments.

Usage:
  python3 -u sts_gates_all_episodes.py

Env vars:
  EPISODES_DIR  - where episodes live (default: /Users/abey/Downloads)
  OUT_DIR       - where to save CSVs (default: /tmp/sts_gate_results)
  EP_START      - first episode number (default: 21)
  EP_END        - last episode number (default: 35)
  GATES         - comma-separated list of gates to run, e.g. "duration,pitch"
                  options: duration, pitch, amplitude, artifact, vad, nisqa, ser
                  default: all except ser
"""

import os, sys, re
import numpy as np
import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TTS_ROOT     = os.environ.get("TTS_ROOT", _SCRIPT_DIR)
sys.path.insert(0, TTS_ROOT)

EPISODES_DIR = os.environ.get("EPISODES_DIR", "/Users/abey/Downloads")
OUT_DIR      = os.environ.get("OUT_DIR", "/tmp/sts_gate_results")
EP_START     = int(os.environ.get("EP_START", 21))
EP_END       = int(os.environ.get("EP_END",   35))
_ALL_GATES   = {"duration", "pitch", "amplitude", "artifact", "vad", "nisqa", "ser"}
_gates_env   = os.environ.get("GATES", "")
RUN_GATES    = {g.strip() for g in _gates_env.split(",") if g.strip()} if _gates_env else (_ALL_GATES - {"ser"})
print(f"Gates to run: {sorted(RUN_GATES)}")

os.makedirs(OUT_DIR, exist_ok=True)

import config
if "nisqa"     in RUN_GATES: from gates import gate_nisqa
if "pitch"     in RUN_GATES: from gates import gate_pitch
if "amplitude" in RUN_GATES: from gates import gate_amplitude
if "artifact"  in RUN_GATES: from gates import gate_artifact
if "duration"  in RUN_GATES: from gates import gate_duration
if "vad"       in RUN_GATES: from gates import gate_vad

HAS_SER = False
if "ser" in RUN_GATES:
    try:
        from gates import gate_ser
        HAS_SER = True
    except Exception as e:
        print(f"SER import failed ({e}) — skipping")

# ── Load models ───────────────────────────────────────────────────────────────
nisqa_state = None
if "nisqa" in RUN_GATES:
    print("Loading NISQA model...")
    nisqa_state = gate_nisqa.load_model()
    print("NISQA ready.")

ser_state = None
if HAS_SER:
    print("Loading SER model (MERaLiON — may take ~30s)...")
    try:
        ser_state = gate_ser.load_model()
        print("SER ready.")
    except Exception as e:
        print(f"SER load failed ({e}) — skipping")
        HAS_SER = False

# ── Thresholds ────────────────────────────────────────────────────────────────
# NISQA
_NQ = config.NISQA_THRESHOLDS

# Pitch
_P_DELTA     = config.PITCH_MEDIAN_THRESHOLD
_P_DELTA_NM  = config.PITCH_MEDIAN_THRESHOLD * (1 + config.PITCH_NEAR_MISS_MARGIN)
_P_STD_ABS   = config.PITCH_STD_ABS_THRESHOLD
_P_STD_ABS_NM= config.PITCH_STD_ABS_THRESHOLD * (1 - config.PITCH_NEAR_MISS_MARGIN)
_P_STD_RATIO = config.PITCH_STD_RATIO_THRESHOLD
_P_STD_RT_NM = config.PITCH_STD_RATIO_THRESHOLD * (1 - config.PITCH_NEAR_MISS_MARGIN)

# Amplitude
_A_LUFS   = config.LUFS_TOLERANCE
_A_LRA    = config.LRA_TOLERANCE
_A_CENT   = config.CENTROID_TOLERANCE
_A_NM_MRG = config.AMPLITUDE_NEAR_MISS_MARGIN

# Duration
_D_LOW  = 1.0 - config.DURATION_TOLERANCE
_D_HIGH = 1.0 + config.DURATION_TOLERANCE

# VAD
_V_POS     = config.POSITION_OFFSET_THRESHOLD
_V_DUR_MIN = config.VAD_DURATION_RATIO_MIN
_V_DUR_MAX = config.VAD_DURATION_RATIO_MAX
_V_NM_MRG  = config.VAD_NEAR_MISS_MARGIN

# Artifact
_ART_HNR_THR   = config.ARTIFACT_HNR_ABS_THRESHOLD
_ART_HNR_NM    = config.ARTIFACT_HNR_ABS_THRESHOLD * (1 - config.ARTIFACT_NEAR_MISS_MARGIN)
_ART_SIL_FAIL  = config.ARTIFACT_SILENCE_FAIL_DB
_ART_SIL_WARN  = config.ARTIFACT_SILENCE_WARN_DB
_ART_COMB_THR  = config.ARTIFACT_COMBINED_THRESHOLD
_ART_COMB_NM   = config.ARTIFACT_COMBINED_THRESHOLD * (1 + config.ARTIFACT_NEAR_MISS_MARGIN)

# ── Episode discovery ─────────────────────────────────────────────────────────
def is_valid_episode(name):
    m = re.match(r'^EP(\d+)$', name)
    if not m: return False
    return EP_START <= int(m.group(1)) <= EP_END

episodes = sorted(
    [d for d in os.listdir(EPISODES_DIR)
     if is_valid_episode(d) and os.path.isdir(os.path.join(EPISODES_DIR, d))],
    key=lambda d: int(re.search(r'\d+', d).group())
)
print(f"\nFound {len(episodes)} episodes: {episodes}\n")

# ── Verdict helpers ───────────────────────────────────────────────────────────
def _verdict3(val, thr_pass, thr_nm, higher_is_better=True):
    """Return PASS/NEAR_MISS/FAIL. higher_is_better=True means val>=thr_pass=PASS."""
    if val is None: return "SKIP"
    if higher_is_better:
        if val >= thr_pass: return "PASS"
        if val >= thr_nm:   return "NEAR_MISS"
        return "FAIL"
    else:
        if val <= thr_pass: return "PASS"
        if val <= thr_nm:   return "NEAR_MISS"
        return "FAIL"

_NQ_DELTA = config.NISQA_DELTA_THRESHOLDS   # e.g. MOS delta ≥ -0.5 to pass
_NQ_REF   = config.NISQA_REF_THRESHOLDS     # permissive floor; below → degraded ref

def nisqa_verdict(s, ref_s=None):
    """
    PASS      — absolute passes all dims
    REVIEW    — absolute fails but delta vs ref passes (STS no worse than original)
    NEAR_MISS — absolute barely fails (all dims within 10%), no usable delta
    FAIL      — absolute fails hard AND delta fails (or no ref)
    """
    if any(s.get(k) is None for k in _NQ):
        return "SKIP"
    abs_fails = [k for k, thr in _NQ.items() if s.get(k, 0) < thr]
    if not abs_fails:
        return "PASS"

    # Check if we have a usable reference (ref not below quality floor)
    ref_usable = (
        ref_s is not None and
        all(ref_s.get(k, 0) >= _NQ_REF.get(k, 0) for k in _NQ_REF)
    )
    if ref_usable:
        deltas = {k: s[k] - ref_s[k] for k in _NQ}
        delta_fails = [k for k in _NQ_DELTA if deltas.get(k, -999) < _NQ_DELTA[k]]
        if not delta_fails:
            return "REVIEW"   # absolute fails but STS no worse than original recording

    # Near-miss: all failing absolute dims within 10% of threshold
    if all(s.get(k, 0) >= _NQ[k] * 0.90 for k in abs_fails):
        return "NEAR_MISS"
    return "FAIL"

def pitch_verdict(tts_med, tts_std, ref_med, ref_std):
    if tts_med is None or tts_std is None: return "SKIP"
    delta     = abs(tts_med - ref_med) if ref_med is not None else None
    std_ratio = (tts_std / ref_std)    if (ref_std and ref_std > 0) else None
    reg_v  = _verdict3(delta,     _P_DELTA,     _P_DELTA_NM,  higher_is_better=False) if delta     is not None else "NO_REF"
    abs_v  = _verdict3(tts_std,   _P_STD_ABS,   _P_STD_ABS_NM)
    rat_v  = _verdict3(std_ratio, _P_STD_RATIO,  _P_STD_RT_NM) if std_ratio is not None else "NO_REF"
    for v in [reg_v, abs_v, rat_v]:
        if v == "FAIL": return "FAIL"
    if "NEAR_MISS" in [reg_v, abs_v, rat_v]: return "NEAR_MISS"
    if "NO_REF"    in [reg_v, abs_v, rat_v]: return "NO_REF"
    return "PASS"

def amplitude_verdict(tts, ref):
    tts_lufs, tts_lra, tts_cent, tts_peak, tts_clip = tts
    ref_lufs, ref_lra, ref_cent, _,          _       = ref
    if tts_lufs is None: return "SKIP"
    if ref_lufs is None: return "NO_REF"
    lufs_d = abs(tts_lufs - ref_lufs)
    lra_d  = abs(tts_lra  - ref_lra)  if (tts_lra  and ref_lra)  else 0.0
    # Centroid excluded from pass/fail for STS: cross-speaker voice change causes
    # systematic +1500 Hz centroid shift (different voice timbre, not a quality failure).
    # Centroid is kept as a diagnostic column only.
    clip   = tts_clip if tts_clip is not None else 0.0
    if clip >= 0.001: return "FAIL"   # hard clip fail
    fails = sum([lufs_d > _A_LUFS, lra_d > _A_LRA])
    nms   = sum([lufs_d > _A_LUFS * (1 - _A_NM_MRG),
                 lra_d  > _A_LRA  * (1 - _A_NM_MRG)])
    if fails >= 2: return "FAIL"
    if fails == 1 or nms >= 2: return "NEAR_MISS"
    return "PASS"

def duration_verdict(ratio):
    if ratio is None: return "NO_REF"
    if _D_LOW <= ratio <= _D_HIGH: return "PASS"
    return "FAIL"

def vad_verdict(ref_pauses, tts_pauses, ref_dur):
    count_thr = min(config.PAUSE_COUNT_THRESHOLD,
                    max(2, ref_dur * config.PAUSE_COUNT_RATE_THRESHOLD)) if ref_dur else 4
    ref_n, tts_n = len(ref_pauses), len(tts_pauses)
    cnt_delta = abs(ref_n - tts_n)
    count_pass = cnt_delta <= count_thr

    if ref_pauses and tts_pauses:
        ref_mids = sorted((p["start"] + p["end"]) / 2 for p in ref_pauses)
        tts_mids = sorted((p["start"] + p["end"]) / 2 for p in tts_pauses)
        n = min(len(ref_mids), len(tts_mids))
        offsets  = [abs(ref_mids[i] - tts_mids[i]) for i in range(n)]
        pos_off  = float(np.median(offsets)) if offsets else 0.0
        ref_durs = [p["duration"] for p in ref_pauses]
        tts_durs = [p["duration"] for p in tts_pauses]
        n2 = min(len(ref_durs), len(tts_durs))
        ratios   = [tts_durs[i] / ref_durs[i] for i in range(n2) if ref_durs[i] > 0]
        dur_rat  = float(np.median(ratios)) if ratios else 1.0
    else:
        pos_off, dur_rat = 0.0, 1.0

    pos_pass = pos_off <= _V_POS
    dur_pass = _V_DUR_MIN <= dur_rat <= _V_DUR_MAX

    nm_cnt = cnt_delta <= count_thr * (1 + _V_NM_MRG)
    nm_pos = pos_off   <= _V_POS   * (1 + _V_NM_MRG)

    fails = sum([not count_pass, not pos_pass, not dur_pass])
    nms   = sum([not nm_cnt, not nm_pos])
    if fails >= 2: verdict = "FAIL"
    elif fails == 1 or nms >= 2: verdict = "NEAR_MISS"
    else: verdict = "PASS"
    return verdict, cnt_delta, pos_off, dur_rat

def artifact_verdict(hnr_d, pause_d, sa_d):
    hnr    = hnr_d.get("hnr_mean")
    med_db = pause_d.get("median_db")
    comb   = sa_d.get("combined")
    hnr_v  = _verdict3(hnr,    _ART_HNR_THR,  _ART_HNR_NM)
    paus_v = _verdict3(med_db, _ART_SIL_FAIL,  _ART_SIL_WARN, higher_is_better=False) if med_db is not None else "SKIP"
    sa_v   = _verdict3(comb,   _ART_COMB_THR,  _ART_COMB_NM,  higher_is_better=False) if comb   is not None else "SKIP"
    for v in [hnr_v, paus_v, sa_v]:
        if v == "FAIL": return "FAIL"
    if "NEAR_MISS" in [hnr_v, paus_v, sa_v]: return "NEAR_MISS"
    if all(v == "SKIP" for v in [hnr_v, paus_v, sa_v]): return "SKIP"
    return "PASS"

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows = []

for ep_name in episodes:
    ep_path = os.path.join(EPISODES_DIR, ep_name)
    print(f"\n{'='*60}\n  {ep_name}\n{'='*60}")

    chars = sorted(d for d in os.listdir(ep_path)
                   if os.path.isdir(os.path.join(ep_path, d)))

    for char in chars:
        conv_dir = os.path.join(ep_path, char, "Converted_Segments")
        rec_dir  = os.path.join(ep_path, char, "Recorded_Segments")
        if not os.path.isdir(conv_dir):
            continue

        wav_files = sorted(f for f in os.listdir(conv_dir) if f.endswith(".wav"))
        char_rows = []

        for wav_file in wav_files:
            conv_path = os.path.join(conv_dir, wav_file)
            rec_path  = os.path.join(rec_dir, wav_file) if os.path.isdir(rec_dir) else None
            has_ref   = bool(rec_path and os.path.exists(rec_path))

            row = {
                "Episode":   ep_name,
                "Character": char,
                "Sample":    os.path.splitext(wav_file)[0],
                "has_ref":   has_ref,
            }

            # ── Duration ──────────────────────────────────────────────────────
            if "duration" in RUN_GATES:
                try:
                    tts_dur = gate_duration.get_duration(conv_path)
                    ref_dur = gate_duration.get_duration(rec_path) if has_ref else None
                    ratio   = round(tts_dur / ref_dur, 4) if ref_dur else None
                    row.update({
                        "dur_tts_s": round(tts_dur, 2),
                        "dur_ref_s": round(ref_dur, 2) if ref_dur else None,
                        "dur_ratio": ratio,
                        "dur_pass":  duration_verdict(ratio),
                    })
                except Exception:
                    row.update({"dur_tts_s": None, "dur_ref_s": None,
                                "dur_ratio": None, "dur_pass": "ERR"})

            # ── Pitch ─────────────────────────────────────────────────────────
            if "pitch" in RUN_GATES:
                try:
                    tts_med, tts_std, tts_rng, tts_vr = gate_pitch.compute_pitch(conv_path)
                    if has_ref:
                        ref_med, ref_std, ref_rng, ref_vr = gate_pitch.compute_pitch(rec_path)
                    else:
                        ref_med = ref_std = ref_rng = ref_vr = None
                    f0_delta  = round(abs(tts_med - ref_med), 2) if (tts_med and ref_med) else None
                    std_ratio = round(tts_std / ref_std, 4)      if (tts_std and ref_std and ref_std > 0) else None
                    row.update({
                        "pitch_tts_med_hz":  round(tts_med, 2) if tts_med else None,
                        "pitch_tts_std_hz":  round(tts_std, 2) if tts_std else None,
                        "pitch_ref_med_hz":  round(ref_med, 2) if ref_med else None,
                        "pitch_ref_std_hz":  round(ref_std, 2) if ref_std else None,
                        "pitch_f0_delta_hz": f0_delta,
                        "pitch_std_ratio":   std_ratio,
                        "pitch_pass":        pitch_verdict(tts_med, tts_std, ref_med, ref_std),
                    })
                except Exception:
                    row.update({"pitch_tts_med_hz": None, "pitch_tts_std_hz": None,
                                "pitch_ref_med_hz": None, "pitch_ref_std_hz": None,
                                "pitch_f0_delta_hz": None, "pitch_std_ratio": None,
                                "pitch_pass": "ERR"})

            # ── Amplitude ─────────────────────────────────────────────────────
            if "amplitude" in RUN_GATES:
                try:
                    tts_amp = gate_amplitude.analyze_audio(conv_path)
                    ref_amp = gate_amplitude.analyze_audio(rec_path) if has_ref else (None,) * 5
                    lufs_d = round(abs(tts_amp[0] - ref_amp[0]), 2) if (tts_amp[0] is not None and ref_amp[0] is not None) else None
                    lra_d  = round(abs(tts_amp[1] - ref_amp[1]), 2) if (tts_amp[1] is not None and ref_amp[1] is not None) else None
                    cent_d = round(abs(tts_amp[2] - ref_amp[2]), 0) if (tts_amp[2] is not None and ref_amp[2] is not None) else None
                    row.update({
                        "amp_tts_lufs":      round(tts_amp[0], 2) if tts_amp[0] is not None else None,
                        "amp_ref_lufs":      round(ref_amp[0], 2) if ref_amp[0] is not None else None,
                        "amp_lufs_delta":    lufs_d,
                        "amp_tts_lra":       round(tts_amp[1], 2) if tts_amp[1] is not None else None,
                        "amp_ref_lra":       round(ref_amp[1], 2) if ref_amp[1] is not None else None,
                        "amp_lra_delta":     lra_d,
                        "amp_tts_cent_hz":   round(tts_amp[2], 0) if tts_amp[2] is not None else None,
                        "amp_ref_cent_hz":   round(ref_amp[2], 0) if ref_amp[2] is not None else None,
                        "amp_cent_delta_hz": cent_d,
                        "amp_tts_peak_dbfs": round(tts_amp[3], 2) if tts_amp[3] is not None else None,
                        "amp_clip_rate_pct": round(tts_amp[4] * 100, 4) if tts_amp[4] is not None else None,
                        "amp_pass":          amplitude_verdict(tts_amp, ref_amp),
                    })
                except Exception:
                    row.update({"amp_tts_lufs": None, "amp_ref_lufs": None,
                                "amp_lufs_delta": None, "amp_pass": "ERR"})

            # ── Artifact ──────────────────────────────────────────────────────
            if "artifact" in RUN_GATES:
                try:
                    hnr_d   = gate_artifact.compute_hnr(conv_path)
                    pause_d = gate_artifact.compute_pause_median_db(conv_path)
                    sa_d    = gate_artifact.compute_spectral_artifact_score(conv_path)
                    row.update({
                        "art_hnr_db":       round(hnr_d.get("hnr_mean"), 2)    if hnr_d.get("hnr_mean")    is not None else None,
                        "art_hnr_vfrac":    round(hnr_d.get("hnr_voiced_frac", 0), 3),
                        "art_pause_med_db": round(pause_d.get("median_db"), 2) if pause_d.get("median_db") is not None else None,
                        "art_sa_h1h2":      round(sa_d.get("h1h2"), 4)         if sa_d.get("h1h2")         is not None else None,
                        "art_sa_cepmidq":   round(sa_d.get("cep_midq"), 4)     if sa_d.get("cep_midq")     is not None else None,
                        "art_sa_sfm_hf":    round(sa_d.get("sfm_hf"), 4)       if sa_d.get("sfm_hf")       is not None else None,
                        "art_sa_combined":  round(sa_d.get("combined"), 4)     if sa_d.get("combined")     is not None else None,
                        "art_pass":         artifact_verdict(hnr_d, pause_d, sa_d),
                    })
                except Exception:
                    row.update({"art_hnr_db": None, "art_pause_med_db": None,
                                "art_sa_combined": None, "art_pass": "ERR"})

            # ── VAD ───────────────────────────────────────────────────────────
            if "vad" in RUN_GATES:
                try:
                    tts_paus, tts_dur_vad = gate_vad.get_pauses(conv_path)
                    if has_ref:
                        ref_paus, ref_dur_vad = gate_vad.get_pauses(rec_path)
                        vad_v, cnt_d, pos_off, dur_rat = vad_verdict(ref_paus, tts_paus, ref_dur_vad)
                        row.update({
                            "vad_ref_pauses": len(ref_paus),
                            "vad_tts_pauses": len(tts_paus),
                            "vad_cnt_delta":  cnt_d,
                            "vad_pos_off_s":  round(pos_off, 3),
                            "vad_dur_ratio":  round(dur_rat, 3),
                            "vad_pass":       vad_v,
                        })
                    else:
                        row.update({"vad_ref_pauses": None, "vad_tts_pauses": len(tts_paus),
                                    "vad_cnt_delta": None, "vad_pos_off_s": None,
                                    "vad_dur_ratio": None, "vad_pass": "NO_REF"})
                except Exception:
                    row.update({"vad_ref_pauses": None, "vad_tts_pauses": None,
                                "vad_cnt_delta": None, "vad_pos_off_s": None,
                                "vad_dur_ratio": None, "vad_pass": "ERR"})

            # ── NISQA ─────────────────────────────────────────────────────────
            if "nisqa" in RUN_GATES:
                try:
                    nq = gate_nisqa.score_single_file(conv_path, nisqa_state["nisqa_weight"])
                    ref_nq = None
                    if has_ref:
                        try:
                            ref_nq = gate_nisqa.score_single_file(rec_path, nisqa_state["nisqa_weight"])
                        except Exception:
                            pass
                    delta_mos = round(nq["MOS"] - ref_nq["MOS"], 3) if ref_nq else None
                    row.update({
                        "nisqa_mos":           round(nq["MOS"], 3),
                        "nisqa_noisiness":     round(nq["Noisiness"], 3),
                        "nisqa_discontinuity": round(nq["Discontinuity"], 3),
                        "nisqa_coloration":    round(nq["Coloration"], 3),
                        "nisqa_loudness":      round(nq["Loudness"], 3),
                        "nisqa_ref_mos":       round(ref_nq["MOS"], 3) if ref_nq else None,
                        "nisqa_delta_mos":     delta_mos,
                        "nisqa_pass":          nisqa_verdict(nq, ref_nq),
                    })
                except Exception:
                    row.update({"nisqa_mos": None, "nisqa_noisiness": None,
                                "nisqa_discontinuity": None, "nisqa_coloration": None,
                                "nisqa_loudness": None, "nisqa_ref_mos": None,
                                "nisqa_delta_mos": None, "nisqa_pass": "ERR"})

            # ── SER ───────────────────────────────────────────────────────────
            if HAS_SER and ser_state:
                try:
                    tts_em = gate_ser.get_emotion(conv_path, ser_state["model"], ser_state["processor"])
                    ref_em = gate_ser.get_emotion(rec_path, ser_state["model"], ser_state["processor"]) if has_ref else (None,) * 7
                    ar_delta = round(abs(tts_em[5] - ref_em[5]), 3) if (tts_em[5] is not None and ref_em[5] is not None) else None
                    em_match = (tts_em[0] == ref_em[0]) if (tts_em[0] and ref_em[0]) else None
                    top2_overlap = bool(tts_em[2] and ref_em[2] and
                                        ((tts_em[2] in {ref_em[0], ref_em[2]}) or (ref_em[2] in {tts_em[0], tts_em[2]})))
                    if em_match is None:   ser_v = "NO_REF"
                    elif em_match:         ser_v = "PASS"
                    elif top2_overlap:     ser_v = "NEAR_MISS"
                    else:                  ser_v = "FAIL"
                    row.update({
                        "ser_ref_top1":      ref_em[0],
                        "ser_tts_top1":      tts_em[0],
                        "ser_tts_conf":      round(tts_em[1], 3) if tts_em[1] else None,
                        "ser_ref_arousal":   round(ref_em[5], 3) if ref_em[5] is not None else None,
                        "ser_tts_arousal":   round(tts_em[5], 3) if tts_em[5] is not None else None,
                        "ser_arousal_delta": ar_delta,
                        "ser_pass":          ser_v,
                    })
                except Exception:
                    row.update({"ser_ref_top1": None, "ser_tts_top1": None, "ser_pass": "ERR"})

            char_rows.append(row)
            all_rows.append(row)

        # Per-character log
        df_c    = pd.DataFrame(char_rows)
        g_check = [("dur","dur_pass"),("pitch","pitch_pass"),("amp","amp_pass"),
                   ("art","art_pass"),("vad","vad_pass"),("nisqa","nisqa_pass")]
        if HAS_SER: g_check.append(("ser","ser_pass"))
        parts = [f"{g}={int((df_c[col]=='PASS').sum())}/{len(df_c)}"
                 for g, col in g_check if col in df_c.columns]
        print(f"  [{char}] {' | '.join(parts)}")

# ── Save all-segments CSV ─────────────────────────────────────────────────────
all_df   = pd.DataFrame(all_rows)
tag      = "_".join(sorted(RUN_GATES))   # e.g. "duration" or "pitch_vad"
seg_path = os.path.join(OUT_DIR, f"sts_{tag}_segments.csv")
all_df.to_csv(seg_path, index=False)
print(f"\nSaved segments: {seg_path}  ({len(all_df):,} rows)")

# ── Episode summary ───────────────────────────────────────────────────────────
GATE_COLS = [
    ("Duration",  "dur_pass"),
    ("Pitch",     "pitch_pass"),
    ("Amplitude", "amp_pass"),
    ("Artifact",  "art_pass"),
    ("VAD",       "vad_pass"),
    ("NISQA",     "nisqa_pass"),
]
if HAS_SER:
    GATE_COLS.append(("SER", "ser_pass"))

ep_rows = []
for ep in episodes:
    ep_df = all_df[all_df["Episode"] == ep]
    row = {"Episode": ep, "Total_segs": len(ep_df)}
    for gate_name, col in GATE_COLS:
        if col in ep_df.columns:
            n_pass   = (ep_df[col] == "PASS").sum()
            n_scored = ep_df[col].notna().sum()
            row[f"{gate_name}_pass"]  = int(n_pass)
            row[f"{gate_name}_total"] = int(n_scored)
            row[f"{gate_name}_rate%"] = round(n_pass / n_scored * 100, 1) if n_scored > 0 else None
    ep_rows.append(row)

ep_sum   = pd.DataFrame(ep_rows)
sum_path = os.path.join(OUT_DIR, f"sts_{tag}_summary.csv")
ep_sum.to_csv(sum_path, index=False)
print(f"Saved summary:  {sum_path}")

# ── Overall ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("OVERALL PASS RATES — ALL EPISODES")
print(f"{'='*60}")
for gate_name, col in GATE_COLS:
    if col in all_df.columns:
        n_pass   = (all_df[col] == "PASS").sum()
        n_scored = all_df[col].notna().sum()
        rate     = n_pass / n_scored * 100 if n_scored > 0 else 0
        print(f"  {gate_name:12s}: {n_pass:,}/{n_scored:,} = {rate:.1f}%")

print(f"\nOutputs:")
print(f"  {seg_path}")
print(f"  {sum_path}")
