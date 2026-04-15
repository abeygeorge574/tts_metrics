"""
Helper: load functions from TTS gates by absolute file path.
Avoids naming conflict between tts_metrics/gates/ and tts_metrics/sts/gates/.
"""
from __future__ import annotations

import os
import importlib.util

_TTS_GATES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "gates"
)


def _load_tts_gate(gate_filename: str):
    """Load a TTS gate module by filename (e.g. 'gate_pitch.py')."""
    path   = os.path.join(_TTS_GATES, gate_filename)
    name   = f"_tts_{gate_filename.replace('.py', '')}"
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Lazily loaded TTS modules ─────────────────────────────────────────────────
_cache = {}

def tts_gate(gate_filename: str):
    if gate_filename not in _cache:
        _cache[gate_filename] = _load_tts_gate(gate_filename)
    return _cache[gate_filename]


# ── Convenience accessors ─────────────────────────────────────────────────────
def get_compute_pitch():
    return tts_gate("gate_pitch.py").compute_pitch

def get_score_single_file():
    return tts_gate("gate_nisqa.py").score_single_file

def get_compute_hnr():
    return tts_gate("gate_artifact.py").compute_hnr

def get_compute_pause_median_db():
    return tts_gate("gate_artifact.py").compute_pause_median_db

def get_compute_spectral_artifact_score():
    return tts_gate("gate_artifact.py").compute_spectral_artifact_score

def get_ser_functions():
    m = tts_gate("gate_ser.py")
    return m.load_model, m.get_emotion

def get_speaker_sim_functions():
    m = tts_gate("gate_speaker_sim.py")
    return m.load_model, m.get_embedding, m.cosine_similarity
