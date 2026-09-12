"""
========================================================================
SIH26104 - Integration Adapter Layer (by Member 6)
========================================================================
Purpose: Bridge the REAL functions built by Member 1, 3, 5 to the exact
input/output shape that the dashboard (Member 4's SIH UI.py) expects
from its mock_* functions.

HOW TO USE:
1. Drop this file in the project ROOT folder (same level as SIH UI.py)
2. In "SIH UI.py", add this import near the top:
       from integration_adapters import (
           real_deepfake_detector,
           real_speaker_verifier,
           real_risk_engine,
       )
3. In "SIH UI.py", find every place that CALLS:
       mock_deepfake_detector(y, sr)      -> replace with real_deepfake_detector(y, sr)
       mock_speaker_verifier(y, sr, spk)  -> replace with real_speaker_verifier(y, sr, spk)
       mock_risk_engine(...)              -> replace with real_risk_engine(...)
   (DO NOT delete the mock_ function definitions - keep them as fallback/reference)
4. Run: streamlit run "SIH UI.py"

NOTE: Adjust REGISTERED_VOICES dict below once Member 3 gives final
reference audio file paths.
========================================================================
"""

import os
import tempfile
import numpy as np
import soundfile as sf

# ---- Real module imports (adjust these paths/names if your repo folder
#      names differ slightly) ----
from deepfake_detection.detector import detect_voice
from risk_engine.risk_engine import calculate_risk
from risk_engine.prevention import take_action

# TODO-MEMBER6: once Member 3 sends the fixed verify_speaker() function,
# uncomment this import and remove the placeholder function below.
# from speaker_verification.verifier import verify_speaker


# Map of "claimed speaker name" -> path of their registered reference audio.
# Fill this in with real files once available (e.g. CEO's registered voice).
REGISTERED_VOICES = {
    "CEO": "speaker_verification/reference_audio/ceo_reference.wav",
    # add more registered speakers here
}


def _save_temp_wav(y: np.ndarray, sr: int) -> str:
    """Helper: saves an in-memory waveform to a temp .wav file, because
    detect_voice() needs a file path, not a raw array."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return tmp.name


def real_deepfake_detector(y: np.ndarray, sr: int) -> dict:
    """
    Drop-in replacement for mock_deepfake_detector(y, sr).
    Wraps Member 1's detect_voice(audio_path).
    """
    audio_path = _save_temp_wav(y, sr)
    try:
        result = detect_voice(audio_path)
    finally:
        os.remove(audio_path)

    return {
        "authenticity_score": round(result["authentic_probability"] * 100, 1),
        "clone_probability": round(result["synthetic_probability"] * 100, 1),
        "model_confidence": round(result.get("model_confidence", 0) * 100, 1),
    }


def real_speaker_verifier(y: np.ndarray, sr: int, claimed_speaker: str) -> dict:
    """
    Drop-in replacement for mock_speaker_verifier(y, sr, claimed_speaker).
    TODO-MEMBER6: wire this to Member 3's real verify_speaker() once she
    sends the fixed (non-hardcoded-path) version.
    """
    reference_path = REGISTERED_VOICES.get(claimed_speaker)

    if reference_path is None or not os.path.exists(reference_path):
        # Unknown / not registered speaker -> low match, matches mock behavior
        match_score = 20.0
    else:
        test_path = _save_temp_wav(y, sr)
        try:
            # --- PLACEHOLDER until real function is wired in ---
            # result = verify_speaker(test_path, reference_path)
            # match_score = round(result["speaker_match_score"] * 100, 1)
            match_score = 50.0  # temporary neutral value, REPLACE ABOVE
        finally:
            os.remove(test_path)

    return {
        "speaker_match_score": match_score,
        "verified": match_score >= 75,
    }


def real_risk_engine(authenticity_score: float, clone_probability: float,
                      speaker_match_score: float, verified: bool) -> dict:
    """
    Drop-in replacement for mock_risk_engine(...).
    Wraps Member 5's calculate_risk() + take_action().
    Converts 0-100 dashboard scale to 0-1 scale that risk_engine.py expects.
    """
    m2_output = {
        "synthetic_probability": clone_probability / 100,
        "model_confidence": 0.9,  # replace with real model_confidence if passed in
    }
    m3_output = {
        "speaker_match_score": speaker_match_score / 100,
    }

    risk_result = calculate_risk(m2_output, m3_output)
    action_result = take_action(risk_result["risk_level"])

    return {
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "action": action_result["action"],
        "message": action_result["message"],
    }
