"""
================================================================================
 SIH26104 - AI-Powered Real-Time Detection and Prevention of
            Voice Cloning Impersonation Attacks
 MEMBER 4 - Streamlit Dashboard / UI
================================================================================

HOW TO RUN:
    pip install -r requirements.txt
    streamlit SIH UI.py

WHAT'S ALREADY DONE (by Member 4 / this script):
    - Full dashboard layout: mic input, waveform, spectrogram, score cards,
      risk banner, action status, detection history log, sidebar controls
    - Custom bluish-grey-white theme with red alert styling
    - Mock/dummy scoring engine (random but realistic-looking numbers) so the
      UI is fully clickable and demo-able RIGHT NOW, even before other
      members finish their models.
    - Session-based history table + CSV export.

WHAT YOU (the team) NEED TO PLUG IN - search for "TODO" in this file:
    1. TODO-MEMBER1  -> replace `mock_deepfake_detector()` with the real
       trained model's inference function (Member 1's output).
    2. TODO-MEMBER2  -> replace `mock_preprocess_audio()` with the real
       audio preprocessing / VAD / noise-reduction pipeline (Member 2).
    3. TODO-MEMBER3  -> replace `mock_speaker_verifier()` with the real
       speaker embedding + cosine similarity function (Member 3), and load
       real registered voice embeddings instead of the dummy dictionary.
    4. TODO-MEMBER5  -> replace `mock_risk_engine()` with the real risk
       scoring + prevention logic (Member 5).
    5. TODO-MEMBER6  -> once everything above is real, this file does NOT
       need to change structurally - Member 6 just verifies the wiring and
       runs the test cases end-to-end.

Everything below is organized so that swapping a "mock_*" function for a
real one is the ONLY change required - the UI calls a single function per
stage and doesn't care what happens inside it.
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import time
import random
from datetime import datetime

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="VoiceGuard AI | Voice Cloning Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# THEME / CUSTOM CSS  -  Bluish + Greyish + White, Red for alerts/risk
# ==============================================================================
CUSTOM_CSS = """
<style>
    /* ---------- Global palette ----------
       Primary blue   : #2F5D9F
       Deep navy      : #1B3358
       Slate grey     : #5C6B7A
       Light grey bg  : #F4F6F9
       Card white     : #FFFFFF
       Danger red     : #D93B3B
       Warning amber  : #E1A63C
       Safe green     : #3AA76D
    ------------------------------------ */

    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F4F6F9 0%, #E9EDF3 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1B3358;
    }
    section[data-testid="stSidebar"] * {
        color: #F4F6F9 !important;
    }

    /* Main title bar */
    .app-header {
        background: linear-gradient(90deg, #1B3358 0%, #2F5D9F 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(27, 51, 88, 0.25);
    }
    .app-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .app-header p {
        margin: 4px 0 0 0;
        color: #C9D6E8;
        font-size: 14px;
    }

    /* Generic card */
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(28, 51, 84, 0.08);
        border: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .card h4 {
        color: #1B3358;
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Score metric cards */
    .metric-box {
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(28,51,84,0.06);
    }
    .metric-box .label {
        font-size: 12px;
        color: #5C6B7A;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-box .value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
    }
    .val-blue   { color: #2F5D9F; }
    .val-red    { color: #D93B3B; }
    .val-amber  { color: #B9871E; }
    .val-green  { color: #2E8B57; }

    /* Risk / Alert banners */
    .alert-critical {
        background: #FDECEC;
        border-left: 6px solid #D93B3B;
        color: #7A1F1F;
        padding: 18px 22px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 14px;
    }
    .alert-medium {
        background: #FFF6E5;
        border-left: 6px solid #E1A63C;
        color: #7A5A16;
        padding: 18px 22px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 14px;
    }
    .alert-safe {
        background: #EAF7EF;
        border-left: 6px solid #3AA76D;
        color: #1F5C3B;
        padding: 18px 22px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 14px;
    }

    .action-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.03em;
    }
    .badge-block { background: #D93B3B; color: white; }
    .badge-verify { background: #E1A63C; color: white; }
    .badge-allow { background: #3AA76D; color: white; }

    /* Risk gauge bar */
    .risk-track {
        width: 100%;
        height: 14px;
        background: #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 8px;
    }
    .risk-fill {
        height: 100%;
        border-radius: 8px;
    }

    /* Buttons */
    div.stButton > button {
        background: #2F5D9F;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #1B3358;
        color: white;
    }

    /* DataFrame header */
    thead tr th {
        background-color: #1B3358 !important;
        color: white !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INIT
# ==============================================================================
if "logs" not in st.session_state:
    st.session_state.logs = []

if "registered_speakers" not in st.session_state:
    # TODO-MEMBER3: replace with real registered voice embeddings loaded from disk
    # e.g. st.session_state.registered_speakers = load_embeddings("voice_bank.pkl")
    st.session_state.registered_speakers = ["CEO", "Manager", "Finance Head", "Unknown / Not Registered"]

if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None


# ==============================================================================
# ============================  MOCK / STUB LOGIC  ============================
# Replace these functions with real module calls. UI code never needs to
# change - only what happens INSIDE these functions.
# ==============================================================================

def mock_preprocess_audio(audio_bytes):
    """
    TODO-MEMBER2: Replace this with the real pipeline:
        - noise reduction
        - voice activity detection (VAD)
        - segmentation into 3-5 sec chunks
        - resampling to model's expected sample rate (commonly 16kHz)
    Must return: (waveform: np.ndarray, sample_rate: int)
    """
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    return y, sr


def mock_deepfake_detector(y, sr):
    """
    TODO-MEMBER1: Replace with real trained model inference.
    Input : waveform (np.ndarray), sample_rate (int)
    Output: dict with 'authenticity_score' and 'clone_probability' (0-100 each,
            should sum to ~100)
    Currently: random but weighted mock so the demo looks realistic.
    """
    clone_prob = round(random.uniform(2, 97), 1)
    return {
        "authenticity_score": round(100 - clone_prob, 1),
        "clone_probability": clone_prob,
    }


def mock_speaker_verifier(y, sr, claimed_speaker):
    """
    TODO-MEMBER3: Replace with real speaker embedding + cosine similarity.
    Input : waveform, sample_rate, claimed_speaker (str)
    Output: dict with 'speaker_match_score' (0-100) and 'verified' (bool)
    """
    if claimed_speaker == "Unknown / Not Registered":
        match_score = round(random.uniform(0, 40), 1)
    else:
        match_score = round(random.uniform(15, 98), 1)
    return {
        "speaker_match_score": match_score,
        "verified": match_score >= 75,
    }


def mock_risk_engine(detection_result, speaker_result):
    """
    TODO-MEMBER5: Replace with the real risk-scoring + prevention logic.
    Input : detection_result (dict from deepfake detector),
             speaker_result (dict from speaker verifier)
    Output: dict with 'risk_level' ('LOW'/'MEDIUM'/'HIGH'/'CRITICAL'),
             'action' (str), 'risk_score' (0-100)
    Simple combined-score logic shown below as a placeholder - replace with
    your team's actual weighting / thresholds.
    """
    clone_prob = detection_result["clone_probability"]
    match_score = speaker_result["speaker_match_score"]

    # naive combined risk score (0-100): higher clone prob & lower match => higher risk
    risk_score = round((clone_prob * 0.6) + ((100 - match_score) * 0.4), 1)

    if risk_score < 30:
        level, action = "LOW", "ALLOW"
    elif risk_score < 60:
        level, action = "MEDIUM", "ADDITIONAL VERIFICATION REQUIRED"
    elif risk_score < 80:
        level, action = "HIGH", "BLOCK + VERIFY IDENTITY"
    else:
        level, action = "CRITICAL", "BLOCK + VERIFY IDENTITY"

    return {"risk_level": level, "action": action, "risk_score": risk_score}


def run_full_pipeline(audio_bytes, claimed_speaker):
    """Orchestrates the whole pipeline. This function is what Member 6 will
    eventually verify end-to-end once every mock_* is replaced with the
    real thing."""
    y, sr = mock_preprocess_audio(audio_bytes)                     # Member 2
    detection_result = real_deepfake_detector(y, sr)                # Member 1
    speaker_result = real_speaker_verifier(y, sr, claimed_speaker)  # Member 3
    risk_result = real_risk_engine(detection_result, speaker_result)  # Member 5
    return y, sr, detection_result, speaker_result, risk_result


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("""
<div class="app-header">
    <h1>🛡️ VoiceGuard AI</h1>
    <p>Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks — SIH26104</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    claimed_speaker = st.selectbox(
        "🧑 Claimed Speaker Identity",
        st.session_state.registered_speakers,
        help="Who is the speaker claiming to be? (used for speaker verification)",
    )

    st.markdown("---")
    st.markdown("### 🚦 Risk Thresholds")
    st.caption("LOW < 30  |  MEDIUM 30-60  |  HIGH 60-80  |  CRITICAL > 80")
    st.caption("TODO-MEMBER5: make these sliders control mock_risk_engine() thresholds")

    st.markdown("---")
    st.markdown("### 📋 Session")
    st.metric("Total Checks This Session", len(st.session_state.logs))
    if st.button("🗑️ Clear History"):
        st.session_state.logs = []
        st.rerun()

    st.markdown("---")
    st.caption("Member 4 — Dashboard / UI module")
    st.caption("SIH26104 · Team build")


# ==============================================================================
# MAIN LAYOUT — INPUT SECTION
# ==============================================================================
left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.markdown('<div class="card"><h4>🎙️ Live Voice Input</h4>', unsafe_allow_html=True)
    audio_value = st.audio_input("Record voice (3–5 seconds recommended)")

    uploaded_file = st.file_uploader(
        "…or upload a voice sample (.wav / .mp3)",
        type=["wav", "mp3"],
        help="Useful for testing with dataset files instead of live mic",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    audio_bytes = None
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        st.audio(audio_bytes)
    elif uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes)

    analyze_clicked = st.button("🔍 Analyze Voice", use_container_width=True, disabled=(audio_bytes is None))

with right_col:
    st.markdown('<div class="card"><h4>📈 Waveform & Spectrogram</h4>', unsafe_allow_html=True)
    waveform_placeholder = st.empty()
    spectrogram_placeholder = st.empty()
    if audio_bytes is None:
        waveform_placeholder.info("Record or upload audio to see waveform & spectrogram here.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# ANALYSIS EXECUTION
# ==============================================================================
if analyze_clicked and audio_bytes is not None:
    with st.spinner("Running detection pipeline..."):
        time.sleep(0.6)  # cosmetic delay so the spinner is visible in demo
        y, sr, detection_result, speaker_result, risk_result = run_full_pipeline(
            audio_bytes, claimed_speaker
        )

    # ---- Waveform ----
    fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#2F5D9F")
    ax_wave.set_facecolor("#FFFFFF")
    fig_wave.patch.set_facecolor("#FFFFFF")
    ax_wave.set_title("Waveform", fontsize=10, color="#1B3358")
    ax_wave.set_xlabel("")
    waveform_placeholder.pyplot(fig_wave, use_container_width=True)
    plt.close(fig_wave)

    # ---- Spectrogram ----
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig_spec, ax_spec = plt.subplots(figsize=(6, 2.3))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma") 
    fig_spec.patch.set_facecolor("#FFFFFF")
    ax_spec.set_title("Mel Spectrogram", fontsize=10, color="#1B3358")
    spectrogram_placeholder.pyplot(fig_spec, use_container_width=True)
    plt.close(fig_spec)

    # ---- Score cards ----
    st.markdown("### 🧠 Detection Scores")
    c1, c2, c3, c4 = st.columns(4)

    auth = detection_result["authenticity_score"]
    clone = detection_result["clone_probability"]
    match = speaker_result["speaker_match_score"]
    risk_score = risk_result["risk_score"]

    def val_class(score, invert=False):
        s = 100 - score if invert else score
        if s >= 70:
            return "val-green" if not invert else "val-red"
        elif s >= 40:
            return "val-amber"
        else:
            return "val-red" if not invert else "val-green"

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Authenticity</div>
            <div class="value {'val-green' if auth>=50 else 'val-red'}">{auth}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Clone Probability</div>
            <div class="value {'val-red' if clone>=50 else 'val-green'}">{clone}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Speaker Match ({claimed_speaker})</div>
            <div class="value {'val-green' if match>=75 else 'val-red'}">{match}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        risk_color = "val-red" if risk_result["risk_level"] in ("HIGH", "CRITICAL") else (
            "val-amber" if risk_result["risk_level"] == "MEDIUM" else "val-green"
        )
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Risk Score</div>
            <div class="value {risk_color}">{risk_score}</div>
        </div>""", unsafe_allow_html=True)

    # ---- Risk bar ----
    bar_color = "#D93B3B" if risk_result["risk_level"] in ("HIGH", "CRITICAL") else (
        "#E1A63C" if risk_result["risk_level"] == "MEDIUM" else "#3AA76D"
    )
    st.markdown(f"""
    <div class="risk-track">
        <div class="risk-fill" style="width:{risk_score}%; background:{bar_color};"></div>
    </div>
    <div style="text-align:right; font-size:12px; color:#5C6B7A; margin-top:4px;">
        Risk Level: <strong>{risk_result['risk_level']}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Alert banner ----
    if risk_result["risk_level"] == "CRITICAL":
        st.markdown(f"""
        <div class="alert-critical">
            🚨 VOICE CLONING ATTACK DETECTED<br>
            <span style="font-weight:400; font-size:14px;">
            Clone Probability: {clone}% &nbsp;|&nbsp; Speaker Match: {match}% &nbsp;|&nbsp; Risk: CRITICAL
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif risk_result["risk_level"] == "HIGH":
        st.markdown(f"""
        <div class="alert-critical">
            🚨 IMPERSONATION SUSPECTED<br>
            <span style="font-weight:400; font-size:14px;">
            Clone Probability: {clone}% &nbsp;|&nbsp; Speaker Match: {match}% &nbsp;|&nbsp; Risk: HIGH
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif risk_result["risk_level"] == "MEDIUM":
        st.markdown(f"""
        <div class="alert-medium">
            ⚠️ ADDITIONAL VERIFICATION REQUIRED<br>
            <span style="font-weight:400; font-size:14px;">
            Clone Probability: {clone}% &nbsp;|&nbsp; Speaker Match: {match}% &nbsp;|&nbsp; Risk: MEDIUM
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-safe">
            ✅ VOICE VERIFIED — ACCESS ALLOWED<br>
            <span style="font-weight:400; font-size:14px;">
            Clone Probability: {clone}% &nbsp;|&nbsp; Speaker Match: {match}% &nbsp;|&nbsp; Risk: LOW
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ---- Action badge ----
    action = risk_result["action"]
    if "BLOCK" in action:
        badge_class = "badge-block"
        badge_icon = "🛑"
    elif "VERIFICATION" in action:
        badge_class = "badge-verify"
        badge_icon = "🔐"
    else:
        badge_class = "badge-allow"
        badge_icon = "✅"

    st.markdown(f"""
    <p style="font-size:15px; color:#1B3358;">
        <strong>Action:</strong>
        <span class="action-badge {badge_class}">{badge_icon} {action}</span>
    </p>
    """, unsafe_allow_html=True)

    # ---- Log this detection ----
    st.session_state.logs.insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Claimed Speaker": claimed_speaker,
        "Authenticity %": auth,
        "Clone Prob %": clone,
        "Speaker Match %": match,
        "Risk Level": risk_result["risk_level"],
        "Action": action,
    })


# ==============================================================================
# DETECTION LOGS / HISTORY
# ==============================================================================
st.markdown("### 📋 Previous Detection Logs")
if len(st.session_state.logs) == 0:
    st.info("No detections yet. Record or upload a voice sample and click 'Analyze Voice'.")
else:
    df_logs = pd.DataFrame(st.session_state.logs)

    def highlight_risk(row):
        color_map = {
            "CRITICAL": "background-color:#FDECEC; color:#7A1F1F;",
            "HIGH": "background-color:#FDECEC; color:#7A1F1F;",
            "MEDIUM": "background-color:#FFF6E5; color:#7A5A16;",
            "LOW": "background-color:#EAF7EF; color:#1F5C3B;",
        }
        style = color_map.get(row["Risk Level"], "")
        return [style] * len(row)

    st.dataframe(
        df_logs.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export Logs as CSV",
        data=csv_bytes,
        file_name=f"voiceguard_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
