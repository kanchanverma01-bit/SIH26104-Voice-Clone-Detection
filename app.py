import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import streamlit as st

# ============================================================
# REAL AI DETECTOR
# ============================================================
from deepfake_detection.detector import detect_voice


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="VoiceGuard | NEXORA",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "theme": "Dark",
    "voice_bytes": None,
    "voice_name": "voice_sample.wav",
    "analysis_result": None,
    "reference_bytes": None,
    "reference_name": "reference_voice.wav",
    "speaker_result": None,
    "risk_score": 0.0,
    "risk_level": "NOT ANALYZED",
    "action": "WAITING",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THEME
# ============================================================
with st.sidebar:
    st.markdown("## 🎨 Appearance")

    theme = st.selectbox(
        "Dashboard Theme",
        ["Dark", "Dim", "Light"],
        index=["Dark", "Dim", "Light"].index(st.session_state.theme),
    )

    st.session_state.theme = theme

    st.markdown("---")

    st.markdown("### 🛡️ System Status")

    st.success("● AI Detection Engine Ready")
    st.success("● Speaker Verification Ready")
    st.success("● Risk Engine Ready")
    st.success("● Prevention Layer Ready")


# ============================================================
# THEME COLORS
# ============================================================
if theme == "Dark":
    BG = "#070B14"
    CARD = "#0D1422"
    CARD2 = "#111B2D"
    TEXT = "#F8FAFC"
    MUTED = "#94A3B8"
    BORDER = "#243247"
    ACCENT = "#38BDF8"
    GREEN = "#22C55E"
    RED = "#EF4444"
    YELLOW = "#F59E0B"

elif theme == "Dim":
    BG = "#171717"
    CARD = "#222222"
    CARD2 = "#2A2A2A"
    TEXT = "#F5F5F5"
    MUTED = "#A3A3A3"
    BORDER = "#3F3F46"
    ACCENT = "#60A5FA"
    GREEN = "#4ADE80"
    RED = "#F87171"
    YELLOW = "#FBBF24"

else:
    BG = "#F5F7FB"
    CARD = "#FFFFFF"
    CARD2 = "#F1F5F9"
    TEXT = "#0F172A"
    MUTED = "#64748B"
    BORDER = "#CBD5E1"
    ACCENT = "#0284C7"
    GREEN = "#16A34A"
    RED = "#DC2626"
    YELLOW = "#D97706"


# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    f"""
<style>

html, body, [class*="css"] {{
    font-family: Inter, Arial, sans-serif;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

section[data-testid="stSidebar"] {{
    background: {CARD};
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}

.hero {{
    background: linear-gradient(
        135deg,
        {CARD} 0%,
        {CARD2} 100%
    );
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 25px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.15);
}}

.badge {{
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(56,189,248,0.12);
    color: {ACCENT};
    border: 1px solid rgba(56,189,248,0.3);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.4px;
}}

.hero-title {{
    font-size: 52px;
    font-weight: 800;
    margin-top: 18px;
    color: {TEXT};
}}

.hero-subtitle {{
    font-size: 22px;
    line-height: 1.5;
    color: {MUTED};
    max-width: 900px;
    margin-top: 8px;
}}

.small-text {{
    color: {ACCENT};
    font-size: 14px;
    font-weight: 600;
    margin-top: 15px;
}}

.section {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 28px;
    margin-top: 22px;
    margin-bottom: 22px;
}}

.section-title {{
    font-size: 25px;
    font-weight: 750;
    color: {TEXT};
    margin-bottom: 5px;
}}

.section-subtitle {{
    color: {MUTED};
    font-size: 14px;
    margin-bottom: 20px;
}}

.metric-card {{
    background: {CARD2};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
}}

.metric-label {{
    color: {MUTED};
    font-size: 13px;
    font-weight: 600;
}}

.metric-value {{
    color: {TEXT};
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}}

.pipeline {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 15px;
}}

.pipeline-item {{
    flex: 1;
    min-width: 120px;
    background: {CARD2};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px 10px;
    text-align: center;
    color: {TEXT};
    font-size: 13px;
    font-weight: 700;
}}

.arrow {{
    color: {ACCENT};
    font-size: 20px;
}}

.status-box {{
    padding: 18px;
    border-radius: 14px;
    border: 1px solid {BORDER};
    background: {CARD2};
    margin-top: 10px;
}}

.footer {{
    text-align: center;
    color: {MUTED};
    padding: 30px 10px 10px 10px;
    font-size: 13px;
}}

div.stButton > button {{
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_temp_audio(audio_bytes, filename):
    """
    Creates a temporary audio file from bytes.
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
        suffix = ".wav"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(audio_bytes)
    temp_file.close()

    return temp_file.name


def safe_percentage(value):
    """
    Converts probability to percentage safely.
    """
    try:
        value = float(value)

        if value <= 1:
            value = value * 100

        return max(0.0, min(100.0, value))

    except Exception:
        return 0.0


def extract_result_value(result, keys, default=0.0):
    """
    Safely extracts a value from detector result.
    Supports slightly different dictionary key names.
    """

    if not isinstance(result, dict):
        return default

    for key in keys:
        if key in result:
            try:
                return float(result[key])
            except Exception:
                pass

    return default


def calculate_speaker_similarity(file1, file2):
    """
    Prototype speaker similarity using MFCC features.

    NOTE:
    This is a prototype fallback.
    Replace this function with Member 3's actual
    speaker-verification model when available.
    """

    try:
        audio1, sr1 = librosa.load(
            file1,
            sr=16000,
            mono=True
        )

        audio2, sr2 = librosa.load(
            file2,
            sr=16000,
            mono=True
        )

        if len(audio1) == 0 or len(audio2) == 0:
            return 0.0

        mfcc1 = librosa.feature.mfcc(
            y=audio1,
            sr=sr1,
            n_mfcc=20
        )

        mfcc2 = librosa.feature.mfcc(
            y=audio2,
            sr=sr2,
            n_mfcc=20
        )

        vector1 = np.mean(mfcc1, axis=1)
        vector2 = np.mean(mfcc2, axis=1)

        denominator = (
            np.linalg.norm(vector1) *
            np.linalg.norm(vector2)
        )

        if denominator == 0:
            return 0.0

        similarity = np.dot(
            vector1,
            vector2
        ) / denominator

        similarity = (similarity + 1) / 2

        return float(
            max(0.0, min(1.0, similarity))
        )

    except Exception as e:
        st.error(f"Speaker verification error: {e}")
        return 0.0


def calculate_risk(
    synthetic_probability,
    speaker_similarity=None
):
    """
    Calculates final security risk.
    """

    deepfake_score = synthetic_probability

    if speaker_similarity is None:
        speaker_mismatch = 50.0
    else:
        speaker_mismatch = (
            1.0 - speaker_similarity
        ) * 100

    risk_score = (
        (deepfake_score * 0.70) +
        (speaker_mismatch * 0.30)
    )

    if (
        deepfake_score >= 70
        and speaker_mismatch >= 40
    ):
        level = "CRITICAL"
        action = "BLOCK / ESCALATE"

    elif deepfake_score >= 70:
        level = "HIGH"
        action = "SECURITY ALERT"

    elif (
        deepfake_score >= 40
        or speaker_mismatch >= 40
    ):
        level = "MEDIUM"
        action = "VERIFY / MONITOR"

    else:
        level = "LOW"
        action = "ALLOW"

    return (
        float(risk_score),
        level,
        action
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
<div class="hero">

<div class="badge">
NEXORA SECURITY SYSTEMS
</div>

<div class="hero-title">
🎙️ VoiceGuard
</div>

<div class="hero-subtitle">
Real-Time AI Detection & Prevention of
Voice Cloning Impersonation Attacks
</div>

<div class="small-text">
Detect the Fake • Verify the Identity •
Calculate Risk • Prevent the Threat
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SECURITY PIPELINE
# ============================================================
st.markdown(
    """
<div class="section">

<div class="section-title">
🔐 Security Pipeline
</div>

<div class="section-subtitle">
Multi-layer protection against AI-generated voice impersonation.
</div>

<div class="pipeline">

<div class="pipeline-item">
🎙️<br>
Live Voice
</div>

<div class="arrow">→</div>

<div class="pipeline-item">
🤖<br>
AI Detection
</div>

<div class="arrow">→</div>

<div class="pipeline-item">
👤<br>
Speaker Verification
</div>

<div class="arrow">→</div>

<div class="pipeline-item">
📊<br>
Risk Score
</div>

<div class="arrow">→</div>

<div class="pipeline-item">
🛡️<br>
Prevention
</div>

<div class="arrow">→</div>

<div class="pipeline-item">
🚨<br>
Alert
</div>

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# VOICE INPUT
# ============================================================
st.markdown(
    """
<div class="section">

<div class="section-title">
🎙️ Voice Analysis
</div>

<div class="section-subtitle">
Provide a voice sample to determine whether the speech
is authentic or potentially AI-generated.
</div>

</div>
""",
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)


with col1:

    st.markdown("### 🎤 Record Voice")

    recorded_audio = st.audio_input(
        "Record a short voice sample"
    )

    if recorded_audio is not None:

        st.session_state.voice_bytes = (
            recorded_audio.getvalue()
        )

        st.session_state.voice_name = (
            "recorded_voice.wav"
        )

        st.audio(
            st.session_state.voice_bytes,
            format="audio/wav"
        )


with col2:

    st.markdown("### 📁 Upload Audio")

    uploaded_audio = st.file_uploader(
        "Upload a voice sample",
        type=[
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "flac"
        ],
    )

    if uploaded_audio is not None:

        st.session_state.voice_bytes = (
            uploaded_audio.getvalue()
        )

        st.session_state.voice_name = (
            uploaded_audio.name
        )

        st.audio(
            st.session_state.voice_bytes
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================
if st.session_state.voice_bytes is not None:

    st.markdown("")

    analyze_button = st.button(
        "🔍  ANALYZE VOICE",
        use_container_width=True,
        type="primary",
    )

    if analyze_button:

        temp_path = None

        try:

            with st.spinner(
                "Running AI voice authenticity analysis..."
            ):

                temp_path = create_temp_audio(
                    st.session_state.voice_bytes,
                    st.session_state.voice_name
                )

                # REAL AASIST / DETECTION MODEL
                result = detect_voice(temp_path)

                st.session_state.analysis_result = result

                synthetic_probability = safe_percentage(
                    extract_result_value(
                        result,
                        [
                            "synthetic_probability",
                            "synthetic_prob",
                            "fake_probability",
                            "deepfake_probability"
                        ]
                    )
                )

                authentic_probability = safe_percentage(
                    extract_result_value(
                        result,
                        [
                            "authentic_probability",
                            "authentic_prob",
                            "real_probability"
                        ]
                    )
                )

                model_confidence = safe_percentage(
                    extract_result_value(
                        result,
                        [
                            "model_confidence",
                            "confidence"
                        ],
                        synthetic_probability / 100
                    )
                )

                # If authentic probability isn't returned,
                # calculate it from synthetic probability.
                if authentic_probability == 0:
                    authentic_probability = (
                        100 - synthetic_probability
                    )

                # If confidence is returned as zero,
                # use synthetic probability.
                if model_confidence == 0:
                    model_confidence = synthetic_probability

                # Initial risk before speaker verification
                risk_score, risk_level, action = calculate_risk(
                    synthetic_probability
                )

                st.session_state.risk_score = risk_score
                st.session_state.risk_level = risk_level
                st.session_state.action = action

            st.success(
                "Voice analysis completed successfully."
            )

        except Exception as e:

            st.error(
                f"AI Detection Error: {e}"
            )

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


# ============================================================
# AI DETECTION RESULT
# ============================================================
if st.session_state.analysis_result is not None:

    result = st.session_state.analysis_result

    synthetic_probability = safe_percentage(
        extract_result_value(
            result,
            [
                "synthetic_probability",
                "synthetic_prob",
                "fake_probability",
                "deepfake_probability"
            ]
        )
    )

    authentic_probability = safe_percentage(
        extract_result_value(
            result,
            [
                "authentic_probability",
                "authentic_prob",
                "real_probability"
            ]
        )
    )

    model_confidence = safe_percentage(
        extract_result_value(
            result,
            [
                "model_confidence",
                "confidence"
            ],
            synthetic_probability / 100
        )
    )

    if authentic_probability == 0:
        authentic_probability = (
            100 - synthetic_probability
        )

    if model_confidence == 0:
        model_confidence = synthetic_probability

    st.markdown(
        """
<div class="section">

<div class="section-title">
🤖 AI Detection Result
</div>

<div class="section-subtitle">
Deepfake detection engine output.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Synthetic Probability
</div>
<div class="metric-value">
{synthetic_probability:.2f}%
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Authentic Probability
</div>
<div class="metric-value">
{authentic_probability:.2f}%
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Model Confidence
</div>
<div class="metric-value">
{model_confidence:.2f}%
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if synthetic_probability >= 70:

        st.error(
            "🚨 HIGH PROBABILITY OF AI-GENERATED / SYNTHETIC VOICE"
        )

    elif synthetic_probability >= 40:

        st.warning(
            "⚠️ Suspicious voice characteristics detected."
        )

    else:

        st.success(
            "✅ Voice appears predominantly authentic."
        )


# ============================================================
# SPEAKER VERIFICATION
# ============================================================
if st.session_state.analysis_result is not None:

    st.markdown(
        """
<div class="section">

<div class="section-title">
👤 Speaker Verification
</div>

<div class="section-subtitle">
Compare the analyzed voice against a trusted reference voice.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    ref_col1, ref_col2 = st.columns(2)

    with ref_col1:

        reference_audio = st.audio_input(
            "Record reference voice"
        )

        if reference_audio is not None:

            st.session_state.reference_bytes = (
                reference_audio.getvalue()
            )

            st.session_state.reference_name = (
                "reference_voice.wav"
            )

            st.audio(
                st.session_state.reference_bytes,
                format="audio/wav"
            )

    with ref_col2:

        reference_upload = st.file_uploader(
            "Upload reference voice",
            type=[
                "wav",
                "mp3",
                "m4a",
                "ogg",
                "flac"
            ],
            key="reference_uploader"
        )

        if reference_upload is not None:

            st.session_state.reference_bytes = (
                reference_upload.getvalue()
            )

            st.session_state.reference_name = (
                reference_upload.name
            )

            st.audio(
                st.session_state.reference_bytes
            )

    if st.session_state.reference_bytes is not None:

        verify_button = st.button(
            "🔐  VERIFY SPEAKER",
            use_container_width=True,
        )

        if verify_button:

            analysis_temp = None
            reference_temp = None

            try:

                with st.spinner(
                    "Comparing speaker characteristics..."
                ):

                    analysis_temp = create_temp_audio(
                        st.session_state.voice_bytes,
                        st.session_state.voice_name
                    )

                    reference_temp = create_temp_audio(
                        st.session_state.reference_bytes,
                        st.session_state.reference_name
                    )

                    similarity = calculate_speaker_similarity(
                        analysis_temp,
                        reference_temp
                    )

                    st.session_state.speaker_result = (
                        similarity
                    )

                    # Recalculate final risk
                    synthetic_probability = safe_percentage(
                        extract_result_value(
                            st.session_state.analysis_result,
                            [
                                "synthetic_probability",
                                "synthetic_prob",
                                "fake_probability",
                                "deepfake_probability"
                            ]
                        )
                    )

                    risk_score, risk_level, action = calculate_risk(
                        synthetic_probability,
                        similarity
                    )

                    st.session_state.risk_score = risk_score
                    st.session_state.risk_level = risk_level
                    st.session_state.action = action

                st.success(
                    "Speaker verification completed."
                )

            except Exception as e:

                st.error(
                    f"Speaker verification failed: {e}"
                )

            finally:

                for path in [
                    analysis_temp,
                    reference_temp
                ]:

                    if (
                        path is not None
                        and os.path.exists(path)
                    ):
                        try:
                            os.remove(path)
                        except Exception:
                            pass


# ============================================================
# SPEAKER RESULT
# ============================================================
if st.session_state.speaker_result is not None:

    similarity_percentage = (
        st.session_state.speaker_result * 100
    )

    st.markdown(
        f"""
<div class="status-box">

<b>Speaker Similarity:</b>
{similarity_percentage:.2f}%

</div>
""",
        unsafe_allow_html=True,
    )

    if similarity_percentage >= 70:

        st.success(
            "✅ Speaker characteristics appear consistent with the reference voice."
        )

    else:

        st.error(
            "🚨 Speaker mismatch detected."
        )


# ============================================================
# FINAL RISK ENGINE
# ============================================================
if st.session_state.analysis_result is not None:

    st.markdown(
        """
<div class="section">

<div class="section-title">
📊 Final Risk Assessment
</div>

<div class="section-subtitle">
Combined AI detection and speaker verification assessment.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    risk_score = st.session_state.risk_score
    risk_level = st.session_state.risk_level
    action = st.session_state.action

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Overall Risk Score
</div>
<div class="metric-value">
{risk_score:.2f}%
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with r2:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Risk Level
</div>
<div class="metric-value">
{risk_level}
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with r3:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-label">
Recommended Action
</div>
<div class="metric-value">
{action}
</div>
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# PREVENTION & RESPONSE
# ============================================================
if st.session_state.analysis_result is not None:

    st.markdown(
        """
<div class="section">

<div class="section-title">
🛡️ Prevention & Response
</div>

<div class="section-subtitle">
Security action generated from the final risk assessment.
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    level = st.session_state.risk_level

    if level == "CRITICAL":

        st.error(
            "🚫 THREAT BLOCKED — High-risk voice impersonation detected. "
            "Interaction should be blocked and escalated."
        )

    elif level == "HIGH":

        st.error(
            "🚨 SECURITY ALERT — Suspicious voice detected. "
            "Additional verification is recommended."
        )

    elif level == "MEDIUM":

        st.warning(
            "⚠️ VERIFICATION REQUIRED — Voice characteristics "
            "show suspicious behavior."
        )

    else:

        st.success(
            "✅ LOW RISK — Interaction can proceed under normal monitoring."
        )


# ============================================================
# SYSTEM RESPONSE
# ============================================================
st.markdown(
    """
<div class="section">

<div class="section-title">
⚡ System Response
</div>

<div class="section-subtitle">
VoiceGuard's automated response workflow.
</div>

</div>
""",
    unsafe_allow_html=True,
)

response1, response2, response3 = st.columns(3)

with response1:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">
Detection
</div>
<div class="metric-value">
🤖 AI Scan
</div>
</div>
""",
        unsafe_allow_html=True,
    )

with response2:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">
Verification
</div>
<div class="metric-value">
👤 Identity
</div>
</div>
""",
        unsafe_allow_html=True,
    )

with response3:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">
Protection
</div>
<div class="metric-value">
🛡️ Prevention
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div class="footer">
<b>VoiceGuard • NEXORA</b><br>
AI-powered voice impersonation defence
</div>
""",
    unsafe_allow_html=True,
)