import os
import tempfile
import streamlit as st

from deepfake_detection.detector import detect_voice


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NEXORA | VoiceGuard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PROFESSIONAL BACKGROUND
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #07111f 0%,
            #0b1728 50%,
            #081321 100%
        );
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 35px;
        padding-bottom: 40px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.caption("NEXORA SECURITY SYSTEMS")

st.title("VoiceGuard")

st.write(
    "Real-Time AI Detection & Prevention of "
    "Voice Cloning Impersonation Attacks"
)

st.divider()


# =========================================================
# VOICE INPUT
# =========================================================

st.subheader("Voice Analysis")

st.caption(
    "Provide a voice sample to analyze whether the speech "
    "is authentic or potentially AI-generated."
)


input_type = st.radio(
    "Select input method",
    [
        "🎙️ Record Voice",
        "📁 Upload Audio"
    ],
    horizontal=True
)


audio_data = None
file_name = "recorded_voice.wav"


# =========================================================
# RECORD VOICE
# =========================================================

if input_type == "🎙️ Record Voice":

    st.info(
        "Record a short voice sample using your microphone."
    )

    audio_data = st.audio_input(
        "Record voice"
    )

    if audio_data is not None:

        st.success(
            "Voice sample captured successfully."
        )

        st.audio(audio_data)


# =========================================================
# UPLOAD AUDIO
# =========================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a voice sample",
        type=[
            "wav",
            "mp3",
            "ogg",
            "m4a"
        ]
    )

    if uploaded_file is not None:

        audio_data = uploaded_file
        file_name = uploaded_file.name

        st.success(
            "Audio file uploaded successfully."
        )

        st.audio(uploaded_file)


# =========================================================
# ANALYZE VOICE
# =========================================================

if audio_data is not None:

    st.write("")

    analyze_button = st.button(
        "🔍  ANALYZE VOICE",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        temp_path = None

        try:

            # -------------------------------------------------
            # CREATE TEMPORARY AUDIO FILE
            # -------------------------------------------------

            suffix = os.path.splitext(file_name)[1]

            if not suffix:
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    audio_data.getbuffer()
                )

                temp_path = temp_file.name


            # -------------------------------------------------
            # RUN AI MODEL
            # -------------------------------------------------

            with st.spinner(
                "AI model is analyzing the voice..."
            ):

                result = detect_voice(temp_path)


            synthetic = (
                result["synthetic_probability"] * 100
            )

            authentic = (
                result["authentic_probability"] * 100
            )

            confidence = (
                result["model_confidence"] * 100
            )


            # =================================================
            # DETECTION RESULTS
            # =================================================

            st.divider()

            st.subheader("Detection Results")

            st.caption(
                "AI-based voice authenticity assessment"
            )

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    label="Synthetic Probability",
                    value=f"{synthetic:.2f}%"
                )


            with col2:

                st.metric(
                    label="Authentic Probability",
                    value=f"{authentic:.2f}%"
                )


            with col3:

                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )


            # =================================================
            # RISK DECISION
            # =================================================

            st.write("")

            st.subheader("Security Decision")


            if synthetic >= 70:

                st.error(
                    "🚨 HIGH RISK — AI / CLONED VOICE DETECTED"
                )

                st.warning(
                    "Suspicious impersonation detected. "
                    "Prevention and speaker verification "
                    "should be triggered."
                )

                risk_level = "HIGH"


            elif synthetic >= 40:

                st.warning(
                    "⚠️ MEDIUM RISK — SUSPICIOUS VOICE"
                )

                st.info(
                    "Additional speaker verification "
                    "is recommended."
                )

                risk_level = "MEDIUM"


            else:

                st.success(
                    "✓ LOW RISK — VOICE APPEARS AUTHENTIC"
                )

                st.info(
                    "No immediate security intervention required."
                )

                risk_level = "LOW"


            # =================================================
            # RISK SCORE
            # =================================================

            st.write("")

            st.subheader("Risk Assessment")

            if risk_level == "HIGH":
                risk_score = synthetic

            elif risk_level == "MEDIUM":
                risk_score = synthetic

            else:
                risk_score = synthetic

            st.progress(
                min(int(risk_score), 100)
            )

            st.caption(
                f"Current AI-generated voice risk indicator: "
                f"{risk_score:.2f}%"
            )


            # =================================================
            # SECURITY PIPELINE
            # =================================================

            st.divider()

            st.subheader("Security Pipeline")

            st.caption(
                "End-to-end voice impersonation defence workflow"
            )

            pipeline = st.columns(6)

            pipeline[0].markdown("**01**\n\nVoice Input")
            pipeline[1].markdown("**02**\n\nAI Detection")
            pipeline[2].markdown("**03**\n\nSpeaker Verification")
            pipeline[3].markdown("**04**\n\nRisk Assessment")
            pipeline[4].markdown("**05**\n\nPrevention")
            pipeline[5].markdown("**06**\n\nAlert")


            # =================================================
            # FINAL RESPONSE
            # =================================================

            st.divider()

            st.subheader("System Response")

            if risk_level == "HIGH":

                st.error(
                    "SECURITY ALERT\n\n"
                    "Potential voice impersonation detected. "
                    "The interaction should be flagged and "
                    "additional identity verification initiated."
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    "VERIFICATION REQUIRED\n\n"
                    "Voice characteristics are suspicious. "
                    "Proceed with additional speaker verification."
                )

            else:

                st.success(
                    "AUTHENTICITY CHECK PASSED\n\n"
                    "The analyzed voice appears authentic "
                    "according to the AI detection model."
                )


        except Exception as e:

            st.error(
                f"Voice analysis failed: {e}"
            )

            st.exception(e)


        finally:

            # -------------------------------------------------
            # DELETE TEMPORARY FILE
            # -------------------------------------------------

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                os.remove(temp_path)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NEXORA • VoiceGuard | "
    "AI-Powered Voice Impersonation Defence Platform")