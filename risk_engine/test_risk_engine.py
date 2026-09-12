from risk_engine import calculate_risk
from prevention import take_action


# ==========================================
# MEMBER 2 - REAL DEEPFAKE DETECTION OUTPUT
# ==========================================

m2_output = {
    "synthetic_probability": 0.9574,
    "model_confidence": 0.9574
}


# ==========================================
# MEMBER 3 - REAL SPEAKER VERIFICATION OUTPUT
# ==========================================

m3_output = {
    "speaker_match_score": 0.64,
    "speaker_verified": False,
    "claimed_identity": "CEO"
}


# ==========================================
# CALCULATE OVERALL RISK
# ==========================================

risk_result = calculate_risk(
    m2_output,
    m3_output
)


# ==========================================
# DECIDE PREVENTION ACTION
# ==========================================

action_result = take_action(
    risk_result["risk_level"]
)


# ==========================================
# DISPLAY RISK ENGINE RESULT
# ==========================================

print("==========================================")
print("        REAL-TIME RISK ENGINE RESULT")
print("==========================================")

print("Risk Score:", risk_result["risk_score"], "%")
print("Risk Level:", risk_result["risk_level"])
print("Deepfake Risk:", risk_result["deepfake_risk"], "%")
print("Speaker Risk:", risk_result["speaker_risk"], "%")
print("Model Confidence:", risk_result["model_confidence"], "%")


# ==========================================
# DISPLAY PREVENTION ACTION
# ==========================================

print("\n==========================================")
print("           PREVENTION ACTION")
print("==========================================")

print("Action:", action_result["action"])
print("Message:", action_result["message"])


# ==========================================
# DISPLAY MEMBER 2 OUTPUT
# ==========================================

print("\n==========================================")
print("        MEMBER 2 MODEL OUTPUT")
print("==========================================")

print(
    "Synthetic Probability:",
    m2_output["synthetic_probability"] * 100,
    "%"
)

print(
    "Authentic Probability:",
    (1 - m2_output["synthetic_probability"]) * 100,
    "%"
)

print(
    "Model Confidence:",
    m2_output["model_confidence"] * 100,
    "%"
)


# ==========================================
# DISPLAY MEMBER 3 OUTPUT
# ==========================================

print("\n==========================================")
print("      MEMBER 3 SPEAKER OUTPUT")
print("==========================================")

print(
    "Speaker Match Score:",
    m3_output["speaker_match_score"] * 100,
    "%"
)

print(
    "Speaker Verified:",
    m3_output["speaker_verified"]
)

print(
    "Claimed Identity:",
    m3_output["claimed_identity"]
)


# ==========================================
# FINAL STATUS
# ==========================================

print("\n==========================================")
print("             FINAL STATUS")
print("==========================================")

print(
    "Risk Level:",
    risk_result["risk_level"]
)

print(
    "Security Action:",
    action_result["action"]
)

print("==========================================")
