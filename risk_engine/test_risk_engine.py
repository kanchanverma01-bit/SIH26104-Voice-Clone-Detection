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
# MEMBER 3 - TEMPORARY TEST OUTPUT
# ==========================================
# IMPORTANT:
# Member 3 ka real output abhi nahi mila hai.
# Isliye filhaal dummy value use kar rahe hain.

m3_output = {
    "speaker_match_score": 0.30
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
# DISPLAY RESULT
# ==========================================

print("==========================================")
print("        REAL-TIME RISK ENGINE RESULT")
print("==========================================")

print("Risk Score:", risk_result["risk_score"], "%")
print("Risk Level:", risk_result["risk_level"])
print("Deepfake Risk:", risk_result["deepfake_risk"], "%")
print("Speaker Risk:", risk_result["speaker_risk"], "%")
print("Model Confidence:", risk_result["model_confidence"], "%")


print("\n==========================================")
print("           PREVENTION ACTION")
print("==========================================")

print("Action:", action_result["action"])
print("Message:", action_result["message"])


print("\n==========================================")
print("        MEMBER 2 MODEL OUTPUT")
print("==========================================")

print("Synthetic Probability:",
      m2_output["synthetic_probability"] * 100, "%")

print("Authentic Probability:",
      (1 - m2_output["synthetic_probability"]) * 100, "%")

print("Model Confidence:",
      m2_output["model_confidence"] * 100, "%")
