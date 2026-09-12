from risk_engine import calculate_risk
from prevention import take_action


# Dummy output from Member 2 (Deepfake Detection)
m2_output = {
    "synthetic_probability": 0.85,
    "model_confidence": 0.92
}


# Dummy output from Member 3 (Speaker Verification)
m3_output = {
    "speaker_match_score": 0.30
}


# Calculate overall risk
risk_result = calculate_risk(m2_output, m3_output)

# Decide security action
action_result = take_action(risk_result["risk_level"])


# Display results
print("===== RISK ENGINE RESULT =====")
print("Risk Score:", risk_result["risk_score"], "%")
print("Risk Level:", risk_result["risk_level"])
print("Deepfake Risk:", risk_result["deepfake_risk"], "%")
print("Speaker Risk:", risk_result["speaker_risk"], "%")
print("Model Confidence:", risk_result["model_confidence"], "%")

print("\n===== PREVENTION ACTION =====")
print("Action:", action_result["action"])
print("Message:", action_result["message"])
