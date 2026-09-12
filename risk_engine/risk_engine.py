def calculate_risk(m2_output, m3_output):
    """
    Combines Deepfake Detection (M2) and
    Speaker Verification (M3) outputs
    to calculate an overall risk score.
    """

    synthetic_probability = float(
        m2_output["synthetic_probability"]
    )

    model_confidence = float(
        m2_output["model_confidence"]
    )

    speaker_match_score = float(
        m3_output["speaker_match_score"]
    )

    # Initial risk calculation
    deepfake_risk = synthetic_probability
    speaker_risk = 1 - speaker_match_score

    # Combined risk score
    risk_score = (
        0.6 * deepfake_risk +
        0.4 * speaker_risk
    )

    # Convert to percentage
    risk_percentage = round(risk_score * 100, 2)

    # Risk level
    if risk_percentage >= 70:
        risk_level = "HIGH"
    elif risk_percentage >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_percentage,
        "risk_level": risk_level,
        "deepfake_risk": round(deepfake_risk * 100, 2),
        "speaker_risk": round(speaker_risk * 100, 2),
        "model_confidence": round(model_confidence * 100, 2)
    }
